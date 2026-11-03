"""Resolve which instruments to record, from Dhan's scrip master.

The option chain moves with spot: if NIFTY rallies 300 points the ATM strike
moves six strikes, and a fixed strike list silently stops covering the money.
`resolve_universe()` is therefore called at startup AND re-checked during the
session so the recorded band stays centred.

Scrip master columns (verified live):
    EXCH_ID, SEGMENT, SECURITY_ID, INSTRUMENT, UNDERLYING_SYMBOL,
    SYMBOL_NAME, DISPLAY_NAME, LOT_SIZE, SM_EXPIRY_DATE, STRIKE_PRICE,
    OPTION_TYPE
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time as dtime

import pandas as pd

log = logging.getLogger(__name__)

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv"

# Index spot lives in the IDX_I segment with these fixed ids.
SPOT_IDS = {
    "NIFTY": ("13", "IDX_I"),
    "BANKNIFTY": ("25", "IDX_I"),
}


@dataclass(frozen=True)
class Instrument:
    security_id: str
    segment: str
    symbol: str
    kind: str                 # spot | future | option
    strike: float | None = None
    option_type: str | None = None
    expiry: str | None = None
    lot_size: int | None = None

    def key(self) -> tuple[str, str]:
        return (self.segment, self.security_id)


def load_scrip_master(cache_path: str | None = None) -> pd.DataFrame:
    """Download the instrument master. ~200k rows, so cache it per day."""
    if cache_path:
        from pathlib import Path

        p = Path(cache_path)
        if p.exists() and date.fromtimestamp(p.stat().st_mtime) == date.today():
            log.info("scrip master from cache %s", p)
            return pd.read_csv(p, low_memory=False)

    log.info("downloading scrip master...")
    df = pd.read_csv(SCRIP_MASTER_URL, low_memory=False)

    if cache_path:
        from pathlib import Path

        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def _norm(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(str).str.upper().str.strip()


def nearest_future(master: pd.DataFrame, symbol: str) -> Instrument | None:
    """Current-month index future -- the deep-book target."""
    f = master[
        (_norm(master, "UNDERLYING_SYMBOL") == symbol)
        & (_norm(master, "INSTRUMENT") == "FUTIDX")
    ].copy()
    if f.empty:
        return None

    f["_exp"] = pd.to_datetime(f["SM_EXPIRY_DATE"], errors="coerce")
    f = f[f["_exp"] >= pd.Timestamp(date.today())].sort_values("_exp")
    if f.empty:
        return None

    r = f.iloc[0]
    return Instrument(
        security_id=str(r["SECURITY_ID"]),
        segment="NSE_FNO" if str(r["EXCH_ID"]).upper() == "NSE" else "BSE_FNO",
        symbol=symbol,
        kind="future",
        expiry=str(r["SM_EXPIRY_DATE"]),
        lot_size=int(r["LOT_SIZE"]) if pd.notna(r["LOT_SIZE"]) else None,
    )


def option_chain_instruments(
    master: pd.DataFrame,
    symbol: str,
    *,
    spot: float,
    n_strikes: int = 10,
    strike_step: int = 50,
) -> list[Instrument]:
    """ATM +/- n_strikes calls and puts for the nearest expiry.

    Far strikes are excluded deliberately: they are illiquid, their OI barely
    moves, and they would burn subscription slots that the money needs.
    """
    o = master[
        (_norm(master, "UNDERLYING_SYMBOL") == symbol)
        & (_norm(master, "INSTRUMENT") == "OPTIDX")
    ].copy()
    if o.empty:
        return []

    o["_exp"] = pd.to_datetime(o["SM_EXPIRY_DATE"], errors="coerce")

    # Roll past today's expiry once the session has closed. Before 15:30 the
    # expiring series is the right one to record (expiry-day gamma is exactly
    # the regime we care about); after close it is dead and the next series
    # carries the open interest.
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    cutoff = pd.Timestamp(date.today())
    if now_ist.time() > dtime(15, 30):
        cutoff = cutoff + pd.Timedelta(days=1)

    o = o[o["_exp"] >= cutoff]
    if o.empty:
        return []

    # Nearest expiry only. Weekly expiry weekdays have been revised more than
    # once by SEBI, so take whatever the master says rather than assuming.
    expiry = o["_exp"].min()
    o = o[o["_exp"] == expiry]

    atm = round(spot / strike_step) * strike_step
    lo, hi = atm - n_strikes * strike_step, atm + n_strikes * strike_step
    o = o[(o["STRIKE_PRICE"] >= lo) & (o["STRIKE_PRICE"] <= hi)]

    out: list[Instrument] = []
    for _, r in o.iterrows():
        opt_type = str(r.get("OPTION_TYPE", "")).upper()
        if opt_type not in ("CE", "PE"):
            continue
        out.append(
            Instrument(
                security_id=str(r["SECURITY_ID"]),
                segment="NSE_FNO" if str(r["EXCH_ID"]).upper() == "NSE" else "BSE_FNO",
                symbol=symbol,
                kind="option",
                strike=float(r["STRIKE_PRICE"]),
                option_type=opt_type,
                expiry=str(r["SM_EXPIRY_DATE"]),
                lot_size=int(r["LOT_SIZE"]) if pd.notna(r["LOT_SIZE"]) else None,
            )
        )
    log.info("%s: %d option legs, expiry %s, ATM %s", symbol, len(out), expiry.date(), atm)
    return out


def resolve_universe(
    master: pd.DataFrame,
    *,
    symbols: list[str],
    spots: dict[str, float],
    n_strikes: int = 10,
    strike_steps: dict[str, int] | None = None,
) -> list[Instrument]:
    """Full recording universe: spot + nearest future + option band per symbol."""
    strike_steps = strike_steps or {"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100}
    universe: list[Instrument] = []

    for sym in symbols:
        if sym in SPOT_IDS:
            sid, seg = SPOT_IDS[sym]
            universe.append(Instrument(security_id=sid, segment=seg, symbol=sym, kind="spot"))

        fut = nearest_future(master, sym)
        if fut:
            universe.append(fut)

        spot = spots.get(sym)
        if spot:
            universe.extend(
                option_chain_instruments(
                    master, sym, spot=spot,
                    n_strikes=n_strikes,
                    strike_step=strike_steps.get(sym, 50),
                )
            )
        else:
            log.warning("%s: no spot price given, skipping option chain", sym)

    return universe


def universe_frame(universe: list[Instrument]) -> pd.DataFrame:
    """Instrument metadata, written alongside each day's ticks.

    Without this the recorded security ids are meaningless later -- ids get
    reused across expiries, so the mapping must be captured at record time.
    """
    return pd.DataFrame(
        [
            {
                "security_id": i.security_id,
                "segment": i.segment,
                "symbol": i.symbol,
                "kind": i.kind,
                "strike": i.strike,
                "option_type": i.option_type,
                "expiry": i.expiry,
                "lot_size": i.lot_size,
            }
            for i in universe
        ]
    )
