"""Download NIFTY history from Dhan and save as parquet.

Dhan serves ~5 years of 1-minute candles WITH open interest, so the model can
be trained on real tape immediately -- no waiting for a collector to accumulate
data. (Intraday strike-level option-chain OI is snapshot-only and still has to
be recorded live; that is a separate job.)

Setup:
    pip install requests pyarrow
    set DHAN_CLIENT_ID=100xxxxxxx
    set DHAN_ACCESS_TOKEN=eyJ...

Run:
    python scripts/fetch_history.py --years 3
    python scripts/fetch_history.py --list-ids      # find security IDs only
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcube.data.dhan import DhanClient, DhanError, load_scrip_master

OUT_DIR = Path("data/raw")

# Verified against the live Dhan scrip master.
# Index spot lives in segment "I" (IDX_I); derivatives in NSE_FNO / BSE_FNO.
SPOT = {
    "NIFTY": {"security_id": "13", "segment": "IDX_I", "instrument": "INDEX"},
    "BANKNIFTY": {"security_id": "25", "segment": "IDX_I", "instrument": "INDEX"},
}


def list_ids() -> None:
    """Print current futures contracts and their security IDs."""
    m = load_scrip_master()
    for sym in ("NIFTY", "BANKNIFTY", "SENSEX"):
        f = m[
            (m.UNDERLYING_SYMBOL.astype(str).str.upper() == sym)
            & (m.INSTRUMENT.astype(str).str.upper() == "FUTIDX")
        ].sort_values("SM_EXPIRY_DATE")
        print(f"\n=== {sym} futures ===")
        if f.empty:
            print("  none found")
            continue
        print(
            f[["SECURITY_ID", "SYMBOL_NAME", "SM_EXPIRY_DATE", "LOT_SIZE", "EXCH_ID"]]
            .head(3)
            .to_string(index=False)
        )


def fetch(symbol: str, years: int, interval: int) -> pd.DataFrame:
    client = DhanClient()
    spec = SPOT[symbol]

    end = datetime.now()
    start = end - timedelta(days=365 * years)

    print(f"  fetching {symbol} {interval}min from {start:%Y-%m-%d} to {end:%Y-%m-%d}...")
    df = client.intraday(
        security_id=spec["security_id"],
        exchange_segment=spec["segment"],
        instrument=spec["instrument"],
        interval=interval,
        from_date=start.strftime("%Y-%m-%d 09:15:00"),
        to_date=end.strftime("%Y-%m-%d 15:30:00"),
        oi=True,
    )
    return df


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--interval", type=int, default=1, choices=[1, 5, 15, 25, 60])
    ap.add_argument("--symbols", nargs="+", default=["NIFTY"])
    ap.add_argument("--list-ids", action="store_true")
    args = ap.parse_args()

    if args.list_ids:
        list_ids()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sym in args.symbols:
        try:
            df = fetch(sym, args.years, args.interval)
        except DhanError as e:
            print(f"  ERROR {sym}: {e}")
            continue

        if df.empty:
            print(f"  {sym}: no data returned")
            continue

        path = OUT_DIR / f"{sym}_{args.interval}min.parquet"
        df.to_parquet(path)

        sessions = df.index.normalize().nunique()
        print(f"  {sym}: {len(df):,} bars across {sessions:,} sessions -> {path}")
        print(f"         {df.index.min()} .. {df.index.max()}")
        # OI is absent on the spot index but present on derivatives -- worth
        # knowing which you actually got before building features on it.
        has_oi = df["open_interest"].notna().any() and (df["open_interest"].fillna(0) != 0).any()
        print(f"         open interest present: {has_oi}")


if __name__ == "__main__":
    main()
