"""Paper trading: forward-test the model on live data. NO ORDERS, NO MONEY.

    python scripts/paper_trade.py --model models/nifty_50_33.joblib

Why this exists: a backtest can only tell you the model fits the past. It
cannot tell you whether the SAME code, fed a live feed, produces the same
signals. The usual way strategies die is online/offline feature skew -- the
live feature values drift from the researched ones, and live P&L stops
matching the backtest without anyone noticing which half is broken.

This runs the identical feature code (mcube.features.bars) over a rolling
buffer, so any divergence is a real bug rather than a reimplementation
difference.

What it does each minute:
    1. aggregate live ticks into a 1-minute bar
    2. append to a buffer seeded with real history (features need 240 bars
       of lookback, so a cold start would emit NaNs until ~13:15)
    3. compute features, predict P(target before stop)
    4. open a simulated trade above the confidence threshold
    5. track open trades against target / stop / timeout
    6. charge full India costs on every close

Everything is logged to data/paper/dt=YYYY-MM-DD/ for later comparison
against what the backtest predicted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcube.collector.feed import LiveFeed
from mcube.collector.instruments import Instrument
from mcube.costs.india_costs import cost_in_index_points
from mcube.data.dhan import DhanClient
from mcube.data.dhan_auth import TokenManager
from mcube.features.bars import bar_features, session_features

IST = ZoneInfo("Asia/Kolkata")
SPOT_IDS = {"NIFTY": "13", "BANKNIFTY": "25"}
SEED_BARS = 400  # > the 240-bar lookback that rv_240 needs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("paper")


@dataclass
class PaperTrade:
    entry_time: str
    side: int
    entry_px: float
    target_px: float
    stop_px: float
    deadline: str
    confidence: float
    exit_time: str | None = None
    exit_px: float | None = None
    exit_reason: str | None = None
    gross_pts: float | None = None
    cost_pts: float | None = None
    net_pts: float | None = None
    net_rs: float | None = None

    @property
    def is_open(self) -> bool:
        return self.exit_time is None


class PaperTrader:
    def __init__(self, bundle: dict, cfg: dict, seed: pd.DataFrame):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.symbol = bundle["symbol"]
        self.target = float(bundle["target"])
        self.stop = float(bundle["stop"])
        self.horizon = int(bundle["horizon_bars"])
        self.cfg = cfg
        self.threshold = cfg["model"]["meta_labeling"]["min_confidence"]

        self.bars = seed.copy()
        self.trades: list[PaperTrade] = []
        self.signals_seen = 0
        self.last_proba: float | None = None
        self.gap_detected = False

        lot = cfg["instruments"][self.symbol]["lot_size"]
        self.lots = cfg["trade"]["lots"]
        self.lot_size = lot

        log.info(
            "model: %s %.0f/%.0f pts, %d-bar horizon | CV AUC %.4f | %d features",
            self.symbol, self.target, self.stop, self.horizon,
            bundle.get("cv_auc", float("nan")), len(self.features),
        )
        log.info("seeded with %d historical bars", len(seed))

    # ----------------------------------------------------------------- bars

    def add_bar(self, ts: pd.Timestamp, o: float, h: float, l: float, c: float, v: float) -> None:
        # A feed gap must not be stitched into the buffer as if the bars were
        # consecutive: rolling windows would then span the hole and every
        # feature would be quietly wrong. Flag it so the caller re-seeds.
        if len(self.bars):
            gap_min = (ts - self.bars.index[-1]).total_seconds() / 60
            if gap_min > 5:
                self.gap_detected = True
                log.warning(
                    "bar gap of %.0f min (%s -> %s). Buffer is stale; "
                    "predictions suspended until re-seeded.",
                    gap_min, self.bars.index[-1].strftime("%H:%M"), ts.strftime("%H:%M"),
                )

        self.bars.loc[ts] = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        self.bars = self.bars.tail(SEED_BARS + 400)

    # ------------------------------------------------------------- predict

    def predict(self) -> float | None:
        """P(target before stop) for the latest bar, or None if not ready."""
        if len(self.bars) < 250 or self.gap_detected:
            return None

        X = pd.concat(
            [bar_features(self.bars), session_features(self.bars.index)], axis=1
        ).replace([np.inf, -np.inf], np.nan)

        # Reindex to the EXACT training column order. A missing or reordered
        # column silently changes predictions, which is the whole failure mode
        # this script exists to catch.
        missing = [f for f in self.features if f not in X.columns]
        if missing:
            log.warning("features missing live: %s", missing[:5])
        X = X.reindex(columns=self.features)

        row = X.iloc[[-1]]
        if row.isna().all(axis=1).iloc[0]:
            return None
        return float(self.model.predict_proba(row)[0, 1])

    # --------------------------------------------------------------- trade

    def step(self, ts: pd.Timestamp, price: float) -> None:
        self._update_open(ts, price)

        # No new entries in the closing window -- a trade opened at 15:28
        # cannot reach a 45-minute barrier.
        minutes_left = (
            datetime.combine(ts.date(), dtime(15, 30)) - ts.to_pydatetime().replace(tzinfo=None)
        ).total_seconds() / 60
        if minutes_left < self.horizon:
            return

        if any(t.is_open for t in self.trades):
            return  # one position at a time, as the backtest assumed

        proba = self.predict()
        if proba is None:
            return
        self.last_proba = proba

        if proba > self.threshold:
            side = 1
        elif proba < 1 - self.threshold:
            side = -1
        else:
            return

        self.signals_seen += 1
        t = PaperTrade(
            entry_time=ts.isoformat(),
            side=side,
            entry_px=price,
            target_px=price + self.target * side,
            stop_px=price - self.stop * side,
            deadline=(ts + pd.Timedelta(minutes=self.horizon)).isoformat(),
            confidence=proba,
        )
        self.trades.append(t)
        log.info(
            "ENTRY %s @ %.2f  target %.2f  stop %.2f  conf %.3f",
            "LONG" if side > 0 else "SHORT", price, t.target_px, t.stop_px, proba,
        )

    def expire_stale(self, now: pd.Timestamp) -> None:
        """Close trades whose deadline has passed, on WALL CLOCK.

        Exit checks are otherwise driven by bar arrival, so a feed outage would
        leave a trade open indefinitely -- past its stop and past its deadline.
        That happened on the first live run: a short blew through its stop
        during a 2h16m network drop and was still logged as open.

        A trade expired during an outage has an unknown true outcome, so it is
        marked `outage` rather than scored. Counting it as a win or a loss
        would be inventing data.
        """
        for t in self.trades:
            if t.is_open and now >= pd.Timestamp(t.deadline):
                self._close(t, now, t.entry_px, "outage")
                log.warning(
                    "trade from %s expired with no price updates -- feed gap. "
                    "Outcome unknown, excluded from hit rate.", t.entry_time,
                )

    def _update_open(self, ts: pd.Timestamp, price: float) -> None:
        for t in self.trades:
            if not t.is_open:
                continue

            hit_target = price >= t.target_px if t.side > 0 else price <= t.target_px
            hit_stop = price <= t.stop_px if t.side > 0 else price >= t.stop_px
            expired = ts >= pd.Timestamp(t.deadline)

            if hit_target:
                self._close(t, ts, t.target_px, "target")
            elif hit_stop:
                self._close(t, ts, t.stop_px, "stop")
            elif expired:
                self._close(t, ts, price, "timeout")

    def _close(self, t: PaperTrade, ts: pd.Timestamp, px: float, reason: str) -> None:
        cost = cost_in_index_points(
            symbol=self.symbol, spot=t.entry_px, cfg=self.cfg,
            instrument=self.cfg["trade"]["instrument"], lots=self.lots,
        )
        gross = (px - t.entry_px) * t.side
        t.exit_time = ts.isoformat()
        t.exit_px = px
        t.exit_reason = reason
        t.gross_pts = gross
        t.cost_pts = cost
        t.net_pts = gross - cost
        # Options P&L moves with delta, so index points convert before rupees.
        delta = self.cfg["backtest"]["atm_delta"] if self.cfg["trade"]["instrument"] == "options" else 1.0
        t.net_rs = t.net_pts * delta * self.lot_size * self.lots

        log.info(
            "EXIT  %-7s @ %.2f  gross %+.1f  cost %.1f  net %+.1f pts (Rs %+,.0f)",
            reason, px, gross, cost, t.net_pts, t.net_rs,
        )

    # -------------------------------------------------------------- report

    def summary(self) -> dict:
        closed = [t for t in self.trades if not t.is_open]
        # Trades that expired during a feed outage have an UNKNOWN outcome.
        # Scoring them either way would be inventing data.
        outages = [t for t in closed if t.exit_reason == "outage"]
        closed = [t for t in closed if t.exit_reason != "outage"]
        if not closed:
            return {"trades": 0, "open": sum(t.is_open for t in self.trades),
                    "excluded_outage": len(outages)}

        net = np.array([t.net_rs for t in closed])
        wins = net > 0
        return {
            "trades": len(closed),
            "open": sum(t.is_open for t in self.trades),
            "excluded_outage": len(outages),
            "hit_rate_pct": round(100 * wins.mean(), 1),
            "net_rs": round(float(net.sum())),
            "avg_rs": round(float(net.mean())),
            "total_cost_rs": round(sum(t.cost_pts for t in closed) * self.lot_size * self.lots),
            "exits": pd.Series([t.exit_reason for t in closed]).value_counts().to_dict(),
        }

    def save(self, root: str = "data/paper") -> Path | None:
        if not self.trades:
            return None
        now = datetime.now(IST)
        d = Path(root) / f"dt={now:%Y-%m-%d}"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"trades_{now:%H%M%S}.parquet"
        pd.DataFrame([asdict(t) for t in self.trades]).to_parquet(path, index=False)
        return path


class BarBuilder:
    """Aggregate ticks into 1-minute bars for one instrument."""

    def __init__(self, security_id: int):
        self.sid = security_id
        self.cur_min: pd.Timestamp | None = None
        self.o = self.h = self.l = self.c = None
        self.v0 = self.v1 = None

    def update(self, row: dict) -> tuple | None:
        """Returns a completed bar when the minute rolls over."""
        if row["security_id"] != self.sid or not row.get("ltp"):
            return None

        ts = pd.Timestamp(row["recv_ns"], unit="ns", tz="UTC").tz_convert(IST)
        minute = ts.floor("1min")
        px = float(row["ltp"])
        vol = row.get("volume")

        if self.cur_min is None:
            self.cur_min, self.o = minute, px
            self.h = self.l = self.c = px
            self.v0 = self.v1 = vol
            return None

        if minute > self.cur_min:
            # `volume` is cumulative for the session, so a bar's own volume is
            # the difference across the minute, not a sum.
            bar = (self.cur_min, self.o, self.h, self.l, self.c,
                   (self.v1 - self.v0) if (self.v0 is not None and self.v1 is not None) else 0.0)
            self.cur_min, self.o = minute, px
            self.h = self.l = self.c = px
            self.v0 = self.v1 = vol
            return bar

        self.h = max(self.h, px)
        self.l = min(self.l, px)
        self.c = px
        if vol is not None:
            self.v1 = vol
        return None


async def main_async(args) -> None:
    import joblib

    bundle = joblib.load(args.model)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    symbol = bundle["symbol"]

    if os.environ.get("DHAN_TOTP_SECRET"):
        tm = TokenManager.from_env()
        provider = lambda force=False: tm.force_refresh() if force else tm.token()
    else:
        tok = os.environ.get("DHAN_ACCESS_TOKEN")
        if not tok:
            raise SystemExit("Set DHAN_ACCESS_TOKEN or DHAN_TOTP_SECRET")
        provider = lambda force=False: tok

    # Seed the buffer with real history: features need 240 bars of lookback,
    # so a cold start would emit nothing usable until mid-afternoon.
    client = DhanClient()
    now = datetime.now(IST)
    seed = await asyncio.to_thread(
        client.intraday,
        security_id=SPOT_IDS[symbol], exchange_segment="IDX_I", instrument="INDEX",
        interval=1,
        from_date=(now - pd.Timedelta(days=10)).strftime("%Y-%m-%d 09:15:00"),
        to_date=now.strftime("%Y-%m-%d 15:30:00"), oi=False,
    )
    seed = seed[["open", "high", "low", "close", "volume"]].tail(SEED_BARS)

    trader = PaperTrader(bundle, cfg, seed)
    builder = BarBuilder(int(SPOT_IDS[symbol]))

    def on_ticks(rows: list[dict]) -> None:
        for row in rows:
            bar = builder.update(row)
            if bar is None:
                continue
            ts, o, h, l, c, v = bar
            trader.add_bar(ts, o, h, l, c, v)
            trader.step(ts, c)

    universe = [Instrument(security_id=SPOT_IDS[symbol], segment="IDX_I",
                           symbol=symbol, kind="spot")]
    feed = LiveFeed(instruments=universe, token_provider=provider,
                    client_id=os.environ["DHAN_CLIENT_ID"], on_ticks=on_ticks)

    async def reseed() -> None:
        """Refetch history to repair the buffer after a feed gap."""
        try:
            now = datetime.now(IST)
            fresh = await asyncio.to_thread(
                client.intraday,
                security_id=SPOT_IDS[symbol], exchange_segment="IDX_I",
                instrument="INDEX", interval=1,
                from_date=(now - pd.Timedelta(days=10)).strftime("%Y-%m-%d 09:15:00"),
                to_date=now.strftime("%Y-%m-%d 15:30:00"), oi=False,
            )
            if not fresh.empty:
                trader.bars = fresh[["open", "high", "low", "close", "volume"]].tail(SEED_BARS)
                trader.gap_detected = False
                log.info("buffer re-seeded with %d bars; predictions resumed", len(trader.bars))
        except Exception as e:
            log.warning("re-seed failed (%s); will retry", str(e)[:100])

    async def report():
        while True:
            await asyncio.sleep(args.report_every)

            # Wall-clock expiry, independent of bar arrival. Without this a
            # feed outage leaves trades open past their deadline forever.
            trader.expire_stale(pd.Timestamp.now(tz=IST))

            if trader.gap_detected:
                await reseed()

            s = trader.summary()
            conf = f"{trader.last_proba:.3f}" if trader.last_proba is not None else "--"
            stale = " [BUFFER STALE]" if trader.gap_detected else ""
            log.info("bars=%d signals=%d last_p=%s%s | %s",
                     len(trader.bars), trader.signals_seen, conf, stale, s)
            trader.save()

    try:
        await asyncio.gather(feed.run(), report())
    finally:
        path = trader.save()
        print("\n" + "=" * 64)
        print("  PAPER TRADING SUMMARY  (no money at risk)")
        print("=" * 64)
        for k, v in trader.summary().items():
            print(f"  {k:<16} {v}")
        if path:
            print(f"  saved -> {path}")
        print("=" * 64)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Paper trade the model live (no orders)")
    ap.add_argument("--model", type=Path, default=Path("models/nifty_50_33.joblib"))
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--report-every", type=float, default=300.0)
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
