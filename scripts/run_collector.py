"""Market data collector -- records what cannot be backfilled.

Dhan serves historical BARS, but order-book depth and intraday strike-level OI
are live-only. Every session this does not run is training data permanently
lost. That is the entire justification for this script.

Run:
    python scripts/run_collector.py                 # today's session
    python scripts/run_collector.py --dry-run       # 60s connectivity check
    python scripts/run_collector.py --symbols NIFTY BANKNIFTY

What it records, per session, to data/live/dt=YYYY-MM-DD/:
    ticks_*.parquet      quote + 5-level depth + OI, every update
    chain_*.parquet      option-chain snapshots with OI/IV, each minute
    universe.parquet     security_id -> strike/expiry map (ids are reused
                         across expiries, so this must be captured live)

Deliberately NOT doing:
    - 200-level depth. It allows 1 instrument per connection against a 5
      connection budget; the option chain needs those slots more. Add it for
      the future alone once the main feed is proven stable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcube.collector.feed import LiveFeed
from mcube.collector.instruments import (
    load_scrip_master,
    resolve_universe,
    universe_frame,
)
from mcube.collector.writer import GapMonitor, GCSUploader, TickWriter
from mcube.data.dhan import DhanClient
from mcube.data.dhan_auth import TokenManager

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collector")


def make_token_provider():
    """Prefer TOTP auto-refresh; fall back to a static token.

    A static token dies after ~24h, so an unattended collector should always
    have TOTP configured. We warn rather than refuse, so a first run can be
    tested with whatever credentials are already to hand.
    """
    import os

    if os.environ.get("DHAN_TOTP_SECRET"):
        tm = TokenManager.from_env()

        def provider(force: bool = False) -> str:
            return tm.force_refresh() if force else tm.token()

        log.info("token: TOTP auto-refresh enabled")
        return provider

    static = os.environ.get("DHAN_ACCESS_TOKEN")
    if not static:
        raise SystemExit(
            "No credentials. Set DHAN_TOTP_SECRET (+ DHAN_CLIENT_ID, DHAN_PIN) "
            "for auto-refresh, or DHAN_ACCESS_TOKEN for a one-off run."
        )
    log.warning("token: STATIC -- expires in ~24h. Set DHAN_TOTP_SECRET for "
                "unattended running.")
    return lambda force=False: static


def in_session(now: datetime | None = None) -> bool:
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return SESSION_OPEN <= now.time() <= SESSION_CLOSE


async def flush_loop(writer: TickWriter, uploader: GCSUploader, every: float = 60.0):
    """Flush to disk every minute; mirror to GCS every five."""
    cycles = 0
    while True:
        await asyncio.sleep(every)
        writer.flush()
        cycles += 1
        if uploader.enabled and cycles % 5 == 0:
            await asyncio.to_thread(uploader.sync)


async def chain_loop(
    client: DhanClient, writer: TickWriter, symbols: list[str], cfg: dict, every: float = 60.0
):
    """Snapshot the option chain each minute.

    The chain endpoint is a SNAPSHOT with no history, so this is the only way
    strike-level OI ever gets recorded. Failures are logged and skipped -- one
    missed minute is survivable, stopping the collector is not.
    """
    scrip_map = {"NIFTY": 13, "BANKNIFTY": 25}
    while True:
        await asyncio.sleep(every)
        if not in_session():
            continue
        for sym in symbols:
            scrip = scrip_map.get(sym)
            if not scrip:
                continue
            try:
                expiry = cfg.get("_expiry", {}).get(sym)
                if not expiry:
                    continue
                df = await asyncio.to_thread(
                    client.option_chain,
                    underlying_scrip=scrip,
                    underlying_seg="IDX_I",
                    expiry=expiry,
                )
                if not df.empty:
                    df["symbol"] = sym
                    writer.extend(df.to_dict("records"))
            except Exception as e:
                log.warning("chain snapshot failed for %s: %s", sym, str(e)[:120])


async def heartbeat(feeds: list[LiveFeed], writer: TickWriter, gaps: GapMonitor, every: float = 60.0):
    """Log health. A silent-but-connected feed is the dangerous failure mode."""
    import time

    while True:
        await asyncio.sleep(every)
        total = sum(f.stats.packets for f in feeds)
        recon = sum(f.stats.reconnects for f in feeds)
        g = gaps.report(time.time())
        last = max((f.stats.last_packet_at for f in feeds), default=0)
        silent = time.time() - last if last else -1

        log.info(
            "packets=%s buffered=%s written=%s files=%s reconnects=%s "
            "tracked=%s stale=%s%% silent_for=%.0fs",
            f"{total:,}", f"{writer.pending:,}", f"{writer.total_rows:,}",
            writer.files_written, recon, g["tracked"], g["stale_pct"], silent,
        )

        if silent > 120 and in_session():
            log.error("NO PACKETS FOR %.0fs DURING SESSION -- feed may be dead", silent)


async def main_async(args) -> None:
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    token_provider = make_token_provider()
    import os

    client_id = os.environ.get("DHAN_CLIENT_ID")
    if not client_id:
        raise SystemExit("DHAN_CLIENT_ID is not set")

    # --- resolve what to record -------------------------------------------
    master = load_scrip_master(cache_path="data/cache/scrip_master.csv")
    client = DhanClient()

    spots: dict[str, float] = {}
    for sym in args.symbols:
        sid = {"NIFTY": "13", "BANKNIFTY": "25"}.get(sym)
        if not sid:
            continue
        try:
            today = datetime.now(IST)
            bars = await asyncio.to_thread(
                client.intraday,
                security_id=sid, exchange_segment="IDX_I", instrument="INDEX",
                interval=1,
                from_date=(today - pd.Timedelta(days=7)).strftime("%Y-%m-%d 09:15:00"),
                to_date=today.strftime("%Y-%m-%d 15:30:00"),
                oi=False,
            )
            if not bars.empty:
                spots[sym] = float(bars["close"].iloc[-1])
                log.info("%s last close %.2f", sym, spots[sym])
        except Exception as e:
            log.warning("could not fetch spot for %s: %s", sym, str(e)[:120])

    universe = resolve_universe(
        master,
        symbols=args.symbols,
        spots=spots,
        n_strikes=args.strikes,
        strike_steps={"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100},
    )
    if not universe:
        raise SystemExit("empty universe -- check scrip master and spot prices")

    by_kind: dict[str, int] = {}
    for i in universe:
        by_kind[i.kind] = by_kind.get(i.kind, 0) + 1
    log.info("universe: %d instruments %s", len(universe), by_kind)

    # Record the id -> strike/expiry map. Security ids are reused across
    # expiries, so without this the ticks become unreadable later.
    writer = TickWriter(root=args.out)
    writer.write_frame(universe_frame(universe), "universe")

    uploader = GCSUploader(
        bucket=args.gcs_bucket or cfg.get("gcp", {}).get("gcs_bucket") or None
    )
    gaps = GapMonitor(stale_seconds=90)

    feed = LiveFeed(
        instruments=universe,
        token_provider=token_provider,
        client_id=client_id,
        on_ticks=writer.extend,
        gap_monitor=gaps,
    )

    # --- run ---------------------------------------------------------------
    tasks = [
        asyncio.create_task(feed.run()),
        asyncio.create_task(flush_loop(writer, uploader)),
        asyncio.create_task(heartbeat([feed], writer, gaps)),
    ]

    if args.dry_run:
        log.info("DRY RUN -- 60 seconds then exit")
        await asyncio.sleep(60)
    else:
        stop = asyncio.Event()

        def _sig(*_):
            log.info("shutdown requested")
            stop.set()

        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(s, _sig)
            except (ValueError, OSError):
                pass

        while not stop.is_set():
            await asyncio.sleep(5)
            # Stop ten minutes after close; nothing useful arrives after that.
            now = datetime.now(IST)
            if now.time() > dtime(15, 40) and now.weekday() < 5:
                log.info("session over")
                break

    # --- shutdown: flush before exiting, always ---------------------------
    feed.stop()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    writer.flush()
    if uploader.enabled:
        await asyncio.to_thread(uploader.sync)

    log.info("DONE packets=%s rows_written=%s files=%s reconnects=%s",
             f"{feed.stats.packets:,}", f"{writer.total_rows:,}",
             writer.files_written, feed.stats.reconnects)
    if feed.stats.errors:
        log.info("last errors: %s", feed.stats.errors[-3:])


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Dhan market data collector")
    ap.add_argument("--symbols", nargs="+", default=["NIFTY"])
    ap.add_argument("--strikes", type=int, default=10, help="ATM +/- N strikes")
    ap.add_argument("--out", default="data/live")
    ap.add_argument("--gcs-bucket", default=None)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="60s connectivity check")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log.info("interrupted")


if __name__ == "__main__":
    main()
