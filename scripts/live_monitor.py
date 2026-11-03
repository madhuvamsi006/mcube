"""Live institutional-flow monitor. READ ONLY -- places no orders.

    python scripts/live_monitor.py --symbols NIFTY

What this is: a window onto the signals the collector is recording, so you can
watch them form during the session instead of reading a log counter.

What this is NOT: a validated prediction. Nothing here has been backtested in
live form. The measured edge from historical bars is +9pp over base rate, which
is real but not enough to clear costs at short targets. Treat every reading
below as a hypothesis under observation, not a trade signal.

The core idea it displays, specific to Indian index options:

    Retail buys options. Institutions write them.

So where open interest BUILDS marks where institutions believe price will not
go:

    PE OI rising   -> puts being written  -> support  -> bullish
    CE OI rising   -> calls being written -> resistance -> bearish
    OI falling on a move -> covering, the move is exhausting rather than starting

Run it alongside the collector (separate terminal). It uses its own websocket
connection, which counts against Dhan's 5-connection limit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcube.collector.feed import LiveFeed
from mcube.collector.instruments import load_scrip_master, resolve_universe
from mcube.data.dhan import DhanClient
from mcube.data.dhan_auth import TokenManager

IST = ZoneInfo("Asia/Kolkata")
HIST = 30  # minutes of OI history to keep


class FlowState:
    """Rolling institutional-flow state, updated from the live feed."""

    def __init__(self, universe):
        self.meta = {int(i.security_id): i for i in universe}
        self.oi: dict[int, deque] = defaultdict(lambda: deque(maxlen=HIST * 2))
        self.quote: dict[int, dict] = {}
        self.spot: float | None = None
        self.future: float | None = None

    def update(self, row: dict) -> None:
        sid = row["security_id"]
        inst = self.meta.get(sid)
        if inst is None:
            return

        if row.get("oi") is not None:
            self.oi[sid].append((row["recv_ns"], row["oi"]))

        if row.get("ltp"):
            self.quote[sid] = row
            if inst.kind == "spot":
                self.spot = row["ltp"]
            elif inst.kind == "future":
                self.future = row["ltp"]

    def _doi(self, sid: int, minutes: int) -> float | None:
        """OI change over the last N minutes, or None if not enough history."""
        hist = self.oi.get(sid)
        if not hist or len(hist) < 2:
            return None
        cutoff = hist[-1][0] - minutes * 60 * 1_000_000_000
        older = [v for t, v in hist if t <= cutoff]
        if not older:
            older = [hist[0][1]]
        return hist[-1][1] - older[-1]

    def writing_bias(self, minutes: int = 5) -> dict:
        """Net option-writing pressure across the recorded strike band.

        Positive = put writing dominates = institutions defending downside.
        """
        d_ce = d_pe = 0.0
        for sid, inst in self.meta.items():
            if inst.kind != "option":
                continue
            d = self._doi(sid, minutes)
            if d is None:
                continue
            if inst.option_type == "CE":
                d_ce += d
            else:
                d_pe += d

        denom = abs(d_ce) + abs(d_pe)
        return {
            "d_ce": d_ce,
            "d_pe": d_pe,
            "bias": (d_pe - d_ce) / denom if denom else 0.0,
        }

    def oi_walls(self, n: int = 3) -> tuple[list, list]:
        """Strikes carrying the most OI -- institutional support and resistance.

        The largest PE OI below spot is where writers are defending; the
        largest CE OI above is where they are capping.
        """
        ce, pe = [], []
        for sid, inst in self.meta.items():
            if inst.kind != "option" or not self.oi.get(sid):
                continue
            latest = self.oi[sid][-1][1]
            entry = (inst.strike, latest, self._doi(sid, 5) or 0.0)
            (ce if inst.option_type == "CE" else pe).append(entry)

        ce.sort(key=lambda x: -x[1])
        pe.sort(key=lambda x: -x[1])
        return ce[:n], pe[:n]

    def book(self) -> dict | None:
        """Order-book imbalance on the future -- the short-horizon pressure read."""
        for sid, inst in self.meta.items():
            if inst.kind != "future":
                continue
            q = self.quote.get(sid)
            if not q or not q.get("bid_px_1"):
                continue

            bid_t = sum(q.get(f"bid_qty_{i}") or 0 for i in range(1, 6))
            ask_t = sum(q.get(f"ask_qty_{i}") or 0 for i in range(1, 6))
            tb, ts = q.get("total_buy_qty") or 0, q.get("total_sell_qty") or 0
            return {
                "ltp": q["ltp"],
                "spread": (q.get("ask_px_1") or 0) - q["bid_px_1"],
                "imb_l1": (
                    ((q.get("bid_qty_1") or 0) - (q.get("ask_qty_1") or 0))
                    / max((q.get("bid_qty_1") or 0) + (q.get("ask_qty_1") or 0), 1)
                ),
                "imb_l5": (bid_t - ask_t) / max(bid_t + ask_t, 1),
                "pressure": (tb - ts) / max(tb + ts, 1),
            }
        return None


def _bar(value: float, width: int = 21) -> str:
    """Render a [-1, 1] value as a centred text bar."""
    value = max(-1.0, min(1.0, value))
    mid = width // 2
    pos = mid + int(round(value * mid))
    cells = ["-"] * width
    cells[mid] = "|"
    cells[max(0, min(width - 1, pos))] = "#"
    return "".join(cells)


def render(state: FlowState) -> str:
    now = datetime.now(IST)
    L = [
        "=" * 74,
        f"  INSTITUTIONAL FLOW MONITOR          {now:%H:%M:%S} IST   (read only)",
        "=" * 74,
    ]

    spot = state.spot or state.future
    if spot:
        basis = (state.future - state.spot) if (state.future and state.spot) else None
        line = f"  spot {state.spot:,.2f}" if state.spot else "  spot      --"
        if state.future:
            line += f"   future {state.future:,.2f}"
        if basis is not None:
            line += f"   basis {basis:+.2f}"
        L.append(line)

    b = state.book()
    if b:
        L += [
            "",
            "  ORDER BOOK (future)          -- sellers        buyers +",
            f"    L1 imbalance  {b['imb_l1']:+.3f}   {_bar(b['imb_l1'])}",
            f"    L5 imbalance  {b['imb_l5']:+.3f}   {_bar(b['imb_l5'])}",
            f"    book pressure {b['pressure']:+.3f}   {_bar(b['pressure'])}",
            f"    spread        {b['spread']:.2f} pts",
        ]

    w5, w15 = state.writing_bias(5), state.writing_bias(15)
    L += [
        "",
        "  OPTION WRITING            -- call writing      put writing +",
        f"    5-min  bias {w5['bias']:+.3f}   {_bar(w5['bias'])}",
        f"           dCE {w5['d_ce']:>+12,.0f}   dPE {w5['d_pe']:>+12,.0f}",
        f"    15-min bias {w15['bias']:+.3f}   {_bar(w15['bias'])}",
    ]

    ce, pe = state.oi_walls()
    if ce or pe:
        L += ["", "  OI WALLS  (largest open interest = institutional defence)"]
        if ce:
            L.append("    calls (resistance / cap):")
            for strike, oi, d in ce:
                L.append(f"      {strike:>9,.0f}   OI {oi:>12,.0f}   5m {d:>+11,.0f}")
        if pe:
            L.append("    puts (support / floor):")
            for strike, oi, d in pe:
                L.append(f"      {strike:>9,.0f}   OI {oi:>12,.0f}   5m {d:>+11,.0f}")

    # --- interpretation ----------------------------------------------------
    bias = w5["bias"]
    if abs(bias) < 0.15:
        read = "BALANCED -- no clear institutional lean"
    elif bias > 0.35:
        read = "PUT WRITING heavy -> institutions defending downside (bullish lean)"
    elif bias > 0:
        read = "mild put writing (slight bullish lean)"
    elif bias < -0.35:
        read = "CALL WRITING heavy -> institutions capping upside (bearish lean)"
    else:
        read = "mild call writing (slight bearish lean)"

    L += [
        "",
        "-" * 74,
        f"  READ: {read}",
    ]
    if b:
        agree = (bias > 0) == (b["imb_l5"] > 0)
        L.append(
            f"  BOOK: {'agrees with' if agree else 'DISAGREES with'} option flow"
            f"{'' if agree else '  (conflicting signals -- no conviction)'}"
        )
    L += [
        "-" * 74,
        "  NOT a trade signal. Unvalidated hypothesis under observation.",
        "  Measured edge on history is +9pp -- real, but short of what costs need.",
        "",
    ]
    return "\n".join(L)


async def main_async(args) -> None:
    if os.environ.get("DHAN_TOTP_SECRET"):
        tm = TokenManager.from_env()
        provider = lambda force=False: tm.force_refresh() if force else tm.token()
    else:
        tok = os.environ.get("DHAN_ACCESS_TOKEN")
        if not tok:
            raise SystemExit("Set DHAN_ACCESS_TOKEN or DHAN_TOTP_SECRET in .env")
        provider = lambda force=False: tok

    client_id = os.environ["DHAN_CLIENT_ID"]
    master = load_scrip_master(cache_path="data/cache/scrip_master.csv")

    client = DhanClient()
    spots = {}
    for sym in args.symbols:
        sid = {"NIFTY": "13", "BANKNIFTY": "25"}.get(sym)
        now = datetime.now(IST)
        bars = await asyncio.to_thread(
            client.intraday, security_id=sid, exchange_segment="IDX_I",
            instrument="INDEX", interval=1,
            from_date=(now - pd.Timedelta(days=7)).strftime("%Y-%m-%d 09:15:00"),
            to_date=now.strftime("%Y-%m-%d 15:30:00"), oi=False,
        )
        if not bars.empty:
            spots[sym] = float(bars["close"].iloc[-1])

    universe = resolve_universe(
        master, symbols=args.symbols, spots=spots, n_strikes=args.strikes
    )
    state = FlowState(universe)

    feed = LiveFeed(
        instruments=universe, token_provider=provider, client_id=client_id,
        on_ticks=lambda rows: [state.update(r) for r in rows],
    )

    async def draw():
        while True:
            await asyncio.sleep(args.refresh)
            print("\033[2J\033[H" + render(state), flush=True)

    await asyncio.gather(feed.run(), draw())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Live institutional flow monitor (read only)")
    ap.add_argument("--symbols", nargs="+", default=["NIFTY"])
    ap.add_argument("--strikes", type=int, default=10)
    ap.add_argument("--refresh", type=float, default=5.0, help="seconds between redraws")
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
