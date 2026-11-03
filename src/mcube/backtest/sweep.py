"""Target sweep — what point capture actually clears costs.

Runs with NO market data. Pure cost arithmetic, so you can run it right now,
before any feed is wired, and see which (instrument, target, hit-rate) cells
are viable.

    python -m mcube.backtest.sweep

Read the output as a feasibility map, not a forecast. It assumes you achieve the
stated hit rate; whether the model does is what the rest of the repo is for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from mcube.costs.india_costs import (
    breakeven_hit_rate,
    cost_in_index_points,
    expectancy_points,
    expectancy_rupees,
)

# Illustrative index levels — override with --spot for current values.
DEFAULT_SPOT = {"NIFTY": 26000.0, "BANKNIFTY": 58000.0, "SENSEX": 85000.0}

TARGET_GRIDS = {
    "NIFTY": [10, 15, 20, 25, 30, 40, 50],
    "BANKNIFTY": [30, 45, 60, 75, 100, 125, 150],
    "SENSEX": [35, 50, 70, 90, 120, 150, 180],
}

HIT_RATES = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

RR = 1.5  # target : stop


def load_cfg(path: Path) -> dict:
    # encoding is explicit: Windows defaults to cp1252 and chokes on the
    # non-ASCII characters in config.yaml.
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sweep_symbol(symbol: str, cfg: dict, spot: float, lots: int) -> None:
    lot_size = cfg["instruments"][symbol]["lot_size"]

    print(f"\n{'=' * 78}")
    print(f"  {symbol}   spot={spot:,.0f}   lot={lot_size}   lots={lots}")
    print(f"{'=' * 78}")

    for instrument in ("futures", "options"):
        cost_pts = cost_in_index_points(
            symbol=symbol, spot=spot, cfg=cfg, instrument=instrument, lots=lots
        )
        print(f"\n  {instrument.upper()}  —  round-trip cost = {cost_pts:.1f} index points")
        print(f"  {'-' * 74}")

        header = f"  {'target/stop':>14} {'breakeven':>11} " + "".join(
            f"{round(h * 100):>9}%" for h in HIT_RATES
        )
        print(header)
        print(f"  {'':>14} {'hit rate':>11} " + "".join(f"{'Rs/trade':>10}" for _ in HIT_RATES))
        print(f"  {'-' * 74}")

        for target in TARGET_GRIDS[symbol]:
            stop = round(target / RR)
            be = breakeven_hit_rate(target_pts=target, stop_pts=stop, cost_pts=cost_pts)

            be_str = "impossible" if be == float("inf") else f"{be * 100:.1f}%"
            row = f"  {f'{target}/{stop}':>14} {be_str:>11} "

            for hr in HIT_RATES:
                rs = expectancy_rupees(
                    hit_rate=hr,
                    target_pts=target,
                    stop_pts=stop,
                    cost_pts=cost_pts,
                    lot_size=lot_size,
                    lots=lots,
                )
                mark = "" if rs > 0 else "-"
                row += f"{mark}{abs(rs):>9,.0f}" if rs < 0 else f"{rs:>10,.0f}"
            print(row)


def annual_projection(symbol: str, cfg: dict, spot: float, lots: int) -> None:
    """Translate per-trade expectancy into an annual figure.

    Trade count matters as much as edge: a thin edge taken often can beat a
    fat edge taken rarely, and vice versa.
    """
    lot_size = cfg["instruments"][symbol]["lot_size"]
    trades_per_day = cfg["backtest"]["max_trades_per_day"]
    sessions = 250

    print(f"\n  Annual projection — {trades_per_day} trades/day x {sessions} sessions")
    print(f"  {'-' * 74}")
    print(f"  {'instrument':>12} {'target/stop':>12} {'hit rate':>10} {'Rs/trade':>12} {'Rs/year':>16}")
    print(f"  {'-' * 74}")

    for instrument in ("futures", "options"):
        cost_pts = cost_in_index_points(
            symbol=symbol, spot=spot, cfg=cfg, instrument=instrument, lots=lots
        )
        for target in TARGET_GRIDS[symbol][:4]:
            stop = round(target / RR)
            for hr in (0.54, 0.58):
                rs = expectancy_rupees(
                    hit_rate=hr,
                    target_pts=target,
                    stop_pts=stop,
                    cost_pts=cost_pts,
                    lot_size=lot_size,
                    lots=lots,
                )
                annual = rs * trades_per_day * sessions
                print(
                    f"  {instrument:>12} {f'{target}/{stop}':>12} {hr * 100:>9.0f}% "
                    f"{rs:>12,.0f} {annual:>16,.0f}"
                )


def main() -> None:
    # Windows consoles default to cp1252 and mangle non-ASCII output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description="Sweep point targets against India cost floor")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--symbol", default="NIFTY", choices=["NIFTY", "BANKNIFTY", "SENSEX", "ALL"])
    ap.add_argument("--spot", type=float, default=None, help="current index level")
    ap.add_argument("--lots", type=int, default=None)
    ap.add_argument("--annual", action="store_true", help="include annual projection")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    lots = args.lots or cfg["trade"]["lots"]

    symbols = ["NIFTY", "BANKNIFTY", "SENSEX"] if args.symbol == "ALL" else [args.symbol]

    for sym in symbols:
        spot = args.spot or DEFAULT_SPOT[sym]
        sweep_symbol(sym, cfg, spot, lots)
        if args.annual:
            annual_projection(sym, cfg, spot, lots)

    print(
        "\n  Rs/trade is EXPECTANCY — the average across wins and losses, not the\n"
        "  size of a winning trade. Negative cells lose money at that hit rate\n"
        "  however many lots you trade, because lots scale cost and P&L alike.\n"
    )


if __name__ == "__main__":
    main()
