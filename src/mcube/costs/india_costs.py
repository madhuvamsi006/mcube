"""India transaction-cost model for NSE/BSE index derivatives.

Costs are charged per round trip and are large relative to a 10-30 point capture,
so they are modelled per-trade inside the backtest loop rather than subtracted
from a final equity curve.

Two asymmetries drive most of the economics and are easy to miss:

  1. Futures STT is charged on NOTIONAL (0.02% of ~Rs19.5L/lot).
     Options STT is charged on PREMIUM (0.1% of ~Rs11k/lot).
     That is a ~35x difference on the dominant line item, which is why small
     point captures are cheaper to express in options than in futures.

  2. Costs shrink wins AND enlarge losses. A nominal 1.5:1 trade nets far worse
     than 1.5:1 after charges. `breakeven_hit_rate()` reports the real number.

All rates come from config.charges and reflect the post-Oct-2024 STT revision.
Verify against a live broker charge sheet before trusting any P&L.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class CostBreakdown:
    """Round-trip charges in rupees, itemised so you can see what dominates."""

    brokerage: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp_duty: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange_txn
            + self.sebi
            + self.stamp_duty
            + self.gst
            + self.slippage
        )

    def in_points(self, lot_size: int, lots: int) -> float:
        """Total cost expressed in underlying index points."""
        return self.total / (lot_size * lots)

    def as_dict(self) -> dict:
        return {**asdict(self), "total": self.total}


def futures_roundtrip(
    *,
    price: float,
    lot_size: int,
    lots: int,
    cfg: dict,
    tick_size: float = 0.05,
) -> CostBreakdown:
    """Round-trip (buy + sell) charges for index futures.

    `price` is the futures price; notional = price * lot_size * lots.
    """
    c = cfg["charges"]["futures"]
    notional = price * lot_size * lots
    turnover = notional * 2  # both legs

    # Flat Rs20 or pct, whichever is lower — per executed order, both legs.
    brokerage = 2 * min(c["brokerage_per_order"], notional * c["brokerage_pct_cap"])

    stt = notional * c["stt_sell_pct"]  # sell leg only
    exchange_txn = turnover * c["exchange_txn_pct"]
    sebi = turnover * c["sebi_pct"]
    stamp_duty = notional * c["stamp_duty_buy_pct"]  # buy leg only
    gst = (brokerage + exchange_txn + sebi) * c["gst_pct"]

    # Crossing the spread on entry and exit.
    slippage = c["slippage_ticks"] * tick_size * lot_size * lots * 2

    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange_txn=exchange_txn,
        sebi=sebi,
        stamp_duty=stamp_duty,
        gst=gst,
        slippage=slippage,
    )


def options_roundtrip(
    *,
    premium: float,
    lot_size: int,
    lots: int,
    cfg: dict,
) -> CostBreakdown:
    """Round-trip charges for index options (long premium, bought then sold).

    Every percentage charge applies to PREMIUM turnover, not notional — this is
    the whole reason small captures are viable in options and not in futures.
    """
    c = cfg["charges"]["options"]
    premium_value = premium * lot_size * lots
    turnover = premium_value * 2

    brokerage = 2 * c["brokerage_per_order"]
    stt = premium_value * c["stt_sell_pct"]
    exchange_txn = turnover * c["exchange_txn_pct"]
    sebi = turnover * c["sebi_pct"]
    stamp_duty = premium_value * c["stamp_duty_buy_pct"]
    gst = (brokerage + exchange_txn + sebi) * c["gst_pct"]

    # Options slippage dwarfs the statutory charges. ATM is ~Rs1 wide on a Rs150
    # premium; OTM strikes and fast moves are materially worse.
    slippage = premium_value * c["slippage_pct"] * 2

    return CostBreakdown(
        brokerage=brokerage,
        stt=stt,
        exchange_txn=exchange_txn,
        sebi=sebi,
        stamp_duty=stamp_duty,
        gst=gst,
        slippage=slippage,
    )


def cost_in_index_points(
    *,
    symbol: str,
    spot: float,
    cfg: dict,
    instrument: str | None = None,
    lots: int | None = None,
    premium: float | None = None,
) -> float:
    """Round-trip cost expressed in UNDERLYING INDEX POINTS.

    This is the number to compare a point target against. For options the raw
    premium cost is divided by delta, because a 1-point premium move needs
    1/delta index points to produce it.
    """
    inst_cfg = cfg["instruments"][symbol]
    lot_size = inst_cfg["lot_size"]
    instrument = instrument or cfg["trade"]["instrument"]
    lots = lots or cfg["trade"]["lots"]

    if instrument == "futures":
        cb = futures_roundtrip(
            price=spot,
            lot_size=lot_size,
            lots=lots,
            cfg=cfg,
            tick_size=inst_cfg["tick_size"],
        )
        return cb.in_points(lot_size, lots)

    # Options: estimate ATM premium if not supplied. Rough rule for weekly ATM
    # index options; replace with live chain data once the feed is wired.
    delta = cfg["backtest"]["atm_delta"]
    if premium is None:
        premium = spot * 0.006

    cb = options_roundtrip(premium=premium, lot_size=lot_size, lots=lots, cfg=cfg)
    premium_points = cb.in_points(lot_size, lots)
    return premium_points / delta


def breakeven_hit_rate(*, target_pts: float, stop_pts: float, cost_pts: float) -> float:
    """Hit rate needed just to break even, after costs.

    Costs hit both legs: a win nets (target - cost), a loss costs (stop + cost).
    That is what turns a nominal 1.5:1 into something much worse.

        p * (target - cost) = (1 - p) * (stop + cost)

    Returns >1.0 when the target cannot clear costs at any hit rate.
    """
    net_win = target_pts - cost_pts
    net_loss = stop_pts + cost_pts
    if net_win <= 0:
        return float("inf")
    return net_loss / (net_win + net_loss)


def expectancy_points(
    *, hit_rate: float, target_pts: float, stop_pts: float, cost_pts: float
) -> float:
    """Expected index points per trade, net of costs. Negative = losing system."""
    return hit_rate * (target_pts - cost_pts) - (1 - hit_rate) * (stop_pts + cost_pts)


def expectancy_rupees(
    *,
    hit_rate: float,
    target_pts: float,
    stop_pts: float,
    cost_pts: float,
    lot_size: int,
    lots: int,
) -> float:
    """Expected rupees per trade — the number to judge the strategy on."""
    pts = expectancy_points(
        hit_rate=hit_rate,
        target_pts=target_pts,
        stop_pts=stop_pts,
        cost_pts=cost_pts,
    )
    return pts * lot_size * lots


def random_walk_base_rate(*, target_pts: float, stop_pts: float) -> float:
    """P(hit target before stop) for a driftless random walk.

    Standard gambler's-ruin result: stop / (target + stop). A closer stop is
    hit more often, so an asymmetric 1.5:1 trade resolves as a loss ~60% of the
    time BEFORE any skill is involved.

    This is the number a model must beat. Comparing accuracy to 50% is wrong
    whenever the barriers are asymmetric.
    """
    return stop_pts / (target_pts + stop_pts)


def required_edge(*, target_pts: float, stop_pts: float, cost_pts: float) -> float:
    """Percentage points of skill needed over a coin flip, after costs.

    required_edge = breakeven_hit_rate - random_walk_base_rate

    This is the honest difficulty measure. Breakeven alone understates the task
    because the natural base rate sits near 40%, not 50% — so a "55% breakeven"
    is really asking for +15pp of genuine predictive skill.

    Widening the target lowers this, because cost shrinks as a share of the move
    while the base rate stays fixed at 1/(1+R) for a given reward:risk ratio.
    """
    be = breakeven_hit_rate(target_pts=target_pts, stop_pts=stop_pts, cost_pts=cost_pts)
    if be == float("inf"):
        return float("inf")
    return be - random_walk_base_rate(target_pts=target_pts, stop_pts=stop_pts)
