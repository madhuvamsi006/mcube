"""Cost model tests.

These numbers drive every go/no-go decision in the project, so they are pinned.
If a rate in config.yaml changes, these tests should fail loudly rather than
letting a stale assumption quietly inflate backtest P&L.
"""

import yaml
import pytest

from mcube.costs.india_costs import (
    breakeven_hit_rate,
    cost_in_index_points,
    expectancy_points,
    futures_roundtrip,
    options_roundtrip,
)


@pytest.fixture
def cfg():
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_futures_stt_dominates(cfg):
    """STT on notional should be the largest single charge on futures."""
    cb = futures_roundtrip(price=26000, lot_size=75, lots=1, cfg=cfg)
    others = [cb.brokerage, cb.exchange_txn, cb.sebi, cb.stamp_duty, cb.gst]
    assert cb.stt > max(others), "STT should dominate futures charges"


def test_options_cheaper_than_futures_in_index_points(cfg):
    """The core economic claim: premium-based STT makes options cheaper.

    If this ever inverts, the instrument recommendation must be revisited.
    """
    fut = cost_in_index_points(symbol="NIFTY", spot=26000, cfg=cfg, instrument="futures", lots=5)
    opt = cost_in_index_points(symbol="NIFTY", spot=26000, cfg=cfg, instrument="options", lots=5)
    assert opt < fut


def test_lots_do_not_change_cost_per_point(cfg):
    """Lots scale P&L and cost together — viability must not depend on size.

    Flat per-order brokerage amortises slightly, so allow a small improvement
    but never a material one.
    """
    one = cost_in_index_points(symbol="NIFTY", spot=26000, cfg=cfg, instrument="futures", lots=1)
    five = cost_in_index_points(symbol="NIFTY", spot=26000, cfg=cfg, instrument="futures", lots=5)
    assert five < one                      # brokerage amortises
    assert (one - five) / one < 0.15       # but only marginally


def test_breakeven_rises_as_target_tightens(cfg):
    """Tighter targets must require higher hit rates. Monotonic by construction."""
    cost = cost_in_index_points(symbol="NIFTY", spot=26000, cfg=cfg, instrument="options", lots=5)
    rates = [
        breakeven_hit_rate(target_pts=t, stop_pts=round(t / 1.5), cost_pts=cost)
        for t in (10, 15, 20, 30, 50)
    ]
    assert rates == sorted(rates, reverse=True)


def test_target_below_cost_is_impossible(cfg):
    """A target that cannot clear costs must report infinity, not a plausible number."""
    assert breakeven_hit_rate(target_pts=3, stop_pts=2, cost_pts=7.0) == float("inf")


def test_costs_hurt_both_legs(cfg):
    """Costs must shrink wins AND enlarge losses.

    This asymmetry is the reason a nominal 1.5:1 trade pays far worse than 1.5:1,
    and the reason small point targets fail.
    """
    no_cost = expectancy_points(hit_rate=0.55, target_pts=20, stop_pts=13, cost_pts=0)
    with_cost = expectancy_points(hit_rate=0.55, target_pts=20, stop_pts=13, cost_pts=5)
    # A 5-point cost must cost a full 5 points of expectancy, not 5 * hit_rate.
    assert no_cost - with_cost == pytest.approx(5.0)


def test_options_stt_charged_on_premium_not_notional(cfg):
    """Guards the single most important modelling detail in the repo."""
    cb = options_roundtrip(premium=150, lot_size=75, lots=1, cfg=cfg)
    premium_value = 150 * 75
    assert cb.stt == pytest.approx(premium_value * 0.001)
    assert cb.stt < 20  # nothing like the ~Rs390 a notional base would give
