"""Walk-forward backtest with full India cost modelling.

Design rules, each guarding a specific way backtests lie:

  1. WALK-FORWARD ONLY. Train on the past, trade the future, roll. Never
     evaluate on K-fold — that is for hyperparameter choice, not for P&L.
  2. COSTS PER TRADE, inside the loop. Charged on entry and exit, never netted
     off an equity curve at the end.
  3. SIGNAL AT BAR CLOSE, FILL AT NEXT BAR OPEN. Filling at the signal bar's
     own close is lookahead and is the most common silent backtest bug.
  4. ONE POSITION AT A TIME, with a daily trade cap. Keeps the simulation
     honest about capital and about overtrading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mcube.costs.india_costs import futures_roundtrip, options_roundtrip


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: int                 # +1 long, -1 short
    entry_px: float
    exit_px: float
    gross_pts: float
    cost_rs: float
    net_rs: float
    exit_reason: str          # 'target' | 'stop' | 'time' | 'eod'
    bars_held: int


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series | None = None

    def stats(self) -> dict:
        if not self.trades:
            return {"n_trades": 0}

        net = np.array([t.net_rs for t in self.trades])
        gross = np.array([t.gross_pts for t in self.trades])
        costs = np.array([t.cost_rs for t in self.trades])
        wins = net > 0

        gross_profit = net[wins].sum()
        gross_loss = -net[~wins].sum()

        # Per-trade Sharpe, annualised by trade frequency rather than calendar.
        sharpe = (net.mean() / net.std() * np.sqrt(len(net))) if net.std() > 0 else 0.0

        eq = np.cumsum(net)
        drawdown = eq - np.maximum.accumulate(eq)

        return {
            "n_trades": len(self.trades),
            "hit_rate": round(100 * wins.mean(), 2),
            "net_pnl": round(net.sum(), 0),
            "total_costs": round(costs.sum(), 0),
            # If this ratio is near or above 1, costs are eating the entire edge
            # and the target is too tight — widen it before touching the model.
            "cost_drag_pct": round(100 * costs.sum() / max(abs(net.sum()) + costs.sum(), 1), 1),
            "avg_net_per_trade": round(net.mean(), 0),
            "avg_gross_pts": round(gross.mean(), 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else np.inf,
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(drawdown.min(), 0),
            "avg_bars_held": round(np.mean([t.bars_held for t in self.trades]), 1),
            "exit_breakdown": pd.Series([t.exit_reason for t in self.trades])
            .value_counts()
            .to_dict(),
        }


def _trade_cost(symbol: str, px: float, cfg: dict) -> float:
    inst = cfg["instruments"][symbol]
    lots = cfg["trade"]["lots"]

    if cfg["trade"]["instrument"] == "futures":
        return futures_roundtrip(
            price=px,
            lot_size=inst["lot_size"],
            lots=lots,
            cfg=cfg,
            tick_size=inst["tick_size"],
        ).total

    premium = px * 0.006  # ATM weekly approximation; use live chain when available
    return options_roundtrip(
        premium=premium, lot_size=inst["lot_size"], lots=lots, cfg=cfg
    ).total


def run_backtest(
    prices: pd.Series,
    signals: pd.Series,
    *,
    cfg: dict,
    symbol: str,
) -> BacktestResult:
    """Simulate the strategy bar by bar.

    Args:
        prices:  1-min close of the UNDERLYING, tz-aware DatetimeIndex.
        signals: +1 / -1 / 0 per bar. Must be computable from information
                 available AT that bar's close — the engine handles the
                 execution lag, so do not shift signals yourself.
    """
    trade_cfg = cfg["trade"]
    inst = cfg["instruments"][symbol]
    lot_size, lots = inst["lot_size"], trade_cfg["lots"]

    if trade_cfg["target_mode"] == "points":
        pts = trade_cfg["points"][symbol]
        target_pts, stop_pts = float(pts["target"]), float(pts["stop"])
    else:
        vol = prices.pct_change().ewm(span=trade_cfg["vol"]["vol_lookback"]).std()
        target_pts = stop_pts = None  # resolved per-bar below

    horizon = int(trade_cfg["horizon_bars"])
    max_daily = cfg["backtest"]["max_trades_per_day"]
    delta = cfg["backtest"]["atm_delta"] if trade_cfg["instrument"] == "options" else 1.0

    px = prices.to_numpy(dtype=float)
    idx = prices.index
    sig = signals.reindex(idx).fillna(0).to_numpy()

    cooldown = pd.Timestamp(cfg["session"]["close"]).time()
    trades: list[Trade] = []

    i = 0
    n = len(px)
    trades_today: dict[pd.Timestamp, int] = {}

    while i < n - 1:
        s = sig[i]
        if s == 0:
            i += 1
            continue

        day = idx[i].normalize()
        if trades_today.get(day, 0) >= max_daily:
            i += 1
            continue

        # No new entries in the closing window.
        minutes_left = (
            pd.Timestamp.combine(idx[i].date(), cooldown) - idx[i].tz_localize(None)
        ).total_seconds() / 60
        if minutes_left < cfg["session"]["cooldown_minutes"]:
            i += 1
            continue

        # RULE 3: signal at close of bar i, fill at open of bar i+1.
        entry_i = i + 1
        entry_px = px[entry_i]
        side = int(np.sign(s))

        if trade_cfg["target_mode"] != "points":
            sigma_pts = vol.iloc[i] * entry_px
            if not np.isfinite(sigma_pts):
                i += 1
                continue
            target_pts = sigma_pts * trade_cfg["vol"]["target_mult"]
            stop_pts = sigma_pts * trade_cfg["vol"]["stop_mult"]

        upper = entry_px + target_pts * side
        lower = entry_px - stop_pts * side

        exit_i, exit_px, reason = None, None, "time"
        end = min(entry_i + horizon, n - 1)

        for j in range(entry_i + 1, end + 1):
            # Day boundary — never hold overnight on a 15-minute strategy.
            if idx[j].normalize() != day:
                exit_i, exit_px, reason = j - 1, px[j - 1], "eod"
                break
            hit_target = px[j] >= upper if side > 0 else px[j] <= upper
            hit_stop = px[j] <= lower if side > 0 else px[j] >= lower
            if hit_target:
                exit_i, exit_px, reason = j, upper, "target"
                break
            if hit_stop:
                exit_i, exit_px, reason = j, lower, "stop"
                break

        if exit_i is None:
            exit_i, exit_px, reason = end, px[end], "time"

        gross_pts = (exit_px - entry_px) * side
        # Options convert index points to premium points via delta.
        gross_rs = gross_pts * delta * lot_size * lots
        cost_rs = _trade_cost(symbol, entry_px, cfg)

        trades.append(
            Trade(
                entry_time=idx[entry_i],
                exit_time=idx[exit_i],
                side=side,
                entry_px=entry_px,
                exit_px=exit_px,
                gross_pts=gross_pts,
                cost_rs=cost_rs,
                net_rs=gross_rs - cost_rs,
                exit_reason=reason,
                bars_held=exit_i - entry_i,
            )
        )
        trades_today[day] = trades_today.get(day, 0) + 1

        # RULE 4: one position at a time — resume after the exit.
        i = exit_i + 1

    result = BacktestResult(trades=trades)
    if trades:
        result.equity = pd.Series(
            np.cumsum([t.net_rs for t in trades]) + cfg["backtest"]["initial_capital"],
            index=[t.exit_time for t in trades],
        )
    return result
