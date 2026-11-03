"""Order-flow microstructure features from 5-level depth snapshots.

Scope note: broker WebSockets on NSE deliver DEPTH SNAPSHOTS at roughly 1 Hz,
not an order-by-order stream. Classic OFI (Cont, Kukanov & Stoikov 2014) is
defined on order events, so the snapshot formulation below approximates it from
consecutive book states. It is noisier than the event-based original and should
be treated as an approximation, not the published estimator.

That quantisation is also why latency optimisation is pointless here: the feed
is ~1s granular before any of this code runs.

Expected snapshot schema (one row per instrument per snapshot):
    ts, symbol, ltp, ltq, volume, oi,
    bid_px_1..5, bid_qty_1..5, ask_px_1..5, ask_qty_1..5
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lvl(df: pd.DataFrame, side: str, field: str, level: int) -> pd.Series:
    return df[f"{side}_{field}_{level}"]


def order_flow_imbalance(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """Snapshot-based OFI at a given book level.

    Per Cont-Kukanov-Stoikov, with consecutive snapshots standing in for events:

        e = 1{Pb_t >= Pb_t-1} * Qb_t  -  1{Pb_t <= Pb_t-1} * Qb_t-1
          - 1{Pa_t <= Pa_t-1} * Qa_t  +  1{Pa_t >= Pa_t-1} * Qa_t-1

    Intuition: a bid that ticks up or thickens is buying pressure; an ask that
    ticks down or thickens is selling pressure. Positive OFI = net buy pressure.
    """
    bid_px, bid_qty = _lvl(df, "bid", "px", level), _lvl(df, "bid", "qty", level)
    ask_px, ask_qty = _lvl(df, "ask", "px", level), _lvl(df, "ask", "qty", level)

    bid_px_prev, bid_qty_prev = bid_px.shift(1), bid_qty.shift(1)
    ask_px_prev, ask_qty_prev = ask_px.shift(1), ask_qty.shift(1)

    bid_term = np.where(bid_px >= bid_px_prev, bid_qty, 0) - np.where(
        bid_px <= bid_px_prev, bid_qty_prev, 0
    )
    ask_term = np.where(ask_px <= ask_px_prev, ask_qty, 0) - np.where(
        ask_px >= ask_px_prev, ask_qty_prev, 0
    )

    return pd.Series(bid_term - ask_term, index=df.index, name=f"ofi_l{level}")


def multi_level_ofi(df: pd.DataFrame, levels: int = 5, decay: float = 0.5) -> pd.Series:
    """Depth-weighted OFI across levels.

    Deeper levels carry information but are easier to spoof and are refreshed
    less reliably in snapshot feeds, so their weight decays geometrically.
    """
    total = pd.Series(0.0, index=df.index)
    for lv in range(1, levels + 1):
        total += (decay ** (lv - 1)) * order_flow_imbalance(df, lv)
    return total.rename("ofi_multi")


def depth_imbalance(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """Normalised book imbalance, in [-1, 1]. Positive = bid-heavy."""
    out = pd.DataFrame(index=df.index)

    for lv in range(1, levels + 1):
        b, a = _lvl(df, "bid", "qty", lv), _lvl(df, "ask", "qty", lv)
        out[f"depth_imb_l{lv}"] = (b - a) / (b + a).replace(0, np.nan)

    bid_total = sum(_lvl(df, "bid", "qty", lv) for lv in range(1, levels + 1))
    ask_total = sum(_lvl(df, "ask", "qty", lv) for lv in range(1, levels + 1))
    out["depth_imb_total"] = (bid_total - ask_total) / (bid_total + ask_total).replace(0, np.nan)

    return out


def microprice(df: pd.DataFrame) -> pd.Series:
    """Size-weighted mid — a better short-horizon fair value than the raw mid.

        microprice = (Pb * Qa + Pa * Qb) / (Qa + Qb)

    Weighted toward the thin side, because that is the side that gets taken
    first. `microprice - mid` is a direct, well-documented drift predictor.
    """
    bid_px, bid_qty = df["bid_px_1"], df["bid_qty_1"]
    ask_px, ask_qty = df["ask_px_1"], df["ask_qty_1"]
    denom = (bid_qty + ask_qty).replace(0, np.nan)
    return ((bid_px * ask_qty + ask_px * bid_qty) / denom).rename("microprice")


def infer_trade_sign(df: pd.DataFrame) -> pd.Series:
    """Aggressor side via the quote rule, falling back to the tick rule.

    Snapshot feeds do not flag the aggressor, so it must be inferred:
      trade above mid -> buyer-initiated
      trade below mid -> seller-initiated
      trade at mid    -> tick rule (direction of the last price change)
    """
    mid = (df["bid_px_1"] + df["ask_px_1"]) / 2
    ltp = df["ltp"]

    sign = pd.Series(0.0, index=df.index)
    sign[ltp > mid] = 1.0
    sign[ltp < mid] = -1.0

    tick = np.sign(ltp.diff()).replace(0, np.nan).ffill().fillna(0)
    return sign.where(sign != 0, tick).rename("trade_sign")


def microstructure_features(df: pd.DataFrame, *, cfg: dict) -> pd.DataFrame:
    """Full microstructure feature matrix from a depth-snapshot frame.

    Raw OFI at 1 Hz is far too noisy to use directly; everything meaningful is
    in the rolling sums and their z-scores.
    """
    levels = cfg["data"]["depth_levels"]
    windows = cfg["features"]["ofi_windows"]

    out = pd.DataFrame(index=df.index)

    mid = (df["bid_px_1"] + df["ask_px_1"]) / 2
    out["mid"] = mid
    out["spread"] = df["ask_px_1"] - df["bid_px_1"]
    out["spread_bps"] = 1e4 * out["spread"] / mid

    mp = microprice(df)
    out["microprice"] = mp
    # The headline drift signal: fair value pulling away from the mid.
    out["microprice_dev"] = (mp - mid) / out["spread"].replace(0, np.nan)

    out = out.join(depth_imbalance(df, levels))

    ofi = multi_level_ofi(df, levels)
    out["ofi"] = ofi

    sign = infer_trade_sign(df)
    signed_vol = sign * df["volume"].diff().clip(lower=0)

    for w in windows:
        bars = w  # 1-min bars
        out[f"ofi_sum_{w}"] = ofi.rolling(bars).sum()
        out[f"ofi_z_{w}"] = (
            (out[f"ofi_sum_{w}"] - out[f"ofi_sum_{w}"].rolling(120).mean())
            / out[f"ofi_sum_{w}"].rolling(120).std()
        )
        out[f"signed_vol_{w}"] = signed_vol.rolling(bars).sum()

        vol_sum = df["volume"].diff().clip(lower=0).rolling(bars).sum()
        out[f"trade_imb_{w}"] = out[f"signed_vol_{w}"] / vol_sum.replace(0, np.nan)

        out[f"ret_{w}"] = mid.pct_change(bars)
        out[f"realized_vol_{w}"] = mid.pct_change().rolling(bars).std()

    # Depth withdrawal: book thinning without trades going through often precedes
    # a fast move — liquidity stepping away ahead of flow.
    total_depth = sum(
        _lvl(df, "bid", "qty", lv) + _lvl(df, "ask", "qty", lv) for lv in range(1, levels + 1)
    )
    out["depth_total"] = total_depth
    out["depth_chg_5"] = total_depth.pct_change(5)
    out["depth_z"] = (
        (total_depth - total_depth.rolling(120).mean()) / total_depth.rolling(120).std()
    )

    return out
