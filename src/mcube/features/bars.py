"""Features computable from 1-minute OHLCV bars alone.

This is the v1 feature set. It exists because Dhan serves years of historical
bars but NOT historical order-book depth or intraday strike-level OI -- those
are live-only and must be recorded going forward.

So the honest framing: this set is weaker than the full design. It has no true
order flow, only proxies inferred from bar shape and volume. If it shows an
edge, the deep-book and OI features should widen it. If it shows nothing, that
is useful too -- it tells us the edge (if any) lives in the microstructure we
are not yet recording, which justifies the collector.

Every feature here is strictly backward-looking. Any centred or forward window
would leak, and the null test would catch it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bar_features(df: pd.DataFrame, *, windows: tuple[int, ...] = (1, 3, 5, 15, 30, 60)) -> pd.DataFrame:
    """Build the bar-based feature matrix.

    Args:
        df: OHLCV indexed by tz-aware timestamp, columns
            [open, high, low, close, volume] and optionally open_interest.
    """
    out = pd.DataFrame(index=df.index)

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # --- bar shape ---------------------------------------------------------
    rng = (h - l).replace(0, np.nan)
    # Where the bar closed inside its own range. Near 1 = buyers held the high
    # into the close; near 0 = sellers. The cheapest available proxy for who
    # was in control during the bar.
    out["close_loc"] = (c - l) / rng
    out["body_ratio"] = (c - o) / rng
    out["upper_wick"] = (h - np.maximum(o, c)) / rng
    out["lower_wick"] = (np.minimum(o, c) - l) / rng
    out["range_bps"] = 1e4 * rng / c

    # --- returns and momentum ---------------------------------------------
    for w in windows:
        out[f"ret_{w}"] = c.pct_change(w)
        out[f"ret_z_{w}"] = (
            out[f"ret_{w}"] - out[f"ret_{w}"].rolling(240).mean()
        ) / out[f"ret_{w}"].rolling(240).std()

    # --- volatility --------------------------------------------------------
    r1 = c.pct_change()
    for w in (15, 60, 240):
        out[f"rv_{w}"] = r1.rolling(w).std()
    out["rv_ratio"] = out["rv_15"] / out["rv_240"].replace(0, np.nan)

    # Parkinson estimator -- uses the high-low range, so it is more efficient
    # than close-to-close vol at short horizons.
    out["parkinson_15"] = np.sqrt(
        (np.log(h / l) ** 2).rolling(15).mean() / (4 * np.log(2))
    )

    # --- volume ------------------------------------------------------------
    out["vol_z_30"] = (v - v.rolling(30).mean()) / v.rolling(30).std()
    out["vol_z_240"] = (v - v.rolling(240).mean()) / v.rolling(240).std()
    out["vol_ratio"] = v.rolling(5).mean() / v.rolling(60).mean().replace(0, np.nan)

    # --- order-flow proxy --------------------------------------------------
    # With no aggressor flag in bar data, bar direction weighted by volume is
    # the standard stand-in for signed flow. Crude next to real OFI, but it
    # carries the same sign information.
    signed_vol = np.sign(c - o) * v
    for w in (5, 15, 30):
        out[f"signed_vol_{w}"] = signed_vol.rolling(w).sum()
        tot = v.rolling(w).sum().replace(0, np.nan)
        out[f"flow_imb_{w}"] = out[f"signed_vol_{w}"] / tot

    # --- VWAP deviation ----------------------------------------------------
    # Session VWAP is the institutional execution benchmark, so distance from
    # it is a genuine reference point rather than an arbitrary moving average.
    tp = (h + l + c) / 3
    day = df.index.normalize()
    cum_pv = (tp * v).groupby(day).cumsum()
    cum_v = v.groupby(day).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v
    out["vwap_dev_bps"] = 1e4 * (c - vwap) / c

    # --- mean reversion vs trend ------------------------------------------
    for w in (15, 60):
        ma = c.rolling(w).mean()
        sd = c.rolling(w).std().replace(0, np.nan)
        out[f"zscore_{w}"] = (c - ma) / sd

    # Consecutive same-direction bars -- short-run streaks mean-revert at this
    # horizon more often than they continue.
    direction = np.sign(c.diff()).fillna(0)
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
    out["streak"] = streak * direction

    # --- open interest, when present (futures only) ------------------------
    if "open_interest" in df.columns and df["open_interest"].notna().any():
        oi = df["open_interest"].astype(float)
        for w in (5, 15, 30):
            out[f"doi_{w}"] = oi.diff(w)
            out[f"doi_pct_{w}"] = oi.pct_change(w)
        # Indian OI buildup taxonomy: price up + OI up = long buildup (fresh
        # conviction); price up + OI down = short covering (exhausting).
        price_chg = c.diff(15)
        oi_chg = oi.diff(15)
        score = pd.Series(0.0, index=df.index)
        score[(price_chg > 0) & (oi_chg > 0)] = 1.0
        score[(price_chg < 0) & (oi_chg > 0)] = -1.0
        score[(price_chg > 0) & (oi_chg < 0)] = 0.5
        score[(price_chg < 0) & (oi_chg < 0)] = -0.5
        out["buildup_score"] = score

    return out


def session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Time-of-day encodings.

    Time of day is one of the strongest conditioning variables in Indian index
    trading: the same reading means different things in the opening spill, the
    midday lull, and the closing drive.
    """
    out = pd.DataFrame(index=index)
    minutes = index.hour * 60 + index.minute
    start, end = 9 * 60 + 15, 15 * 60 + 30
    prog = (minutes - start) / (end - start)

    out["session_progress"] = prog
    out["tod_sin"] = np.sin(2 * np.pi * prog)
    out["tod_cos"] = np.cos(2 * np.pi * prog)
    out["minutes_from_open"] = minutes - start
    out["minutes_to_close"] = end - minutes
    out["dow"] = index.dayofweek

    out["is_opening_30"] = (minutes < start + 30).astype(int)
    out["is_midday"] = ((minutes >= 11 * 60) & (minutes < 13 * 60 + 30)).astype(int)
    out["is_closing_30"] = (minutes >= end - 30).astype(int)

    return out
