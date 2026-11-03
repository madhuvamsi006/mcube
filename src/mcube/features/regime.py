"""Regime and context features.

In Indian index derivatives, time-of-day and days-to-expiry are not minor
controls — they are among the strongest conditioning variables available. The
same OFI reading means different things at 09:20, at 12:30, and at 15:10 on
expiry day. A model without these will blend incompatible regimes and look
mediocre everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Session structure of NSE/BSE equity derivatives. Each block has a genuinely
# different microstructure signature.
SESSION_BLOCKS = {
    "open_auction_spill": ("09:15", "09:30"),  # overnight gap digestion, wide spreads
    "morning_trend": ("09:30", "11:00"),       # highest institutional participation
    "midday_lull": ("11:00", "13:30"),         # thin, mean-reverting, most false signals
    "afternoon_trend": ("13:30", "15:00"),     # European open adds flow
    "closing_drive": ("15:00", "15:30"),       # expiry pin, MTM squaring
}


def time_of_day_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical time encodings plus session-block dummies."""
    out = pd.DataFrame(index=index)

    minutes = index.hour * 60 + index.minute
    session_start, session_end = 9 * 60 + 15, 15 * 60 + 30
    progress = (minutes - session_start) / (session_end - session_start)

    out["session_progress"] = progress
    # Cyclical encoding: 09:16 and 09:15 should be near each other in feature
    # space, which a raw minute counter does not give you.
    out["tod_sin"] = np.sin(2 * np.pi * progress)
    out["tod_cos"] = np.cos(2 * np.pi * progress)

    times = pd.Series(index.strftime("%H:%M"), index=index)
    for name, (start, end) in SESSION_BLOCKS.items():
        out[f"block_{name}"] = ((times >= start) & (times < end)).astype(int)

    return out


def expiry_features(
    index: pd.DatetimeIndex,
    expiry_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Days to expiry and expiry-day flags.

    Expiry day is a different market: gamma rises sharply, option writers hedge
    aggressively, and the index often pins toward max pain. Validation keeps
    expiry days as a separate slice (config.validation.separate_expiry_eval) —
    pooling them flatters metrics and hides that the model works in only one
    regime.

    NOTE: SEBI has repeatedly revised weekly expiry structure (BANKNIFTY weeklies
    discontinued; NIFTY/SENSEX expiry weekdays moved more than once). Pass the
    actual exchange calendar here — do not infer it from a weekday rule.
    """
    out = pd.DataFrame(index=index)
    dates = index.normalize()
    expiries = pd.DatetimeIndex(sorted(pd.DatetimeIndex(expiry_dates).normalize().unique()))

    pos = expiries.searchsorted(dates, side="left").clip(0, len(expiries) - 1)
    next_expiry = expiries[pos]

    dte = (next_expiry - dates).days
    out["days_to_expiry"] = dte
    out["is_expiry_day"] = (dte == 0).astype(int)
    out["is_expiry_week"] = (dte <= 4).astype(int)

    # Gamma grows roughly as 1/sqrt(T). This is the mechanism behind most
    # expiry-day behaviour, so give the model the shape directly.
    out["gamma_proxy"] = 1.0 / np.sqrt(np.maximum(dte, 0.25))

    # Expiry-day afternoon is where pinning actually happens.
    minutes = index.hour * 60 + index.minute
    out["expiry_afternoon"] = ((dte == 0) & (minutes >= 13 * 60)).astype(int)

    return out


def volatility_regime(spot: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    """Realised and implied volatility regime.

    A 20-point NIFTY target is a routine move at 20 VIX and an unusual one at
    10 VIX. Without this the model cannot know which world it is in.
    """
    out = pd.DataFrame(index=spot.index)

    returns = spot.pct_change()
    for w in (15, 60, 240):
        out[f"rv_{w}"] = returns.rolling(w).std() * np.sqrt(375)  # 375 min/session

    out["rv_ratio"] = out["rv_15"] / out["rv_240"].replace(0, np.nan)  # short vs long vol

    if vix is not None:
        v = vix.reindex(spot.index).ffill()
        out["vix"] = v
        out["vix_chg"] = v.diff(15)
        out["vix_z"] = (v - v.rolling(2000).mean()) / v.rolling(2000).std()
        # IV richer than RV = options expensive; buying premium is penalised.
        out["iv_rv_spread"] = v / 100 - out["rv_60"]

    return out


def cross_index_features(
    nifty: pd.Series,
    banknifty: pd.Series,
    sensex: pd.Series | None = None,
) -> pd.DataFrame:
    """Lead-lag and dislocation between the three indices.

    BANKNIFTY is higher-beta and often leads NIFTY at turns. NIFTY and SENSEX
    track nearly the same economy on different exchanges, so a divergence
    between them is usually a short-lived dislocation rather than information.
    """
    out = pd.DataFrame(index=nifty.index)

    bn = banknifty.reindex(nifty.index).ffill()
    out["bn_nifty_ratio"] = bn / nifty
    out["bn_nifty_ratio_chg"] = out["bn_nifty_ratio"].pct_change(15)

    n_ret = nifty.pct_change(5)
    b_ret = bn.pct_change(5)
    out["bn_lead"] = b_ret - n_ret  # BANKNIFTY outrunning NIFTY
    out["corr_60"] = n_ret.rolling(60).corr(b_ret)

    if sensex is not None:
        sx = sensex.reindex(nifty.index).ffill()
        s_ret = sx.pct_change(5)
        out["sensex_nifty_div"] = s_ret - n_ret
        out["sensex_nifty_div_z"] = (
            (out["sensex_nifty_div"] - out["sensex_nifty_div"].rolling(120).mean())
            / out["sensex_nifty_div"].rolling(120).std()
        )

    return out
