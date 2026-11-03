"""Institutional flow detection for NSE/BSE index derivatives.

This is the core of the thesis, so it is worth being precise about what is and
is not observable.

NOT OBSERVABLE on NSE/BSE: order-by-order (MBO) data. No retail feed exposes it,
so true iceberg detection and queue-position modelling are impossible. Anything
claiming "institutional order detection" from 5-level depth is inferring, not
observing. We do not pretend otherwise.

OBSERVABLE, and genuinely strong — India exposes positioning data that US equity
desks do not get:

  1. Strike-level OI changes      institutions express directional views by
                                  WRITING options. dOI at a strike is the single
                                  most direct institutional-intent signal here.
  2. Put-call parity dislocation  synthetic futures (K + C - P) vs actual futures.
                                  A persistent gap means someone is lifting one
                                  side of the chain aggressively.
  3. Futures basis vs spot        institutional hedging pressure.
  4. Participant-wise OI          NSE publishes F&O OI split FII/DII/Pro/Client
                                  daily. Literally labelled institutional
                                  positioning. Daily, so it is a regime filter,
                                  not a 15-minute trigger.
  5. Large-trade clustering       trades far above median size, inferred from
                                  volume bursts against depth.

The standard Indian OI buildup taxonomy (price x OI) is the backbone:

    price UP   + OI UP    -> LONG BUILDUP      bullish, new money long
    price DOWN + OI UP    -> SHORT BUILDUP     bearish, new money short
    price UP   + OI DOWN  -> SHORT COVERING    bullish but exhausting
    price DOWN + OI DOWN  -> LONG UNWINDING    bearish but exhausting

The distinction that matters: buildup is new conviction, unwinding is exit.
A rally on short covering is far weaker than a rally on long buildup, and the
model should be able to tell them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# OI buildup classification
# ---------------------------------------------------------------------------

def classify_buildup(price_chg: pd.Series, oi_chg: pd.Series) -> pd.Series:
    """Map (price change, OI change) to the four-state buildup taxonomy."""
    conditions = [
        (price_chg > 0) & (oi_chg > 0),
        (price_chg < 0) & (oi_chg > 0),
        (price_chg > 0) & (oi_chg < 0),
        (price_chg < 0) & (oi_chg < 0),
    ]
    choices = ["long_buildup", "short_buildup", "short_covering", "long_unwinding"]
    return pd.Series(np.select(conditions, choices, default="neutral"), index=price_chg.index)


def buildup_score(price_chg: pd.Series, oi_chg: pd.Series) -> pd.Series:
    """Signed conviction score in [-1, 1] — a model-friendly form of the taxonomy.

    New positions (buildup) are weighted at full strength; exits (covering,
    unwinding) at half, because they reflect position closure rather than fresh
    conviction and tend to be weaker directional signals.
    """
    score = pd.Series(0.0, index=price_chg.index)
    score[(price_chg > 0) & (oi_chg > 0)] = 1.0   # long buildup
    score[(price_chg < 0) & (oi_chg > 0)] = -1.0  # short buildup
    score[(price_chg > 0) & (oi_chg < 0)] = 0.5   # short covering
    score[(price_chg < 0) & (oi_chg < 0)] = -0.5  # long unwinding
    return score


# ---------------------------------------------------------------------------
# Option chain features
# ---------------------------------------------------------------------------

def option_chain_features(
    chain: pd.DataFrame,
    *,
    spot: pd.Series,
    strike_step: int,
    windows: list[int],
    n_strikes: int = 5,
) -> pd.DataFrame:
    """Per-bar institutional-flow features from the intraday option chain.

    Args:
        chain: long-format snapshots with columns
               [ts, strike, opt_type ('CE'|'PE'), ltp, oi, volume, iv]
        spot:  underlying close, indexed by ts
        strike_step: 50 (NIFTY) / 100 (BANKNIFTY, SENSEX)
        windows: lookbacks in bars for dOI aggregation
        n_strikes: how many strikes either side of ATM to include

    Returns a DataFrame indexed by ts.
    """
    chain = chain.copy()
    chain["ts"] = pd.to_datetime(chain["ts"])

    # Restrict to a band around ATM. Far strikes are illiquid and their OI moves
    # are noise, not institutional intent.
    atm = (spot / strike_step).round() * strike_step
    chain = chain.merge(atm.rename("atm"), left_on="ts", right_index=True, how="left")
    chain["moneyness"] = (chain["strike"] - chain["atm"]) / strike_step
    chain = chain[chain["moneyness"].abs() <= n_strikes]

    ce = chain[chain["opt_type"] == "CE"]
    pe = chain[chain["opt_type"] == "PE"]

    feats = pd.DataFrame(index=spot.index)

    ce_oi = ce.groupby("ts")["oi"].sum()
    pe_oi = pe.groupby("ts")["oi"].sum()
    ce_vol = ce.groupby("ts")["volume"].sum()
    pe_vol = pe.groupby("ts")["volume"].sum()

    # --- Put-call ratio and its dynamics -----------------------------------
    # Level is a crowding measure; the CHANGE is the flow signal. Change leads.
    feats["pcr_oi"] = (pe_oi / ce_oi.replace(0, np.nan)).reindex(feats.index)
    feats["pcr_vol"] = (pe_vol / ce_vol.replace(0, np.nan)).reindex(feats.index)

    price_chg = spot.diff()

    for w in windows:
        d_ce = ce_oi.diff(w).reindex(feats.index)
        d_pe = pe_oi.diff(w).reindex(feats.index)

        feats[f"d_ce_oi_{w}"] = d_ce
        feats[f"d_pe_oi_{w}"] = d_pe
        feats[f"pcr_oi_chg_{w}"] = feats["pcr_oi"].diff(w)

        # Net option-writing pressure.
        # Call writing (CE OI up) is bearish — writers are selling upside.
        # Put writing (PE OI up) is bullish — writers are selling downside.
        # This is the cleanest single institutional-intent feature available.
        total = (d_ce.abs() + d_pe.abs()).replace(0, np.nan)
        feats[f"writing_bias_{w}"] = (d_pe - d_ce) / total

        # Buildup score on the underlying's own futures OI.
        pc = price_chg.rolling(w).sum()
        feats[f"buildup_score_{w}"] = buildup_score(pc, d_ce + d_pe)

    # --- Max pain drift ----------------------------------------------------
    # Strike where option writers lose least. Writers are overwhelmingly
    # institutional, and spot tends to gravitate toward it near expiry.
    feats["max_pain"] = _max_pain(chain)
    feats["max_pain_dist"] = (spot - feats["max_pain"]) / spot

    # --- IV skew -----------------------------------------------------------
    # Put IV rich relative to call IV = demand for downside protection.
    if "iv" in chain.columns:
        otm_put_iv = pe[pe["moneyness"] < 0].groupby("ts")["iv"].mean()
        otm_call_iv = ce[ce["moneyness"] > 0].groupby("ts")["iv"].mean()
        feats["iv_skew"] = (otm_put_iv - otm_call_iv).reindex(feats.index)
        feats["iv_skew_chg_5"] = feats["iv_skew"].diff(5)

    return feats


def _max_pain(chain: pd.DataFrame) -> pd.Series:
    """Strike minimising total option-writer payout, per snapshot.

    Vectorised over (timestamp x candidate expiry price x strike): pivoting to
    a dense grid and broadcasting is far faster than a per-group Python loop,
    which matters once this runs over months of 1-minute chain data.
    """
    ce = chain[chain["opt_type"] == "CE"].pivot_table(
        index="ts", columns="strike", values="oi", aggfunc="sum"
    )
    pe = chain[chain["opt_type"] == "PE"].pivot_table(
        index="ts", columns="strike", values="oi", aggfunc="sum"
    )
    if ce.empty or pe.empty:
        return pd.Series(dtype=float)

    strikes = ce.columns.union(pe.columns).to_numpy(dtype=float)
    ce = ce.reindex(columns=strikes).fillna(0.0)
    pe = pe.reindex(columns=strikes).fillna(0.0)

    # payout[e, k]: writer loss at strike k if the index expires at strikes[e].
    diff = strikes[:, None] - strikes[None, :]
    call_payout = np.clip(diff, 0, None)   # calls hurt writers above the strike
    put_payout = np.clip(-diff, 0, None)   # puts hurt writers below it

    total_pain = ce.to_numpy() @ call_payout.T + pe.to_numpy() @ put_payout.T
    return pd.Series(strikes[np.argmin(total_pain, axis=1)], index=ce.index)


# ---------------------------------------------------------------------------
# Put-call parity dislocation
# ---------------------------------------------------------------------------

def parity_dislocation(
    chain: pd.DataFrame,
    *,
    futures: pd.Series,
    spot: pd.Series,
    strike_step: int,
) -> pd.DataFrame:
    """Synthetic futures from put-call parity vs the actual futures price.

    For index options over a short horizon, discounting is negligible:

        F_synthetic = K + (C - P)      at the ATM strike

    When the synthetic trades above the actual future, someone is paying up for
    calls (or dumping puts) in the chain faster than the futures book reprices.
    The gap mean-reverts, and its sign leads short-horizon direction — this is
    one of the few places aggressive institutional option flow is directly
    visible rather than inferred.
    """
    atm = (spot / strike_step).round() * strike_step

    c = chain[chain["opt_type"] == "CE"].set_index(["ts", "strike"])["ltp"]
    p = chain[chain["opt_type"] == "PE"].set_index(["ts", "strike"])["ltp"]

    keys = pd.MultiIndex.from_arrays([atm.index, atm.to_numpy()], names=["ts", "strike"])
    call_px = c.reindex(keys).to_numpy()
    put_px = p.reindex(keys).to_numpy()

    synthetic = atm.to_numpy() + (call_px - put_px)

    out = pd.DataFrame(index=spot.index)
    out["synthetic_fut"] = synthetic
    out["parity_gap"] = synthetic - futures.reindex(spot.index).to_numpy()
    out["parity_gap_z"] = (
        (out["parity_gap"] - out["parity_gap"].rolling(60).mean())
        / out["parity_gap"].rolling(60).std()
    )
    out["parity_gap_chg_5"] = out["parity_gap"].diff(5)
    return out


def futures_basis(futures: pd.Series, spot: pd.Series) -> pd.DataFrame:
    """Futures premium/discount to spot — institutional hedging pressure.

    Persistent discount means sustained hedging or short interest; premium
    expansion means leveraged long demand. The CHANGE matters more than the
    level, since the level drifts with carry as expiry approaches.
    """
    out = pd.DataFrame(index=spot.index)
    out["basis"] = futures - spot
    out["basis_pct"] = out["basis"] / spot
    out["basis_chg_5"] = out["basis"].diff(5)
    out["basis_chg_15"] = out["basis"].diff(15)
    out["basis_z"] = (
        (out["basis"] - out["basis"].rolling(120).mean()) / out["basis"].rolling(120).std()
    )
    return out


# ---------------------------------------------------------------------------
# Participant-wise OI (NSE daily, free)
# ---------------------------------------------------------------------------

def participant_features(participant_oi: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily FII/DII/Pro/Client F&O positioning, forward-filled onto bars.

    NSE publishes this after market close, so it is lagged by a day and cannot
    trigger a 15-minute entry. It is a REGIME FILTER: 'are institutions net long
    index futures today?' conditions how much to trust an intraday long signal.

    Expected columns: date, client_type, future_index_long, future_index_short,
                      option_index_call_long, option_index_put_long, ...

    IMPORTANT: lag by one day before use. The data is not available during the
    session it describes — using same-day values is lookahead bias.
    """
    df = participant_oi.copy()
    df["date"] = pd.to_datetime(df["date"])

    fii = df[df["client_type"] == "FII"].set_index("date")
    dii = df[df["client_type"] == "DII"].set_index("date")

    daily = pd.DataFrame(index=fii.index)
    daily["fii_fut_net"] = fii["future_index_long"] - fii["future_index_short"]
    daily["dii_fut_net"] = dii["future_index_long"] - dii["future_index_short"]
    daily["fii_fut_net_chg"] = daily["fii_fut_net"].diff()

    long_ratio_den = (fii["future_index_long"] + fii["future_index_short"]).replace(0, np.nan)
    daily["fii_long_ratio"] = fii["future_index_long"] / long_ratio_den

    if "option_index_call_long" in fii.columns:
        daily["fii_opt_bias"] = (
            fii["option_index_call_long"] - fii["option_index_put_long"]
        )

    # Lag one day — published after close.
    daily = daily.shift(1)

    out = daily.reindex(index.normalize()).ffill()
    out.index = index
    return out


def institutional_feature_set(
    *,
    spot: pd.Series,
    futures: pd.Series,
    chain: pd.DataFrame,
    strike_step: int,
    cfg: dict,
    participant_oi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble every institutional feature into one matrix indexed by bar."""
    windows = cfg["features"]["oi_windows"]
    n_strikes = cfg["data"]["option_chain_strikes"] // 2

    parts = [
        option_chain_features(
            chain, spot=spot, strike_step=strike_step, windows=windows, n_strikes=n_strikes
        ),
        parity_dislocation(chain, futures=futures, spot=spot, strike_step=strike_step),
        futures_basis(futures, spot),
    ]

    if participant_oi is not None and not participant_oi.empty:
        parts.append(participant_features(participant_oi, spot.index))

    return pd.concat(parts, axis=1)
