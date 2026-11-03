"""Triple-barrier labelling (Lopez de Prado, *Advances in Financial ML*, ch. 3).

A fixed-horizon label ("+1 if the 15-min return is positive") ignores path, and
so does not describe the trade you actually take. Triple-barrier labels answer
the question the strategy asks: *did price hit the profit target before the stop,
within the time limit?*

That makes the label identical to the trade spec in config.trade — the model is
trained on exactly the thing being traded.

Labels:
     1  profit target hit first
    -1  stop loss hit first
     0  neither — vertical (time) barrier expired
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewm_volatility(close: pd.Series, span: int = 60) -> pd.Series:
    """EWM stdev of 1-bar returns — used to scale barriers to current regime."""
    returns = close.pct_change()
    return returns.ewm(span=span, min_periods=span // 2).std()


def _barriers_in_points(
    close: pd.Series,
    cfg: dict,
    symbol: str,
) -> tuple[pd.Series, pd.Series]:
    """Resolve target/stop barriers into absolute points, per bar.

    'points' mode is a flat target (what the desk thinks in).
    'vol' mode scales by EWM sigma, so the trade adapts to the volatility
    regime — a flat 20 points is a very different bet at 10 VIX than at 20.
    """
    trade = cfg["trade"]

    if trade["target_mode"] == "points":
        pts = trade["points"][symbol]
        target = pd.Series(float(pts["target"]), index=close.index)
        stop = pd.Series(float(pts["stop"]), index=close.index)
        return target, stop

    vol_cfg = trade["vol"]
    sigma = ewm_volatility(close, span=vol_cfg["vol_lookback"])
    sigma_pts = sigma * close  # fractional sigma -> points
    return sigma_pts * vol_cfg["target_mult"], sigma_pts * vol_cfg["stop_mult"]


def triple_barrier_labels(
    close: pd.Series,
    *,
    cfg: dict,
    symbol: str,
    side: pd.Series | None = None,
    event_idx: pd.Index | None = None,
) -> pd.DataFrame:
    """Label every event bar by which barrier is touched first.

    Args:
        close:     1-min close series of the UNDERLYING, DatetimeIndex, tz-aware.
        cfg:       parsed config.yaml
        symbol:    NIFTY | BANKNIFTY | SENSEX
        side:      optional +1/-1 per event. When given, barriers are applied in
                   the direction of the bet (meta-labelling: the primary model
                   picks side, this labels whether taking it was correct).
        event_idx: bars to label. Defaults to every bar.

    Returns DataFrame indexed by event time:
        t1          timestamp the label resolved (for CV purging — essential)
        label       +1 / -1 / 0
        barrier     'target' | 'stop' | 'vertical'
        ret_pts     realised move in index points at t1
        target_pts  barrier used
        stop_pts    barrier used
        bars_held   how long the trade lasted
    """
    if event_idx is None:
        event_idx = close.index

    horizon = int(cfg["trade"]["horizon_bars"])
    target_pts, stop_pts = _barriers_in_points(close, cfg, symbol)

    px = close.to_numpy(dtype=float)
    n = len(px)
    pos_of = {ts: i for i, ts in enumerate(close.index)}

    out: list[dict] = []
    for ts in event_idx:
        i = pos_of.get(ts)
        if i is None or i >= n - 1:
            continue

        tgt = target_pts.iloc[i]
        stp = stop_pts.iloc[i]
        if not np.isfinite(tgt) or not np.isfinite(stp):
            continue  # vol warmup

        direction = 1.0 if side is None else float(side.get(ts, 1.0))
        entry = px[i]

        # Barriers in the direction of the bet.
        upper = entry + tgt * direction
        lower = entry - stp * direction

        end = min(i + horizon, n - 1)
        path = px[i + 1 : end + 1]
        if path.size == 0:
            continue

        if direction > 0:
            hit_target = np.flatnonzero(path >= upper)
            hit_stop = np.flatnonzero(path <= lower)
        else:
            hit_target = np.flatnonzero(path <= upper)
            hit_stop = np.flatnonzero(path >= lower)

        first_target = hit_target[0] if hit_target.size else np.inf
        first_stop = hit_stop[0] if hit_stop.size else np.inf

        if first_target < first_stop:
            k, label, barrier = int(first_target), 1, "target"
        elif first_stop < first_target:
            k, label, barrier = int(first_stop), -1, "stop"
        else:
            k, label, barrier = len(path) - 1, 0, "vertical"

        t1 = close.index[i + 1 + k]
        out.append(
            {
                "event": ts,
                "t1": t1,
                "label": label,
                "barrier": barrier,
                "ret_pts": (px[i + 1 + k] - entry) * direction,
                "target_pts": tgt,
                "stop_pts": stp,
                "bars_held": k + 1,
            }
        )

    if not out:
        return pd.DataFrame(
            columns=["t1", "label", "barrier", "ret_pts", "target_pts", "stop_pts", "bars_held"]
        )

    return pd.DataFrame(out).set_index("event")


def label_summary(labels: pd.DataFrame) -> dict:
    """Sanity stats. Check these BEFORE training anything.

    What to look for:
      - A large `vertical` share means the target is too wide for the horizon;
        most trades will time out rather than resolve.
      - target_rate far from ~50% means the barriers are asymmetric in practice,
        so a naive accuracy number will be misleading.
    """
    n = len(labels)
    if n == 0:
        return {"n": 0}

    counts = labels["barrier"].value_counts()
    resolved = labels[labels["label"] != 0]

    return {
        "n": n,
        "target_hits": int(counts.get("target", 0)),
        "stop_hits": int(counts.get("stop", 0)),
        "vertical": int(counts.get("vertical", 0)),
        "vertical_pct": round(100 * counts.get("vertical", 0) / n, 1),
        # Base rate among resolved trades — the number the model must beat.
        "target_rate": round(100 * (resolved["label"] == 1).mean(), 2) if len(resolved) else None,
        "avg_bars_held": round(labels["bars_held"].mean(), 1),
        "avg_win_pts": round(labels.loc[labels.label == 1, "ret_pts"].mean(), 1)
        if (labels.label == 1).any()
        else None,
        "avg_loss_pts": round(labels.loc[labels.label == -1, "ret_pts"].mean(), 1)
        if (labels.label == -1).any()
        else None,
    }
