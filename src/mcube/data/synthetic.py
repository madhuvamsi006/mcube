"""Synthetic market generator — pipeline validation before real data exists.

Purpose: prove the feature -> label -> train -> backtest chain is correct and
leak-free BEFORE a single real tick is collected. Swapping in live data then
becomes a one-line adapter change rather than a debugging expedition.

Two modes, and running both is the whole point:

  SIGNAL mode  A latent "institutional pressure" process drives BOTH the
               observable features (noisily) and future returns. A correct
               pipeline should recover roughly 54-58% accuracy.

  NULL mode    Features are pure noise, independent of returns. A correct
               pipeline MUST report ~50%. Anything materially above 50% on
               null data is proof of leakage in the pipeline itself.

The null test is the more valuable of the two. It is the only way to prove your
backtest is not lying to you, and it costs nothing to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _session_index(n_days: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    """1-min bars across NSE sessions (09:15-15:30), weekdays only."""
    days = pd.bdate_range(start, periods=n_days, tz="Asia/Kolkata")
    stamps = []
    for d in days:
        stamps.append(
            pd.date_range(
                d + pd.Timedelta(hours=9, minutes=15),
                d + pd.Timedelta(hours=15, minutes=30),
                freq="1min",
            )
        )
    return pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in stamps])).tz_convert(
        "Asia/Kolkata"
    )


def generate(
    n_days: int = 180,
    *,
    spot0: float = 26000.0,
    mode: str = "signal",
    signal_strength: float = 0.22,
    seed: int = 7,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Generate a synthetic NIFTY tape with depth, option chain and futures.

    Args:
        mode: 'signal' (recoverable edge) or 'null' (no edge — leakage test)
        signal_strength: correlation between latent pressure and future drift.
            0.22 with the noise levels below lands near a realistic AUC of
            0.55-0.60. Higher values flatter the pipeline and teach you
            nothing about whether it would work on real tape.
    """
    rng = np.random.default_rng(seed)
    idx = _session_index(n_days)
    n = len(idx)

    # --- latent institutional pressure: slow, mean-reverting --------------
    pressure = np.zeros(n)
    theta, sigma_p = 0.02, 1.0
    for i in range(1, n):
        pressure[i] = pressure[i - 1] * (1 - theta) + rng.normal(0, sigma_p)
    pressure /= pressure.std()

    # --- returns -----------------------------------------------------------
    # Intraday vol ~0.7%/day over 375 bars.
    base_vol = 0.007 / np.sqrt(375)
    noise = rng.normal(0, base_vol, n)

    if mode == "signal":
        # Pressure at time t drives the drift realised AFTER t. The feature is
        # observable at t; the payoff is not. This is a legitimate predictive
        # relationship, not leakage.
        drift = signal_strength * base_vol * pressure
        returns = drift + noise * np.sqrt(1 - min(signal_strength**2, 0.9))
    elif mode == "null":
        returns = noise  # features will be built from pressure, which is independent
    else:
        raise ValueError(f"mode must be 'signal' or 'null', got {mode!r}")

    spot = pd.Series(spot0 * np.exp(np.cumsum(returns)), index=idx, name="spot")

    # --- depth snapshots, consistent with pressure -------------------------
    depth = pd.DataFrame(index=idx)
    mid = spot.to_numpy()
    tick = 0.05
    spread_ticks = rng.integers(1, 4, n)

    # Observable imbalance = pressure + measurement noise. The model must dig
    # the signal out of the noise, exactly as it would on a real book.
    obs_imb = np.tanh(0.25 * pressure + rng.normal(0, 1.6, n))

    for lv in range(1, 6):
        depth[f"bid_px_{lv}"] = mid - tick * (spread_ticks / 2 + lv - 1)
        depth[f"ask_px_{lv}"] = mid + tick * (spread_ticks / 2 + lv - 1)
        base_qty = rng.lognormal(7.5 - 0.15 * lv, 0.5, n)
        depth[f"bid_qty_{lv}"] = base_qty * (1 + 0.5 * obs_imb)
        depth[f"ask_qty_{lv}"] = base_qty * (1 - 0.5 * obs_imb)

    depth["ltp"] = mid + rng.choice([-tick, 0, tick], n) * spread_ticks / 2
    depth["ltq"] = rng.lognormal(4, 1, n)
    depth["volume"] = np.cumsum(rng.lognormal(6, 0.8, n))
    depth["oi"] = 1e7 + np.cumsum(rng.normal(0, 5e3, n))

    # --- futures: spot + carry + pressure-driven basis ---------------------
    # Basis carries only a faint trace of the latent state; on real tape it is
    # dominated by carry, hedging demand and plain noise.
    futures = spot * (1 + 0.0004) + pressure * 0.4 + rng.normal(0, 4.0, n)
    futures.name = "futures"

    # --- option chain: OI responds to pressure -----------------------------
    chain = _generate_chain(idx, spot, pressure, rng)

    return {"spot": spot, "futures": futures, "depth": depth, "chain": chain}


def _generate_chain(
    idx: pd.DatetimeIndex, spot: pd.Series, pressure: np.ndarray, rng
) -> pd.DataFrame:
    """ATM +/- 5 strike chain, sampled every 5 minutes to keep size sane.

    OI moves with pressure: positive pressure means put writing (bullish),
    negative means call writing (bearish) — the mechanism the institutional
    features are designed to detect.
    """
    step = 50
    sample = idx[::5]
    pos = np.arange(len(idx))[::5]

    rows = []
    for k, ts in enumerate(sample):
        s = spot.iloc[pos[k]]
        p = pressure[pos[k]]
        atm = round(s / step) * step

        for off in range(-5, 6):
            strike = atm + off * step
            moneyness = (strike - s) / s

            for opt in ("CE", "PE"):
                intrinsic = max(s - strike, 0) if opt == "CE" else max(strike - s, 0)
                tv = s * 0.004 * np.exp(-((moneyness * 30) ** 2))
                ltp = max(intrinsic + tv + rng.normal(0, 0.3), 0.05)

                # Put writing on positive pressure, call writing on negative.
                bias = p if opt == "PE" else -p
                # Deliberately low SNR: OI is a weak, noisy proxy for intent.
                oi = max(rng.lognormal(10, 0.9) * (1 + 0.05 * bias), 1.0)

                rows.append(
                    {
                        "ts": ts,
                        "strike": strike,
                        "opt_type": opt,
                        "ltp": ltp,
                        "oi": oi,
                        "volume": rng.lognormal(8, 0.6),
                        "iv": 0.13 + 0.02 * abs(moneyness * 20) + rng.normal(0, 0.005),
                    }
                )

    return pd.DataFrame(rows)
