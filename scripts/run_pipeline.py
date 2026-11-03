"""End-to-end pipeline run: data -> features -> labels -> train -> backtest.

Run the NULL test first, every time:

    python scripts/run_pipeline.py --mode null      # MUST report ~50%
    python scripts/run_pipeline.py --mode signal    # should report ~54-58%

If null reports materially above 50%, the pipeline is leaking and no result
from signal mode means anything. Fix that before reading any other number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcube.backtest.engine import run_backtest
from mcube.data.synthetic import generate
from mcube.features.institutional import institutional_feature_set
from mcube.features.microstructure import microstructure_features
from mcube.features.regime import time_of_day_features, volatility_regime
from mcube.labeling.triple_barrier import label_summary, triple_barrier_labels
from mcube.models.train import train_primary


def build_features(data: dict, cfg: dict) -> pd.DataFrame:
    spot, futures = data["spot"], data["futures"]

    micro = microstructure_features(data["depth"], cfg=cfg)
    inst = institutional_feature_set(
        spot=spot,
        futures=futures,
        chain=data["chain"],
        strike_step=cfg["instruments"]["NIFTY"]["strike_step"],
        cfg=cfg,
    )
    tod = time_of_day_features(spot.index)
    vol = volatility_regime(spot)

    X = pd.concat([micro, inst, tod, vol], axis=1)

    # Option chain is sampled every 5 min; forward-fill ONLY (never bfill —
    # backfilling would pull future values into the past, which is leakage).
    X = X.ffill()

    # Drop columns that are constant or entirely missing.
    X = X.loc[:, X.notna().any()]
    # Drop raw price levels: non-stationary, and they let the model memorise
    # which time period a bar came from rather than learning structure.
    X = X.drop(columns=[c for c in ("mid", "microprice", "max_pain", "synthetic_fut") if c in X], errors="ignore")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    return X


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signal", "null"], default="signal")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    banner = "NULL TEST (expect ~50% — anything higher means leakage)" if args.mode == "null" \
        else "SIGNAL TEST (expect ~54-58%)"
    print(f"\n{'=' * 70}\n  {banner}\n{'=' * 70}")

    print(f"\n[1/5] generating {args.days} sessions of synthetic tape ({args.mode} mode)...")
    data = generate(n_days=args.days, mode=args.mode)
    print(f"      {len(data['spot']):,} bars, {len(data['chain']):,} chain rows")

    print("[2/5] building features...")
    X = build_features(data, cfg)
    print(f"      {X.shape[1]} features x {X.shape[0]:,} bars")

    print("[3/5] triple-barrier labelling...")
    labels = triple_barrier_labels(data["spot"], cfg=cfg, symbol="NIFTY")
    summary = label_summary(labels)
    print(f"      {summary['n']:,} labels | target {summary['target_hits']:,} "
          f"| stop {summary['stop_hits']:,} | vertical {summary['vertical_pct']}%")
    print(f"      base rate (resolved): {summary['target_rate']}%  <- model must beat this")

    print("[4/5] training with purged CV...")
    result = train_primary(X, labels, cfg=cfg)
    s = result.summary()
    print(f"      accuracy {s['accuracy_mean']:.3f} +/- {s['accuracy_std']:.3f}"
          f" | AUC {s['auc_mean']:.3f} | folds {s['n_folds']}")

    if result.feature_importance is not None:
        print("\n      top features:")
        for name, imp in result.feature_importance.head(8).items():
            print(f"        {name:<28} {imp:>8.1f}")

    print("\n[5/5] walk-forward backtest...")
    proba = result.primary.predict_proba(X.replace([np.inf, -np.inf], np.nan))[:, 1]
    thresh = cfg["model"]["meta_labeling"]["min_confidence"]
    signals = pd.Series(
        np.where(proba > thresh, 1, np.where(proba < 1 - thresh, -1, 0)), index=X.index
    )

    bt = run_backtest(data["spot"], signals, cfg=cfg, symbol="NIFTY")
    stats = bt.stats()

    print(f"\n{'-' * 70}")
    for k, v in stats.items():
        print(f"  {k:<22} {v}")
    print(f"{'-' * 70}")

    if args.mode == "null":
        # AUC, not accuracy. Triple-barrier labels are imbalanced by
        # construction (a nearer stop is hit more often), so accuracy on
        # null data sits well above 50% with zero skill. AUC is invariant
        # to that base rate, which makes it the correct leakage check.
        auc = s["auc_mean"]
        if auc > 0.52:
            print("")
            print("  *** FAIL: AUC %.3f on NULL data. Pipeline is leaking. ***" % auc)
            print("      Investigate before trusting any signal-mode result.")
            sys.exit(1)
        print("")
        print("  PASS: AUC %.3f on null data - no leakage detected." % auc)
        print("        accuracy %.3f is class imbalance, not skill."
              % s["accuracy_mean"])
    else:
        print("\n  Backtest P&L here is on SYNTHETIC data. It validates the\n"
              "  machinery, not the strategy. Real numbers need real tape.\n")


if __name__ == "__main__":
    main()
