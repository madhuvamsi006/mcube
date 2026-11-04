"""Train on real Dhan history. This is the moment of truth.

    python scripts/train_real.py --symbol NIFTY --target 20 --stop 13
    python scripts/train_real.py --shuffle-test     # leakage check on real data

The shuffle test is the real-data equivalent of the synthetic null test: it
destroys the time-alignment between features and labels while keeping every
other step identical. AUC must collapse to ~0.50. If it does not, the pipeline
is leaking and no other number here means anything.

Report AUC, not accuracy. Triple-barrier labels are imbalanced by construction
(a nearer stop is hit more often), so accuracy flatters a model that has learned
nothing but the base rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcube.costs.india_costs import (
    breakeven_hit_rate,
    cost_in_index_points,
    expectancy_rupees,
    random_walk_base_rate,
)
from mcube.features.bars import bar_features, session_features
from mcube.labeling.triple_barrier import label_summary, triple_barrier_labels
from mcube.models.train import train_primary
from mcube.validation.robustness import robustness_report

# Window pairs are deliberately varied. If the verdict depends on which
# row you read, there is no robust edge.
ROBUST_GRID = ((6, 2), (9, 3), (12, 3), (12, 6))
ROBUST_SEEDS = (0, 1, 2)

DATA_DIR = Path("data/raw")


def load(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_spot_1min.parquet"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/fetch_history.py first.")
    df = pd.read_parquet(path).sort_index()

    # Keep only regular session bars; pre/post prints have different dynamics.
    t = df.index.time
    mask = (t >= pd.Timestamp("09:15").time()) & (t <= pd.Timestamp("15:30").time())
    return df[mask]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NIFTY")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--stop", type=float, default=None)
    # A wide target on a short horizon mostly times out, and the subset that
    # does resolve is a high-volatility selection that inflates AUC. Scale the
    # horizon with the target.
    ap.add_argument("--horizon", type=int, default=None,
                    help="vertical barrier in bars; must scale with the target")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--save", type=Path, default=None,
                    help="persist the fitted model + feature list for live use")
    ap.add_argument("--skip-robustness", action="store_true",
                    help="skip the robustness grid (faster, but no verdict)")
    ap.add_argument("--shuffle-test", action="store_true",
                    help="break feature/label alignment; AUC must fall to ~0.50")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.target:
        cfg["trade"]["points"][args.symbol] = {"target": args.target, "stop": args.stop or args.target / 1.5}
    if args.horizon:
        cfg["trade"]["horizon_bars"] = args.horizon
        cfg["validation"]["embargo_bars"] = max(args.horizon * 2, 30)

    pts = cfg["trade"]["points"][args.symbol]
    target, stop = float(pts["target"]), float(pts["stop"])

    print("=" * 72)
    print(f"  {args.symbol}  target {target:g} / stop {stop:g} pts  "
          f"horizon {cfg['trade']['horizon_bars']} min")
    if args.shuffle_test:
        print("  *** SHUFFLE TEST -- expect AUC ~0.50 ***")
    print("=" * 72)

    df = load(args.symbol)
    print(f"\n[1/4] {len(df):,} bars | {df.index.min().date()} .. {df.index.max().date()} "
          f"| {df.index.normalize().nunique():,} sessions")

    print("[2/4] features...")
    X = pd.concat([bar_features(df), session_features(df.index)], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().sum() > len(X) * 0.5]
    print(f"      {X.shape[1]} features")

    print("[3/4] labels...")
    labels = triple_barrier_labels(df["close"], cfg=cfg, symbol=args.symbol)
    s = label_summary(labels)
    base = random_walk_base_rate(target_pts=target, stop_pts=stop)
    print(f"      {s['n']:,} labels | target {s['target_hits']:,} | stop {s['stop_hits']:,} "
          f"| timeout {s['vertical_pct']}%")
    print(f"      actual base rate {s['target_rate']}%  (random walk predicts {base*100:.1f}%)")
    if s['vertical_pct'] > 50:
        print(f"      [WARNING] {s['vertical_pct']}% of labels timed out. The horizon is too")
        print(f"                short for a {target:g}-point target. The resolved subset is a")
        print(f"                high-volatility selection, which INFLATES AUC. Widen --horizon.")

    if args.shuffle_test:
        # Permute ONLY the label values. Index and t1 must stay intact: they
        # drive the purging, and scrambling them breaks the CV itself rather
        # than testing it (which inflates AUC and falsely reports a leak).
        rng = np.random.default_rng(0)
        labels = labels.copy()
        labels["label"] = rng.permutation(labels["label"].to_numpy())

    print("[4/4] training (purged CV)...")
    result = train_primary(X, labels, cfg=cfg)
    r = result.summary()

    print("\n" + "-" * 72)
    print(f"  AUC            {r['auc_mean']:.4f} +/- {r['auc_std']:.4f}   <-- judge on this")
    print(f"  accuracy       {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}")
    print(f"  precision@conf {r['precision_mean']:.4f}  on {r['total_signals']:,} signals")
    print("-" * 72)

    if args.shuffle_test:
        verdict = "PASS - no leakage" if r["auc_mean"] < 0.52 else "FAIL - LEAKING"
        print(f"\n  SHUFFLE TEST: {verdict} (AUC {r['auc_mean']:.4f})\n")
        return

    # Persist for the paper trader. The FEATURE LIST is saved alongside the
    # model: live must build exactly these columns in exactly this order, or
    # predictions drift from what was validated without any error being raised.
    if args.save:
        import joblib

        args.save.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": result.primary,
                "features": list(X.columns),
                "symbol": args.symbol,
                "target": target,
                "stop": stop,
                "horizon_bars": cfg["trade"]["horizon_bars"],
                "trained_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
                "train_rows": int(len(X)),
                "cv_auc": r["auc_mean"],
                "cv_precision": r["precision_mean"],
                "base_rate": s["target_rate"],
            },
            args.save,
        )
        print(f"\n  saved model -> {args.save}  ({X.shape[1]} features)")

    if result.feature_importance is not None:
        print("\n  top features:")
        for n, imp in result.feature_importance.head(10).items():
            print(f"    {n:<24} {imp:>8.0f}")

    # --- cost floor --------------------------------------------------------
    spot = float(df["close"].iloc[-1])
    print("\n" + "=" * 76)
    print("  COST FLOOR")
    print("=" * 76)
    costs = {}
    for inst in ("futures", "options"):
        cost = cost_in_index_points(symbol=args.symbol, spot=spot, cfg=cfg,
                                    instrument=inst, lots=cfg["trade"]["lots"])
        be = breakeven_hit_rate(target_pts=target, stop_pts=stop, cost_pts=cost)
        costs[inst] = (cost, be)
        print(f"  {inst:<9} round-trip {cost:5.1f} pts  ->  breakeven {be*100:5.1f}%")

    # --- robustness --------------------------------------------------------
    # Deliberately NOT reporting a verdict from the single CV number above.
    # That is how "profitable" was claimed for a model whose answer flips with
    # the split. The grid below is the number that decides.
    if args.skip_robustness:
        print("\n  [robustness check SKIPPED -- no tradeability verdict produced]\n")
        return

    inst = cfg["trade"]["instrument"]
    cost_pts, be = costs[inst]
    print(f"\n  running robustness grid on {inst} "
          f"({len(ROBUST_GRID)} window pairs x {len(ROBUST_SEEDS)} seeds)...")

    rep = robustness_report(
        X.loc[X.index.intersection(labels[labels.label != 0].index)],
        (labels.loc[X.index.intersection(labels[labels.label != 0].index), "label"] > 0).astype(int),
        params=cfg["model"]["lightgbm"],
        breakeven=be,
        grid=ROBUST_GRID,
        seeds=ROBUST_SEEDS,
        embargo_bars=max(cfg["trade"]["horizon_bars"] * 2, 90),
    )
    print("\n" + rep.render())

    s = rep.summary()
    if rep.verdict == "ROBUST":
        rs = expectancy_rupees(
            hit_rate=s["pooled_hit_min"], target_pts=target, stop_pts=stop,
            cost_pts=cost_pts, lot_size=cfg["instruments"][args.symbol]["lot_size"],
            lots=cfg["trade"]["lots"],
        )
        print(f"\n  Worst-case expectancy: Rs {rs:,.0f}/trade at {cfg['trade']['lots']} lots.")
        print("  Next gate is forward paper trading, not more backtesting.\n")
    else:
        print("\n  NOT TRADEABLE. Forward paper trading will not rescue a model whose")
        print("  backtested answer depends on the split -- fix the edge first.\n")


if __name__ == "__main__":
    main()
