"""Robustness gate: does the edge survive arbitrary validation choices?

Why this exists. A single walk-forward configuration reported "+Rs699/trade,
profitable" for the 50/33 model. Changing only the train/test window lengths --
an arbitrary choice with no economic meaning -- moved the measured hit rate
between 44.5% and 56.1%, straddling the 45.4% breakeven. The result flipped
between losing and winning depending on a setting nobody should have to pick
carefully.

A real edge does not behave that way. If the answer moves when you poke the
validation setup, what is being measured is mostly noise.

So: never grade a model on one split. Grade it on a grid, and report the WORST
cell, not the average.

Three traps this guards against:

  1. One lucky split. The grid covers several train/test window pairs.

  2. Mean-of-fold-means. Averaging per-fold hit rates over-weights folds that
     produced three signals as heavily as folds that produced three hundred.
     Hit rates are POOLED across folds here, which is the honest aggregate.
     The gap between pooled and mean-of-means is itself reported, because a
     large gap means the per-fold numbers are being driven by tiny samples.

  3. Seed luck. The model is refit under several random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mcube.validation.purged_cv import walk_forward_splits

# Train/test window pairs, in months. Deliberately varied: if the verdict
# depends on which row you look at, there is no robust edge.
DEFAULT_GRID = ((6, 2), (9, 3), (12, 3), (12, 6))
DEFAULT_SEEDS = (0, 1, 2)

# A cell must clear breakeven by at least this much to count as passing.
# Landing exactly on breakeven is a losing trade after one bad fill.
MARGIN = 0.01


@dataclass
class Cell:
    train_months: int
    test_months: int
    seed: int
    folds: int
    signals: int
    pooled_hit: float          # all signals pooled -- the honest number
    mean_of_folds: float       # per-fold average -- shown for contrast only
    worst_fold: float
    best_fold: float

    def passes(self, breakeven: float) -> bool:
        return self.pooled_hit > breakeven + MARGIN


@dataclass
class RobustnessReport:
    breakeven: float
    base_rate: float
    cells: list[Cell] = field(default_factory=list)

    @property
    def pooled_hits(self) -> np.ndarray:
        return np.array([c.pooled_hit for c in self.cells])

    @property
    def verdict(self) -> str:
        if not self.cells:
            return "NO DATA"
        passes = [c.passes(self.breakeven) for c in self.cells]
        if all(passes):
            return "ROBUST"
        if any(passes):
            return "FRAGILE"
        return "NO EDGE"

    def summary(self) -> dict:
        h = self.pooled_hits
        return {
            "verdict": self.verdict,
            "cells": len(self.cells),
            "cells_passing": sum(c.passes(self.breakeven) for c in self.cells),
            "breakeven": round(self.breakeven, 4),
            "base_rate": round(self.base_rate, 4),
            "pooled_hit_min": round(float(h.min()), 4),
            "pooled_hit_median": round(float(np.median(h)), 4),
            "pooled_hit_max": round(float(h.max()), 4),
            # The decision number. A strategy is only as good as its worst
            # configuration, because you do not get to pick the regime.
            "worst_case_margin": round(float(h.min() - self.breakeven), 4),
        }

    def render(self) -> str:
        L = [
            "=" * 76,
            "  ROBUSTNESS CHECK -- does the edge survive arbitrary split choices?",
            "=" * 76,
            f"  breakeven {self.breakeven*100:.1f}%   base rate {self.base_rate*100:.1f}%"
            f"   (needs +{(self.breakeven-self.base_rate)*100:.1f}pp of real skill)",
            "",
            f"  {'train/test':>11} {'seed':>5} {'folds':>6} {'signals':>8} "
            f"{'pooled':>8} {'meanfold':>9} {'worst':>7} {'best':>7}  verdict",
            "  " + "-" * 72,
        ]
        for c in sorted(self.cells, key=lambda x: (x.train_months, x.test_months, x.seed)):
            ok = "PASS" if c.passes(self.breakeven) else "fail"
            L.append(
                f"  {f'{c.train_months}m/{c.test_months}m':>11} {c.seed:>5} {c.folds:>6} "
                f"{c.signals:>8,} {c.pooled_hit*100:>7.1f}% {c.mean_of_folds*100:>8.1f}% "
                f"{c.worst_fold*100:>6.1f}% {c.best_fold*100:>6.1f}%  {ok}"
            )

        s = self.summary()
        L += [
            "  " + "-" * 72,
            f"  pooled hit across cells: {s['pooled_hit_min']*100:.1f}% .. "
            f"{s['pooled_hit_max']*100:.1f}%  (median {s['pooled_hit_median']*100:.1f}%)",
            f"  worst-case margin over breakeven: {s['worst_case_margin']*100:+.1f}pp",
            "",
            f"  VERDICT: {self.verdict}  ({s['cells_passing']}/{s['cells']} cells pass)",
        ]

        if self.verdict == "ROBUST":
            L.append("    Edge holds in every configuration tested. Necessary, not")
            L.append("    sufficient -- forward paper trading is still required.")
        elif self.verdict == "FRAGILE":
            L.append("    The answer DEPENDS ON THE SPLIT, which means it is mostly noise.")
            L.append("    Do not trade this. Widen the target, add features, or collect more")
            L.append("    data until the verdict stops moving.")
        else:
            L.append("    No configuration clears breakeven. The signal, if any, is too")
            L.append("    small to pay transaction costs.")

        # Pooled and mean-of-folds diverging means per-fold numbers are being
        # driven by folds with very few signals.
        gaps = [abs(c.pooled_hit - c.mean_of_folds) for c in self.cells]
        if gaps and max(gaps) > 0.05:
            L += [
                "",
                f"  NOTE: pooled and mean-of-folds differ by up to {max(gaps)*100:.1f}pp.",
                "    Some folds produce very few signals; their hit rates are noise and",
                "    must not be averaged as equals. Trust the pooled column.",
            ]
        L.append("=" * 76)
        return "\n".join(L)


def robustness_report(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    params: dict,
    breakeven: float,
    quantile: float = 0.95,
    grid: tuple = DEFAULT_GRID,
    seeds: tuple = DEFAULT_SEEDS,
    embargo_bars: int = 90,
    min_signals_per_fold: int = 20,
) -> RobustnessReport:
    """Refit across a grid of split configurations and seeds.

    Args:
        X, y:     feature matrix and binary target, aligned and time-ordered.
        params:   LightGBM params (objective is set here).
        breakeven: hit rate needed to break even after costs.
        quantile: confidence cut. 0.95 keeps the top 5% of signals.
    """
    import lightgbm as lgb

    p = {k: v for k, v in params.items() if k != "objective"}
    report = RobustnessReport(breakeven=breakeven, base_rate=float(y.mean()))

    for train_m, test_m in grid:
        splits = walk_forward_splits(
            X.index, train_months=train_m, test_months=test_m, embargo_bars=embargo_bars
        )
        if not splits:
            continue

        for seed in seeds:
            fold_hits: list[float] = []
            all_true: list[np.ndarray] = []

            for tr, te in splits:
                model = lgb.LGBMClassifier(
                    objective="binary", random_state=seed, **p, verbose=-1
                )
                model.fit(X.iloc[tr], y.iloc[tr])

                proba = model.predict_proba(X.iloc[te])[:, 1]
                y_te = y.iloc[te].to_numpy()

                sel = proba >= np.quantile(proba, quantile)
                if sel.sum() < min_signals_per_fold:
                    continue

                picked = y_te[sel]
                all_true.append(picked)
                fold_hits.append(float(picked.mean()))

            if not all_true:
                continue

            pooled = np.concatenate(all_true)
            report.cells.append(
                Cell(
                    train_months=train_m,
                    test_months=test_m,
                    seed=seed,
                    folds=len(fold_hits),
                    signals=int(pooled.size),
                    pooled_hit=float(pooled.mean()),
                    mean_of_folds=float(np.mean(fold_hits)),
                    worst_fold=float(np.min(fold_hits)),
                    best_fold=float(np.max(fold_hits)),
                )
            )

    return report
