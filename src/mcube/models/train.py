"""Training: LightGBM baseline with purged CV, plus meta-labelling.

Why LightGBM and not a Transformer: on tabular microstructure features with a
low signal-to-noise ratio, gradient boosting beats deep nets at this data scale,
trains in minutes, and exposes feature importances that tell you when the model
has broken. Revisit deep sequence models only once a tabular baseline has shown
a real edge.

Two-stage design (Lopez de Prado, ch. 3):
  PRIMARY  predicts direction (up / down).
  META     predicts whether the primary's call is worth taking.

The meta stage is where ML earns its keep. Improving direction accuracy past
~55% is very hard; learning to SKIP the bad setups is comparatively tractable
and does more for net P&L, because every avoided trade saves a full round trip
of costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from mcube.validation.purged_cv import PurgedKFold, assert_no_leakage

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    accuracy: float
    auc: float
    precision_at_threshold: float
    n_signals: int


@dataclass
class TrainResult:
    primary: object
    meta: object | None
    folds: list[FoldResult] = field(default_factory=list)
    feature_importance: pd.Series | None = None

    def summary(self) -> dict:
        if not self.folds:
            return {}
        acc = np.array([f.accuracy for f in self.folds])
        auc = np.array([f.auc for f in self.folds])
        prec = np.array([f.precision_at_threshold for f in self.folds])
        return {
            "n_folds": len(self.folds),
            "accuracy_mean": round(acc.mean(), 4),
            "accuracy_std": round(acc.std(), 4),
            "auc_mean": round(auc.mean(), 4),
            "auc_std": round(auc.std(), 4),
            "precision_mean": round(prec.mean(), 4),
            "total_signals": int(sum(f.n_signals for f in self.folds)),
        }


def _check_sane(result: TrainResult) -> list[str]:
    """Flag results that are too good to be true.

    An AUC above ~0.65 on 1-15 minute index direction is not a breakthrough; it
    is almost always leakage. Common causes: a feature computed with centred or
    forward-looking windows, labels not purged from training folds, or features
    joined on a timestamp that is not strictly in the past.
    """
    warnings: list[str] = []
    s = result.summary()
    if not s:
        return warnings

    if s["auc_mean"] > 0.65:
        warnings.append(
            f"AUC {s['auc_mean']:.3f} is implausibly high for this problem. "
            "Suspect leakage: check for forward-looking feature windows and "
            "confirm purging actually ran."
        )
    if s["accuracy_mean"] > 0.62:
        warnings.append(
            f"Accuracy {s['accuracy_mean']:.3f} exceeds what this signal class "
            "plausibly supports. Audit the feature pipeline before trusting it."
        )
    if s["accuracy_std"] < 0.005 and len(result.folds) > 3:
        warnings.append(
            "Accuracy is suspiciously stable across folds — regimes should "
            "produce more variation. Check that folds are genuinely disjoint."
        )
    return warnings


def train_primary(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    cfg: dict,
) -> TrainResult:
    """Train the direction model under purged K-fold CV.

    Args:
        X:      feature matrix indexed by event timestamp.
        labels: output of `triple_barrier_labels` (needs 'label' and 't1').
    """
    if lgb is None:
        raise ImportError("lightgbm is required: pip install lightgbm")

    # Drop vertical-barrier (0) samples: direction is undefined when neither
    # barrier was touched. They return in the meta stage as "do not trade".
    resolved = labels[labels["label"] != 0]
    common = X.index.intersection(resolved.index)

    X_use = X.loc[common].copy()
    y = (resolved.loc[common, "label"] > 0).astype(int)
    t1 = resolved.loc[common, "t1"]

    X_use = X_use.replace([np.inf, -np.inf], np.nan)

    cv = PurgedKFold(
        n_splits=cfg["validation"]["n_splits"],
        t1=t1,
        embargo_bars=cfg["validation"]["embargo_bars"],
    )

    params = dict(cfg["model"]["lightgbm"])
    threshold = cfg["model"]["meta_labeling"]["min_confidence"]

    folds: list[FoldResult] = []
    importances: list[pd.Series] = []
    model = None

    for i, (train_idx, test_idx) in enumerate(cv.split(X_use)):
        assert_no_leakage(train_idx, test_idx, t1)

        X_tr, y_tr = X_use.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X_use.iloc[test_idx], y.iloc[test_idx]

        model = lgb.LGBMClassifier(**params, verbose=-1)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba > 0.5).astype(int)

        # Precision among high-confidence calls — the trades actually taken.
        confident = (proba > threshold) | (proba < 1 - threshold)
        if confident.sum() > 0:
            conf_pred = pred[confident]
            conf_true = y_te.to_numpy()[confident]
            precision = accuracy_score(conf_true, conf_pred)
        else:
            precision = float("nan")

        folds.append(
            FoldResult(
                fold=i,
                n_train=len(train_idx),
                n_test=len(test_idx),
                accuracy=accuracy_score(y_te, pred),
                auc=roc_auc_score(y_te, proba) if y_te.nunique() > 1 else float("nan"),
                precision_at_threshold=precision,
                n_signals=int(confident.sum()),
            )
        )
        importances.append(pd.Series(model.feature_importances_, index=X_use.columns))

    result = TrainResult(
        primary=model,
        meta=None,
        folds=folds,
        feature_importance=pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        if importances
        else None,
    )

    for w in _check_sane(result):
        print(f"  [WARNING] {w}")

    return result


def train_meta(
    X: pd.DataFrame,
    labels: pd.DataFrame,
    primary_side: pd.Series,
    *,
    cfg: dict,
) -> TrainResult:
    """Train the meta model: given the primary's direction, should we trade?

    The meta target is binary — did the primary's call reach the profit target
    before the stop. Features are the primary features plus the primary's own
    confidence, so the meta model can learn where the primary is unreliable
    (typically the midday lull and low-volatility regimes).
    """
    if lgb is None:
        raise ImportError("lightgbm is required: pip install lightgbm")

    common = X.index.intersection(labels.index).intersection(primary_side.index)
    X_meta = X.loc[common].copy()
    X_meta["primary_side"] = primary_side.loc[common]

    # 1 = primary was right and the trade resolved profitably; 0 = everything
    # else, including vertical-barrier timeouts (a timeout still costs a round
    # trip, so it belongs in the "should not have traded" class).
    lab = labels.loc[common]
    y = ((lab["label"] == primary_side.loc[common]) & (lab["label"] != 0)).astype(int)
    t1 = lab["t1"]

    cv = PurgedKFold(
        n_splits=cfg["validation"]["n_splits"],
        t1=t1,
        embargo_bars=cfg["validation"]["embargo_bars"],
    )

    params = dict(cfg["model"]["lightgbm"])
    folds: list[FoldResult] = []
    model = None

    for i, (train_idx, test_idx) in enumerate(cv.split(X_meta)):
        assert_no_leakage(train_idx, test_idx, t1)

        model = lgb.LGBMClassifier(**params, verbose=-1)
        model.fit(X_meta.iloc[train_idx], y.iloc[train_idx])

        proba = model.predict_proba(X_meta.iloc[test_idx])[:, 1]
        pred = (proba > 0.5).astype(int)
        y_te = y.iloc[test_idx]

        take = proba > cfg["model"]["meta_labeling"]["min_confidence"]
        folds.append(
            FoldResult(
                fold=i,
                n_train=len(train_idx),
                n_test=len(test_idx),
                accuracy=accuracy_score(y_te, pred),
                auc=roc_auc_score(y_te, proba) if y_te.nunique() > 1 else float("nan"),
                precision_at_threshold=y_te[take].mean() if take.sum() else float("nan"),
                n_signals=int(take.sum()),
            )
        )

    return TrainResult(primary=None, meta=model, folds=folds)
