"""Purged K-fold cross-validation with embargo.

THE most important file in this repo. Read before using it.

Triple-barrier labels span multiple bars: a label stamped 10:00 may not resolve
until 10:15. A random train/test split puts the 10:00 sample in train and the
10:05 sample in test while their label windows overlap — the training set then
contains the test set's future. Models trained that way show 70%+ accuracy and
Sharpe above 5, and make money only in the backtest.

If a first backtest looks spectacular, suspect this before celebrating. It is
the single most common failure mode in retail quant research.

Two guards (Lopez de Prado, ch. 7):
  PURGE   drop training samples whose label window overlaps the test window.
  EMBARGO drop training samples for a further E bars after the test window,
          since serial correlation leaks a little past the boundary.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """K-fold over time-ordered samples with purging and embargo.

    Folds are contiguous blocks of time, never shuffled.

    Args:
        n_splits: number of folds.
        t1:       Series mapping event start -> label resolution time.
                  Comes straight from `triple_barrier_labels()['t1']`.
        embargo_bars: bars to embargo after each test fold. Should be >= the
                  label horizon; config uses 2x for margin.
    """

    def __init__(self, n_splits: int = 6, t1: pd.Series | None = None, embargo_bars: int = 30):
        if t1 is None:
            raise ValueError("PurgedKFold requires t1 (label end times) to purge correctly.")
        if not isinstance(t1.index, pd.DatetimeIndex):
            raise ValueError("t1 must be indexed by event timestamp (DatetimeIndex).")

        self.n_splits = n_splits
        self.t1 = t1.sort_index()
        self.embargo_bars = embargo_bars

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if len(X) != len(self.t1):
            raise ValueError(
                f"X ({len(X)}) and t1 ({len(self.t1)}) must align. "
                "Pass the feature matrix indexed by the same events as the labels."
            )

        indices = np.arange(len(X))
        embargo = self.embargo_bars
        test_blocks = np.array_split(indices, self.n_splits)

        for test_idx in test_blocks:
            if test_idx.size == 0:
                continue

            test_start = self.t1.index[test_idx[0]]
            # Test fold occupies [test_start, test_end]: both the event times and
            # the time their labels resolve.
            test_end = self.t1.iloc[test_idx].max()

            # PURGE: drop any train sample whose label window [event, t1] overlaps
            # the test window at all.
            train_mask = (self.t1 <= test_start) | (self.t1.index >= test_end)

            # EMBARGO: additionally drop train samples starting just after the
            # test window, where serial correlation still bleeds through.
            if embargo > 0:
                after = self.t1.index.searchsorted(test_end)
                embargo_end_pos = min(after + embargo, len(self.t1) - 1)
                embargo_end = self.t1.index[embargo_end_pos]
                train_mask &= ~(
                    (self.t1.index > test_end) & (self.t1.index <= embargo_end)
                )

            train_idx = indices[train_mask.to_numpy()]
            if train_idx.size == 0:
                continue

            yield train_idx, test_idx


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    train_months: int = 6,
    test_months: int = 1,
    embargo_bars: int = 30,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward splits — the honest backtest.

    Train on `train_months`, trade the next `test_months`, roll forward. Unlike
    K-fold this never trains on data after the test period, so it reflects how
    the model would actually have been deployed.

    Report walk-forward results, not K-fold results. K-fold is for hyperparameter
    selection only.
    """
    # Drop tz before to_period: month bucketing is tz-independent here, and
    # pandas warns on the implicit conversion otherwise.
    naive = index.tz_localize(None) if index.tz is not None else index
    index_periods = naive.to_period("M")
    periods = index_periods.unique()
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    pos = np.arange(len(index))

    for i in range(train_months, len(periods), test_months):
        train_periods = periods[max(0, i - train_months) : i]
        test_periods = periods[i : i + test_months]
        if len(test_periods) == 0:
            break

        train_mask = index_periods.isin(train_periods)
        test_mask = index_periods.isin(test_periods)

        train_idx = pos[train_mask]
        test_idx = pos[test_mask]

        # Embargo the tail of train so the boundary does not leak into test.
        if embargo_bars > 0 and train_idx.size > embargo_bars:
            train_idx = train_idx[:-embargo_bars]

        if train_idx.size and test_idx.size:
            splits.append((train_idx, test_idx))

    return splits


def assert_no_leakage(train_idx: np.ndarray, test_idx: np.ndarray, t1: pd.Series) -> None:
    """Fail loudly if any training label window overlaps the test window.

    Cheap to call, and catches the error class that silently invalidates months
    of work. Call it inside the training loop.
    """
    if train_idx.size == 0 or test_idx.size == 0:
        return

    test_start = t1.index[test_idx[0]]
    test_end = t1.iloc[test_idx].max()

    train_events = t1.index[train_idx]
    train_ends = t1.iloc[train_idx]

    overlap = ((train_ends > test_start) & (train_events < test_end)).sum()
    if overlap:
        raise AssertionError(
            f"LEAKAGE: {overlap} training labels overlap the test window "
            f"[{test_start} .. {test_end}]. Results from this split are invalid."
        )
