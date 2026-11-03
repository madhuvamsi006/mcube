"""Purged CV tests — the guard against the failure mode that invalidates everything."""

import numpy as np
import pandas as pd
import pytest

from mcube.validation.purged_cv import PurgedKFold, assert_no_leakage, walk_forward_splits


@pytest.fixture
def overlapping_labels():
    """15-bar label windows on 1-min bars — exactly the overlap triple-barrier creates."""
    idx = pd.date_range("2024-01-01 09:15", periods=500, freq="1min", tz="Asia/Kolkata")
    return pd.Series(idx + pd.Timedelta(minutes=15), index=idx)


def test_requires_t1():
    """Without label end times purging is impossible — must refuse, not silently skip."""
    with pytest.raises(ValueError, match="t1"):
        PurgedKFold(n_splits=3)


def test_no_overlap_between_train_and_test(overlapping_labels):
    t1 = overlapping_labels
    X = pd.DataFrame({"f": np.arange(len(t1))}, index=t1.index)
    cv = PurgedKFold(n_splits=4, t1=t1, embargo_bars=30)

    for train_idx, test_idx in cv.split(X):
        # Raises if any training label window overlaps the test window.
        assert_no_leakage(train_idx, test_idx, t1)


def test_purging_actually_removes_samples(overlapping_labels):
    """A purged split must train on strictly fewer samples than a naive one."""
    t1 = overlapping_labels
    X = pd.DataFrame({"f": np.arange(len(t1))}, index=t1.index)
    cv = PurgedKFold(n_splits=4, t1=t1, embargo_bars=30)

    for train_idx, test_idx in cv.split(X):
        assert len(train_idx) + len(test_idx) < len(X), "nothing was purged"


def test_leakage_detector_catches_deliberate_leak(overlapping_labels):
    """The assert must actually fire when train and test genuinely overlap."""
    t1 = overlapping_labels
    train_idx = np.arange(0, 200)
    test_idx = np.arange(195, 300)  # deliberate overlap
    with pytest.raises(AssertionError, match="LEAKAGE"):
        assert_no_leakage(train_idx, test_idx, t1)


def test_walk_forward_never_trains_on_future():
    """Walk-forward is the honest backtest: train indices must precede test indices."""
    idx = pd.date_range("2023-01-01", periods=400, freq="D", tz="Asia/Kolkata")
    splits = walk_forward_splits(idx, train_months=3, test_months=1, embargo_bars=5)

    assert splits, "expected at least one split"
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "training data must precede test data"
