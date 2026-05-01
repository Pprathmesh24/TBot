"""
Tests for walk-forward splitter — purge, embargo, and no-leakage guarantees.

These tests are the most important in Phase 5.  If any fail, the model
AUC numbers are meaningless.
"""

import numpy as np
import pandas as pd
import pytest

from tbot.ml.walk_forward import WalkForwardSplit


def _make_df(n_months: int = 18, freq: str = "5min") -> pd.DataFrame:
    """Synthetic M5 signal DataFrame with one row per 5-minute candle."""
    timestamps = pd.date_range("2020-01-01", periods=n_months * 30 * 24 * 12,
                               freq=freq, tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "label": "WIN"})


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_folds_are_produced():
    df = _make_df(24)
    splitter = WalkForwardSplit(n_train_months=12, n_test_months=3)
    folds = splitter.get_all_folds(df)
    assert len(folds) >= 3, "Expected at least 3 folds for 24 months of data"


def test_fold_numbers_are_sequential():
    df = _make_df(24)
    folds = WalkForwardSplit().get_all_folds(df)
    for i, fold in enumerate(folds, start=1):
        assert fold.fold_num == i


def test_train_and_test_are_non_empty():
    df = _make_df(24)
    for fold in WalkForwardSplit().get_all_folds(df):
        assert fold.n_train > 0
        assert fold.n_test  > 0


def test_test_windows_are_non_overlapping():
    df = _make_df(24)
    folds = WalkForwardSplit().get_all_folds(df)
    for a, b in zip(folds, folds[1:]):
        assert a.test_end <= b.test_start, \
            f"Fold {a.fold_num} test overlaps fold {b.fold_num} test"


# ---------------------------------------------------------------------------
# Strict time ordering — the no-leakage guarantee
# ---------------------------------------------------------------------------

def test_no_train_index_appears_in_test():
    """Core leakage check: train and test sets must be disjoint."""
    df = _make_df(24)
    for fold in WalkForwardSplit().get_all_folds(df):
        overlap = np.intersect1d(fold.train_idx, fold.test_idx)
        assert len(overlap) == 0, \
            f"Fold {fold.fold_num}: {len(overlap)} rows appear in both train and test"


def test_all_train_timestamps_before_test_start():
    """Every training timestamp must be strictly before the test window."""
    df = _make_df(24)
    timestamps = pd.to_datetime(df["timestamp"])
    for fold in WalkForwardSplit().get_all_folds(df):
        train_ts = timestamps.iloc[fold.train_idx]
        assert (train_ts < fold.test_start).all(), \
            f"Fold {fold.fold_num}: some train timestamps are >= test_start"


def test_all_test_timestamps_within_test_window():
    df = _make_df(24)
    timestamps = pd.to_datetime(df["timestamp"])
    for fold in WalkForwardSplit().get_all_folds(df):
        test_ts = timestamps.iloc[fold.test_idx]
        assert (test_ts >= fold.test_start).all()
        assert (test_ts <  fold.test_end).all()


# ---------------------------------------------------------------------------
# Embargo gap
# ---------------------------------------------------------------------------

def test_embargo_gap_exists():
    """Last training timestamp must be at least embargo_weeks before test_start."""
    from datetime import timedelta
    df       = _make_df(24)
    ts       = pd.to_datetime(df["timestamp"])
    splitter = WalkForwardSplit(embargo_weeks=1)
    for fold in splitter.get_all_folds(df):
        last_train = ts.iloc[fold.train_idx[-1]]
        gap = fold.test_start - last_train
        assert gap >= timedelta(weeks=1) - timedelta(minutes=5), \
            f"Fold {fold.fold_num}: embargo gap {gap} < 1 week"


# ---------------------------------------------------------------------------
# Expanding window
# ---------------------------------------------------------------------------

def test_train_size_grows_each_fold():
    """Each fold should have more training data than the previous one."""
    df = _make_df(24)
    folds = WalkForwardSplit().get_all_folds(df)
    for a, b in zip(folds, folds[1:]):
        assert b.n_train > a.n_train, \
            f"Fold {b.fold_num} has fewer training rows than fold {a.fold_num}"


# ---------------------------------------------------------------------------
# Smoke test on real dataset (if it exists)
# ---------------------------------------------------------------------------

def test_real_dataset_fold_summary(tmp_path):
    """Smoke test: splitter runs on real labeled dataset without errors."""
    import json
    from pathlib import Path

    stats_path = Path("data/features/dataset_stats.json")
    if not stats_path.exists():
        pytest.skip("Real dataset not built yet — run build_training_dataset.py first")

    df = pd.read_parquet("data/features/signals_labeled.parquet")
    splitter = WalkForwardSplit()
    folds = splitter.get_all_folds(df)

    assert len(folds) >= 4, f"Expected ≥4 folds on 5-year data, got {len(folds)}"

    print(f"\n{len(folds)} folds on real dataset:")
    for fold in folds:
        print(" ", fold.summary())
        # Each fold: train must precede test with embargo gap
        assert fold.n_train > 100
        assert fold.n_test  > 50
