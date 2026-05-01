"""
Walk-forward cross-validation with purging and embargo.

Why not regular k-fold?
    Financial data is time-ordered. Random k-fold lets the model train on
    rows *after* the rows it's predicting — it accidentally sees the future.
    Walk-forward strictly respects time order.

Why purging?
    Triple-barrier labels are forward-looking: a signal at candle T has its
    label determined by candles T+1 … T+48.  If those future candles fall
    inside the test window, including the signal in training leaks future
    information.  Purging removes training samples whose label window
    overlaps the test period.

Why embargo?
    Rolling indicators (ATR, EMA, RSI) computed at candle T share data with
    candles T-13 … T.  A training sample just before the test boundary is
    feature-correlated with test samples.  Embargo adds a time buffer
    between the last training sample and the first test sample to break
    this correlation.

Fold structure (expanding train window):

    2020          2021          2022          2023
    │─────────────│─────────────│─────────────│
    Fold 1: [─── TRAIN (12mo) ───][emb][─TEST (3mo)─]
    Fold 2: [─── TRAIN (15mo) ──────][emb][─TEST─]
    Fold 3: [─── TRAIN (18mo) ──────────][emb][─TEST─]
    ...

Usage:
    from tbot.ml.walk_forward import WalkForwardSplit

    splitter = WalkForwardSplit()
    for fold in splitter.split(df):
        X_train = X.iloc[fold.train_idx]
        X_test  = X.iloc[fold.test_idx]
        ...
        print(fold.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, List

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One walk-forward fold — train/test index arrays plus diagnostics."""
    fold_num:    int
    train_idx:   np.ndarray
    test_idx:    np.ndarray
    train_start: pd.Timestamp
    train_end:   pd.Timestamp   # last kept training sample (after purge+embargo)
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    n_train:     int
    n_test:      int
    n_purged:    int             # samples dropped because label window overlaps test
    n_embargoed: int             # samples dropped by embargo buffer

    def summary(self) -> str:
        return (
            f"Fold {self.fold_num:2d} | "
            f"train {self.train_start.date()} → {self.train_end.date()} "
            f"({self.n_train:,}) | "
            f"test {self.test_start.date()} → {self.test_end.date()} "
            f"({self.n_test:,}) | "
            f"purged={self.n_purged} embargoed={self.n_embargoed}"
        )


class WalkForwardSplit:
    """
    Expanding-window walk-forward splitter with purge and embargo.

    Args:
        n_train_months:  minimum training window for the first fold
        n_test_months:   test window size per fold
        embargo_weeks:   time buffer between last training and first test sample
                         (prevents feature leakage via rolling indicators)
        timeout_candles: labeling horizon — signals within this duration before
                         test_start are purged from training
        candle_minutes:  candle size in minutes (5 for M5)
    """

    def __init__(
        self,
        n_train_months:  int = 12,
        n_test_months:   int = 3,
        embargo_weeks:   int = 1,
        timeout_candles: int = 48,
        candle_minutes:  int = 5,
    ):
        self.n_train_months  = n_train_months
        self.n_test_months   = n_test_months
        self.embargo_td      = timedelta(weeks=embargo_weeks)
        self.purge_td        = timedelta(minutes=timeout_candles * candle_minutes)

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        """
        Yield Fold objects for each walk-forward window.

        Args:
            df: DataFrame with a 'timestamp' column, sorted ascending.
        """
        timestamps = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
        start      = timestamps.iloc[0]
        end        = timestamps.iloc[-1]

        test_start = start + pd.DateOffset(months=self.n_train_months)
        fold_num   = 0

        while test_start < end:
            test_end = min(
                test_start + pd.DateOffset(months=self.n_test_months),
                end + timedelta(seconds=1),   # include the last row
            )

            # embargo_td (1 week) > purge_td (4h) — embargo is the binding constraint
            # Use the more conservative (earlier) cutoff so both are satisfied
            cutoff = test_start - max(self.embargo_td, self.purge_td)

            # Separate counts for diagnostics
            purge_cutoff = test_start - self.purge_td
            embargo_only_cutoff = test_start - self.embargo_td

            train_mask   = timestamps < cutoff
            test_mask    = (timestamps >= test_start) & (timestamps < test_end)

            train_idx = np.where(train_mask)[0]
            test_idx  = np.where(test_mask)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                test_start = test_end
                continue

            # Count purged (label window overlaps test) vs embargoed (feature overlap)
            n_purged    = int(((timestamps >= purge_cutoff) &
                               (timestamps < test_start)).sum())
            n_embargoed = int(((timestamps >= embargo_only_cutoff) &
                               (timestamps < purge_cutoff)).sum())

            fold_num += 1
            yield Fold(
                fold_num    = fold_num,
                train_idx   = train_idx,
                test_idx    = test_idx,
                train_start = timestamps.iloc[train_idx[0]],
                train_end   = timestamps.iloc[train_idx[-1]],
                test_start  = pd.Timestamp(test_start),
                test_end    = pd.Timestamp(test_end),
                n_train     = len(train_idx),
                n_test      = len(test_idx),
                n_purged    = n_purged,
                n_embargoed = n_embargoed,
            )

            test_start = test_end

    def get_all_folds(self, df: pd.DataFrame) -> List[Fold]:
        return list(self.split(df))
