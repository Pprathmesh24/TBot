"""
Phase 5 — XGBoost walk-forward training.

Trains XGBoost with purged+embargoed walk-forward CV, collects out-of-fold
predictions for calibration (Chunk 3), and saves the final model.

Usage:
    uv run python scripts/train_model.py

Outputs:
    models/xgb_v1.pkl           — final model trained on all data
    models/oof_predictions.npz  — out-of-fold probabilities + true labels
                                   (fed into calibration in Chunk 3)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from tbot.ml.walk_forward import WalkForwardSplit

# Columns that are not features
_NON_FEATURE_COLS = {"label", "timestamp"}


@dataclass
class WalkForwardResult:
    fold_aucs:  List[float]         = field(default_factory=list)
    oof_proba:  np.ndarray | None   = None   # predicted probability of WIN, per row
    oof_true:   np.ndarray | None   = None   # actual binary label, per row
    oof_idx:    np.ndarray | None   = None   # original df row index, per row
    n_folds:    int                 = 0
    mean_auc:   float               = 0.0
    best_iterations: List[int]      = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Walk-forward AUC ({self.n_folds} folds):"]
        for i, auc in enumerate(self.fold_aucs, 1):
            lines.append(f"  Fold {i:2d}  AUC={auc:.4f}  best_iter={self.best_iterations[i-1]}")
        lines.append(f"  Mean AUC = {self.mean_auc:.4f}  (target > 0.55)")
        return "\n".join(lines)


def make_model(n_pos: int, n_neg: int) -> XGBClassifier:
    """
    Create an XGBClassifier with sensible defaults for financial tabular data.

    scale_pos_weight: compensates for class imbalance.
        Value = n_neg / n_pos ≈ 1.84 for our 35/65 split.
        Without this, XGBoost ignores rare WIN examples.
    """
    return XGBClassifier(
        n_estimators          = 1000,       # upper bound — early stopping decides actual count
        max_depth             = 5,          # shallow trees → less overfitting on noisy M5 data
        learning_rate         = 0.05,       # slow learning + early stopping = better generalisation
        subsample             = 0.8,        # row sampling per tree
        colsample_bytree      = 0.8,        # feature sampling per tree
        min_child_weight      = 10,         # minimum samples per leaf — regularises small nodes
        scale_pos_weight      = n_neg / n_pos if n_pos > 0 else 1.0,
        eval_metric           = "logloss",  # logloss has smoother gradients than AUC for training
        early_stopping_rounds = 50,         # stop if no improvement for 50 rounds
        tree_method           = "hist",     # fast histogram method — good on Apple Silicon
        random_state          = 42,
        verbosity             = 0,          # suppress XGBoost internal logs
    )


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _binary_labels(df: pd.DataFrame) -> np.ndarray:
    """WIN=1, LOSS or NEUTRAL=0."""
    return (df["label"] == "WIN").astype(int).values


def train_walk_forward(
    df: pd.DataFrame,
    splitter: WalkForwardSplit | None = None,
) -> WalkForwardResult:
    """
    Run walk-forward training and collect out-of-fold predictions.

    Args:
        df:       labeled feature DataFrame (signals_labeled.parquet)
        splitter: WalkForwardSplit instance; defaults to standard 12/3/1 config

    Returns:
        WalkForwardResult with per-fold AUC and OOF predictions
    """
    if splitter is None:
        splitter = WalkForwardSplit()

    feature_cols = _feature_cols(df)
    X = df[feature_cols].values.astype(np.float32)
    y = _binary_labels(df)

    result    = WalkForwardResult()
    folds     = splitter.get_all_folds(df)
    n_folds   = len(folds)

    # Pre-allocate OOF arrays
    oof_proba = np.full(len(df), np.nan)
    oof_true  = y.copy()

    print(f"Starting walk-forward training — {n_folds} folds")

    for fold in folds:
        train_idx = fold.train_idx
        test_idx  = fold.test_idx

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # Use last 15% of training rows as XGBoost's early-stopping eval set
        # (still inside training window — no leakage)
        val_split  = int(len(train_idx) * 0.85)
        X_tr, y_tr = X_train[:val_split], y_train[:val_split]
        X_val, y_val = X_train[val_split:], y_train[val_split:]

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos

        model = make_model(n_pos, n_neg)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        proba  = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, proba)
        best_i = model.best_iteration

        oof_proba[test_idx] = proba
        result.fold_aucs.append(auc)
        result.best_iterations.append(best_i)

        print(f"  {fold.summary()}  →  AUC={auc:.4f}  best_iter={best_i}")

    result.n_folds  = n_folds
    result.mean_auc = float(np.mean(result.fold_aucs)) if result.fold_aucs else 0.0

    # Only keep rows that were in some test fold
    covered = ~np.isnan(oof_proba)
    result.oof_proba = oof_proba[covered]
    result.oof_true  = oof_true[covered]
    result.oof_idx   = np.where(covered)[0]

    return result


def train_final_model(df: pd.DataFrame, n_estimators: int = 300) -> XGBClassifier:
    """
    Train on the full dataset — this is the model saved to disk and used
    in production (live trading + backtest).

    We use the median best_iteration from walk-forward as n_estimators so
    early stopping is not needed (and we can train on 100% of data).
    """
    feature_cols = _feature_cols(df)
    X = df[feature_cols].values.astype(np.float32)
    y = _binary_labels(df)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    model = XGBClassifier(
        n_estimators     = n_estimators,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 10,
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0,
        eval_metric      = "logloss",
        tree_method      = "hist",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X, y, verbose=False)
    return model


def save_model(model: XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved → {path}")


def load_model(path: Path) -> XGBClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)
