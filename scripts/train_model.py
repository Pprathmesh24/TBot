"""
Phase 5 — Train XGBoost confidence model.

Usage:
    uv run python scripts/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tbot.ml.train import (
    save_model,
    train_final_model,
    train_walk_forward,
)
from tbot.ml.walk_forward import WalkForwardSplit

PARQUET_PATH = Path("data/features/signals_labeled.parquet")
MODEL_PATH   = Path("models/xgb_v1.pkl")
OOF_PATH     = Path("models/oof_predictions.npz")
RESULTS_PATH = Path("models/walk_forward_results.json")


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load dataset
    # ------------------------------------------------------------------ #
    print(f"Loading {PARQUET_PATH} …")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} rows  ·  {df['label'].value_counts().to_dict()}")

    # ------------------------------------------------------------------ #
    # 2. Walk-forward training
    # ------------------------------------------------------------------ #
    splitter = WalkForwardSplit(
        n_train_months  = 12,
        n_test_months   = 3,
        embargo_weeks   = 1,
        timeout_candles = 48,
    )
    result = train_walk_forward(df, splitter)

    print()
    print(result.summary())

    if result.mean_auc < 0.50:
        print("\n  WARNING: mean AUC < 0.50 — model is worse than random. Investigate features.")
    elif result.mean_auc < 0.55:
        print("\n  NOTE: mean AUC 0.50–0.55 — weak signal. Still better than hardcoded 0.8/0.9.")
    else:
        print("\n  GOOD: mean AUC > 0.55 — meaningful edge detected.")

    # ------------------------------------------------------------------ #
    # 3. Save OOF predictions (used by calibration in Chunk 3)
    # ------------------------------------------------------------------ #
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OOF_PATH,
        proba = result.oof_proba,
        true  = result.oof_true,
        idx   = result.oof_idx,
    )
    print(f"\nOOF predictions saved → {OOF_PATH}  ({len(result.oof_proba):,} rows)")

    # ------------------------------------------------------------------ #
    # 4. Train final model (all data, median best_iter from walk-forward)
    # ------------------------------------------------------------------ #
    median_iter = int(np.median(result.best_iterations)) if result.best_iterations else 300
    print(f"\nTraining final model on all data  (n_estimators={median_iter}) …")

    from tbot.ml.train import _feature_cols
    feature_cols = _feature_cols(df)
    final_model = train_final_model(df, n_estimators=median_iter)
    save_model(final_model, MODEL_PATH, feature_cols=feature_cols)

    # ------------------------------------------------------------------ #
    # 5. Save results JSON (for dashboard / reporting)
    # ------------------------------------------------------------------ #
    results_dict = {
        "n_folds":        result.n_folds,
        "mean_auc":       round(result.mean_auc, 4),
        "fold_aucs":      [round(a, 4) for a in result.fold_aucs],
        "best_iterations": result.best_iterations,
        "median_iter":    median_iter,
    }
    RESULTS_PATH.write_text(json.dumps(results_dict, indent=2))
    print(f"Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
