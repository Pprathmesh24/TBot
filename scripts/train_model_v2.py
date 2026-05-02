"""
Phase 9 Chunk 6 — Train XGBoost v2 (with macro features).

Reads the rebuilt signals_labeled.parquet (now 44 features: 34 original + 10 macro)
and trains a new walk-forward model.  Compares mean AUC against xgb_v1.

Usage:
    .venv/bin/python scripts/train_model_v2.py
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

PARQUET_PATH  = Path("data/features/signals_labeled.parquet")
MODEL_PATH    = Path("models/xgb_v2.pkl")
OOF_PATH      = Path("models/oof_predictions_v2.npz")
RESULTS_PATH  = Path("models/walk_forward_results_v2.json")
V1_RESULTS    = Path("models/walk_forward_results.json")


def main() -> None:
    print(f"Loading {PARQUET_PATH} …")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} rows  ·  {df['label'].value_counts().to_dict()}")
    print(f"  Features: {len(df.columns) - 2} (excl. label + timestamp)")

    splitter = WalkForwardSplit(
        n_train_months  = 12,
        n_test_months   = 3,
        embargo_weeks   = 1,
        timeout_candles = 48,
    )
    result = train_walk_forward(df, splitter)

    print()
    print(result.summary())

    # Compare against v1
    if V1_RESULTS.exists():
        v1 = json.loads(V1_RESULTS.read_text())
        v1_auc = v1["mean_auc"]
        delta  = result.mean_auc - v1_auc
        sign   = "+" if delta >= 0 else ""
        print(f"\n  xgb_v1 mean AUC: {v1_auc:.4f}")
        print(f"  xgb_v2 mean AUC: {result.mean_auc:.4f}  ({sign}{delta:.4f})")
        if delta >= 0.02:
            print("  ✓ Macro features improved AUC by ≥ 0.02 — target met")
        elif delta >= 0:
            print("  ~ Macro features improved AUC but by < 0.02 — marginal gain")
        else:
            print("  ✗ Macro features did NOT improve AUC — investigate feature importances")

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OOF_PATH, proba=result.oof_proba, true=result.oof_true, idx=result.oof_idx)
    print(f"\nOOF predictions saved → {OOF_PATH}")

    median_iter = int(np.median(result.best_iterations)) if result.best_iterations else 300
    print(f"Training final model on all data  (n_estimators={median_iter}) …")

    from tbot.ml.train import _feature_cols
    feature_cols = _feature_cols(df)
    final_model  = train_final_model(df, n_estimators=median_iter)
    save_model(final_model, MODEL_PATH, feature_cols=feature_cols)

    results_dict = {
        "n_folds":         result.n_folds,
        "mean_auc":        round(result.mean_auc, 4),
        "fold_aucs":       [round(a, 4) for a in result.fold_aucs],
        "best_iterations": result.best_iterations,
        "median_iter":     median_iter,
    }
    RESULTS_PATH.write_text(json.dumps(results_dict, indent=2))
    print(f"Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
