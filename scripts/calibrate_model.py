"""
Phase 5 Chunk 3 — Fit and save the probability calibrator.

Reads the OOF predictions saved by train_model.py and fits isotonic
regression to map raw XGBoost scores → true win probabilities.

Usage:
    .venv/bin/python scripts/calibrate_model.py

Outputs:
    models/calibrator_v1.pkl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tbot.ml.calibration import (
    calibration_report,
    fit_calibrator,
    save_calibrator,
)

OOF_PATH  = Path("models/oof_predictions.npz")
CAL_PATH  = Path("models/calibrator_v1.pkl")


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load OOF predictions
    # ------------------------------------------------------------------ #
    print(f"Loading OOF predictions from {OOF_PATH} …")
    data      = np.load(OOF_PATH)
    oof_proba = data["proba"]
    oof_true  = data["true"]
    print(f"  {len(oof_proba):,} out-of-fold predictions")
    print(f"  Raw score range: [{oof_proba.min():.3f}, {oof_proba.max():.3f}]")
    print(f"  Actual win rate: {oof_true.mean():.3f}")

    # ------------------------------------------------------------------ #
    # 2. Fit calibrator
    # ------------------------------------------------------------------ #
    print("\nFitting isotonic calibrator …")
    calibrator = fit_calibrator(oof_proba, oof_true)

    # ------------------------------------------------------------------ #
    # 3. Calibration report
    # ------------------------------------------------------------------ #
    print("\nCalibration table (predicted score vs actual win rate):")
    print(calibration_report(oof_proba, oof_true, calibrator))

    # ------------------------------------------------------------------ #
    # 4. Threshold analysis — what score threshold to use as min_confidence
    # ------------------------------------------------------------------ #
    print("\nThreshold analysis (calibrated score → expected win rate):")
    print(f"  {'Threshold':>12}  {'Signals kept':>14}  {'Expected win%':>14}")
    from tbot.ml.calibration import calibrate_array
    cal_proba = calibrate_array(oof_proba, calibrator)

    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask      = cal_proba >= threshold
        n_kept    = mask.sum()
        win_rate  = oof_true[mask].mean() if n_kept > 0 else 0.0
        pct_kept  = 100 * n_kept / len(cal_proba)
        print(f"  {threshold:>12.2f}  {n_kept:>10,} ({pct_kept:.0f}%)  {win_rate:>13.1%}")

    # ------------------------------------------------------------------ #
    # 5. Save
    # ------------------------------------------------------------------ #
    save_calibrator(calibrator, CAL_PATH)
    print(f"\nCalibrator saved → {CAL_PATH}")


if __name__ == "__main__":
    main()
