"""
Phase 9 Chunk 6 — Calibrate xgb_v2 probability scores.

Usage:
    .venv/bin/python scripts/calibrate_model_v2.py

Outputs:
    models/calibrator_v2.pkl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tbot.ml.calibration import (
    calibration_report,
    calibrate_array,
    fit_calibrator,
    save_calibrator,
)

OOF_PATH = Path("models/oof_predictions_v2.npz")
CAL_PATH = Path("models/calibrator_v2.pkl")


def main() -> None:
    print(f"Loading OOF predictions from {OOF_PATH} …")
    data      = np.load(OOF_PATH)
    oof_proba = data["proba"]
    oof_true  = data["true"]
    print(f"  {len(oof_proba):,} out-of-fold predictions")
    print(f"  Raw score range: [{oof_proba.min():.3f}, {oof_proba.max():.3f}]")
    print(f"  Actual win rate: {oof_true.mean():.3f}")

    print("\nFitting isotonic calibrator …")
    calibrator = fit_calibrator(oof_proba, oof_true)

    print("\nCalibration table (predicted score vs actual win rate):")
    print(calibration_report(oof_proba, oof_true, calibrator))

    print("\nThreshold analysis (calibrated score → expected win rate):")
    print(f"  {'Threshold':>12}  {'Signals kept':>14}  {'Expected win%':>14}")
    cal_proba = calibrate_array(oof_proba, calibrator)

    for threshold in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask     = cal_proba >= threshold
        n_kept   = mask.sum()
        win_rate = oof_true[mask].mean() if n_kept > 0 else 0.0
        pct_kept = 100 * n_kept / len(cal_proba)
        print(f"  {threshold:>12.2f}  {n_kept:>10,} ({pct_kept:.0f}%)  {win_rate:>13.1%}")

    save_calibrator(calibrator, CAL_PATH)
    print(f"\nCalibrator saved → {CAL_PATH}")


if __name__ == "__main__":
    main()
