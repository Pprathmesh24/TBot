"""
Probability calibration for XGBoost confidence scores.

Raw XGBoost predict_proba scores are not well-calibrated — a score of 0.65
does not automatically mean 65% historical win rate.  Isotonic regression
fits a monotonic step function that maps raw scores to true probabilities.

Why isotonic (not Platt/sigmoid)?
    Platt scaling assumes the miscalibration is a smooth sigmoid shift.
    Financial model outputs often have irregular miscalibration shapes
    (overconfident in the middle, underconfident at extremes).
    Isotonic regression makes no shape assumption — it fits any monotonic
    curve, which is more flexible and empirically better here.

Usage:
    calibrator = fit_calibrator(oof_proba, oof_true)
    calibrated_score = calibrate(raw_score, calibrator)
    save_calibrator(calibrator, Path("models/calibrator_v1.pkl"))
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(oof_proba: np.ndarray, oof_true: np.ndarray) -> IsotonicRegression:
    """
    Fit isotonic regression calibrator on out-of-fold predictions.

    Args:
        oof_proba: raw model probabilities from walk-forward OOF predictions
        oof_true:  actual binary labels (WIN=1, else=0)

    Returns:
        Fitted IsotonicRegression that maps raw score → calibrated probability
    """
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof_proba, oof_true)
    return cal


def calibrate(raw_score: float, calibrator: IsotonicRegression) -> float:
    """Map a single raw score to a calibrated probability."""
    return float(calibrator.predict([raw_score])[0])


def calibrate_array(raw_scores: np.ndarray, calibrator: IsotonicRegression) -> np.ndarray:
    """Map an array of raw scores to calibrated probabilities."""
    return calibrator.predict(raw_scores).astype(np.float32)


def calibration_report(
    oof_proba: np.ndarray,
    oof_true: np.ndarray,
    calibrator: IsotonicRegression,
    n_bins: int = 10,
) -> str:
    """
    Print a calibration table: predicted probability bucket vs actual win rate.
    A well-calibrated model has predicted ≈ actual in every row.
    """
    raw_fraction_pos, raw_mean_pred = calibration_curve(oof_true, oof_proba, n_bins=n_bins)
    cal_proba = calibrate_array(oof_proba, calibrator)
    cal_fraction_pos, cal_mean_pred = calibration_curve(oof_true, cal_proba, n_bins=n_bins)

    lines = [
        f"{'Bin':>6}  {'Raw pred':>10}  {'Raw actual':>10}  {'Raw gap':>9}  "
        f"{'Cal pred':>10}  {'Cal actual':>10}  {'Cal gap':>9}",
        "-" * 80,
    ]
    for rp, ra, cp, ca in zip(raw_mean_pred, raw_fraction_pos, cal_mean_pred, cal_fraction_pos):
        lines.append(
            f"{'':>6}  {rp:>10.3f}  {ra:>10.3f}  {rp-ra:>+9.3f}  "
            f"{cp:>10.3f}  {ca:>10.3f}  {cp-ca:>+9.3f}"
        )

    raw_brier  = float(np.mean((oof_proba - oof_true) ** 2))
    cal_brier  = float(np.mean((cal_proba  - oof_true) ** 2))
    lines += [
        "-" * 80,
        f"Brier score (lower=better):  raw={raw_brier:.4f}  calibrated={cal_brier:.4f}",
        f"Improvement: {100*(raw_brier - cal_brier)/raw_brier:.1f}%",
    ]
    return "\n".join(lines)


def save_calibrator(calibrator: IsotonicRegression, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path: Path) -> IsotonicRegression:
    with open(path, "rb") as f:
        return pickle.load(f)
