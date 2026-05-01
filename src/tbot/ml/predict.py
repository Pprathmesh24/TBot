"""
ML inference — score a single signal's feature dict → calibrated win probability.

Lazy-loads the XGBoost model + isotonic calibrator on first call and caches
them for the lifetime of the process (one load per run, not per signal).

Graceful fallback: if model files are missing, returns 0.5 (neutral).
This lets the system run without a trained model during early development.

Usage:
    from tbot.ml.predict import score_signal

    prob = score_signal(features_dict)  # → float in [0, 1]
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np

_DEFAULT_MODEL_PATH = Path("models/xgb_v1.pkl")
_DEFAULT_CAL_PATH   = Path("models/calibrator_v1.pkl")


@lru_cache(maxsize=1)
def _load_artifacts(model_path: str, cal_path: str):
    """Load model + calibrator once and cache. lru_cache key is the path strings."""
    import json

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)

    # Feature names: prefer the .features.json sidecar file (always present
    # after train_model.py), fall back to booster names (None if trained on
    # numpy arrays), then empty list as last resort.
    names_path = Path(model_path).with_suffix(".features.json")
    if names_path.exists():
        feature_names: List[str] = json.loads(names_path.read_text())
    else:
        feature_names = model.get_booster().feature_names or []

    return model, calibrator, feature_names


def model_available(
    model_path: Path = _DEFAULT_MODEL_PATH,
    cal_path:   Path = _DEFAULT_CAL_PATH,
) -> bool:
    return model_path.exists() and cal_path.exists()


def score_signal(
    features:    dict,
    model_path:  Path = _DEFAULT_MODEL_PATH,
    cal_path:    Path = _DEFAULT_CAL_PATH,
) -> float:
    """
    Score a feature dict → calibrated win probability.

    Args:
        features:   output of build_features_fast() — flat dict of floats
        model_path: path to xgb_v1.pkl
        cal_path:   path to calibrator_v1.pkl

    Returns:
        Calibrated probability in [0, 1].
        Returns 0.5 if model files are missing or scoring fails.
    """
    if not model_available(model_path, cal_path):
        return 0.5

    try:
        model, calibrator, feature_names = _load_artifacts(
            str(model_path), str(cal_path)
        )
        # Build feature vector in exact training order (unknown features → 0.0)
        x = np.array(
            [[features.get(f, 0.0) for f in feature_names]],
            dtype=np.float32,
        )
        raw   = float(model.predict_proba(x)[0, 1])
        cal   = float(calibrator.predict([raw])[0])
        return float(np.clip(cal, 0.0, 1.0))

    except Exception:
        return 0.5
