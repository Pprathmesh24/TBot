"""
AITradingAgentV2 — SMC signal generator with optional ML confidence scoring.

Phase 3 behaviour (no model):
    Hardcoded confidence: FVG=0.65, OB=0.70, Sweep=0.75.

Phase 5+ behaviour (model present):
    XGBoost + isotonic calibrator replace hardcoded scores.
    Each signal's 32 features are built at the signal candle, scored by
    the model, and the calibrated probability becomes the confidence.
    Signals below min_confidence are filtered out.

Interface is identical to AITradingAgentV1 so the backtest engine works
without changes:
    agent = AITradingAgentV2()
    agent.run_on_df(df)
    signals = agent.current_signals
"""

from __future__ import annotations

from typing import List

import pandas as pd

from tbot.core.smc.structure_v2 import EnrichedMarketStructureAnalyzer
from tbot.ml.predict import model_available, score_signal
from tbot.features.builder import build_features_fast, precompute_indicators


class AITradingAgentV2:
    """
    SMC-enriched agent with ML confidence scoring.

    Args:
        min_confidence: signals below this calibrated probability are dropped.
                        Default 0.60 → historically ~68.7% win rate on M5 XAU.
        use_ml:         if True and model files exist, use XGBoost scoring.
                        if False (or model missing), fall back to hardcoded scores.
    """

    def __init__(self, min_confidence: float = 0.60, use_ml: bool = True):
        self.min_confidence  = min_confidence
        self.use_ml          = use_ml
        self.current_signals: List[dict] = []
        self.analyzer        = _CompatStub()   # kept for engine.py compatibility

    def run_on_df(self, df: pd.DataFrame) -> None:
        """
        Run the enriched analyzer on a full candle DataFrame.
        If the ML model is available, replaces hardcoded confidence with
        calibrated XGBoost scores before filtering.
        """
        self.current_signals = []

        # --- 1. SMC detection (unchanged from Phase 3) ---
        analyzer = EnrichedMarketStructureAnalyzer(df)
        raw      = analyzer.get_entry_signals()

        # --- 2. ML scoring (Phase 5+) ---
        if self.use_ml and model_available():
            raw = self._score_with_ml(df, raw, analyzer)

        # --- 3. Confidence filter ---
        self.current_signals = [
            s for s in raw if s.get("confidence", 0) >= self.min_confidence
        ]

    # ------------------------------------------------------------------

    def _score_with_ml(
        self,
        df:       pd.DataFrame,
        signals:  List[dict],
        analyzer: EnrichedMarketStructureAnalyzer,
    ) -> List[dict]:
        """
        Build features for each signal and replace confidence with
        the calibrated ML probability.  Signals that can't be scored
        (e.g. idx < 50) keep their hardcoded confidence.
        """
        fvgs   = analyzer._fvgs
        obs    = analyzer._obs
        sweeps = analyzer._sweeps

        # Pre-compute indicators once for the full df (fast — O(n) single pass)
        pc = precompute_indicators(df)

        # Timestamp → integer index lookup
        ts_to_idx = {ts: i for i, ts in enumerate(df["timestamp"])}

        scored: List[dict] = []
        for sig in signals:
            raw_ts = sig.get("timestamp")
            if raw_ts is None:
                scored.append(sig)
                continue

            ts  = pd.Timestamp(raw_ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            idx = ts_to_idx.get(ts)

            if idx is None:
                scored.append(sig)
                continue

            features = build_features_fast(df, pc, idx, sig, fvgs, obs, sweeps)
            if not features:
                # idx < 50 — not enough history; keep hardcoded score
                scored.append(sig)
                continue

            ml_score = score_signal(features)

            sig_copy              = dict(sig)
            sig_copy["confidence"] = ml_score
            scored.append(sig_copy)

        return scored


class _CompatStub:
    """Placeholder so engine.py can still set agent.analyzer.candles."""
    def __init__(self):
        self.candles = []
