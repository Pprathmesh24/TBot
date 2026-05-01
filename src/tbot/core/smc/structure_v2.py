"""
EnrichedMarketStructureAnalyzer — combines FVG, OB, and Liquidity Sweep
detection into a unified signal source.

Three entry triggers:
    1. FVG retracement:    price trades back into an active Fair Value Gap
    2. OB retracement:     price trades back into an active Order Block
    3. Liquidity sweep:    a wick sweep happened recently → reversal entry

Stops are ATR-based (dynamic), not fixed percentages.
Each zone emits at most one signal to prevent duplicate entries.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from tbot.core.smc.fvg import FVGDetector
from tbot.core.smc.liquidity import LiquidityDetector
from tbot.core.smc.order_blocks import OrderBlockDetector


# Confidence scores per trigger type (replaced by ML in Phase 5)
_CONF_FVG    = 0.65
_CONF_OB     = 0.70
_CONF_SWEEP  = 0.75


class EnrichedMarketStructureAnalyzer:
    """
    Args:
        df:               candle DataFrame from load_candles()
        min_separation:   minimum candles between consecutive signals (anti-spam)
        sweep_lookback:   how many candles back to look for a recent sweep
        atr_sl_mult:      stop-loss distance = this × ATR below/above zone
        rr_ratio:         take-profit = entry + rr_ratio × risk
    """

    def __init__(
        self,
        df: pd.DataFrame,
        min_separation: int = 3,
        sweep_lookback: int = 10,
        atr_sl_mult:    float = 0.5,
        rr_ratio:       float = 2.0,
    ):
        self.df            = df.reset_index(drop=True)
        self.min_separation = min_separation
        self.sweep_lookback = sweep_lookback
        self.atr_sl_mult    = atr_sl_mult
        self.rr_ratio       = rr_ratio

        self._atr = self._compute_atr()

        # Pre-compute all zones once
        self._fvgs   = FVGDetector(min_atr_mult=0.3).detect(self.df)
        self._obs    = OrderBlockDetector(swing_lookback=20, ob_lookback=10).detect(self.df)
        self._sweeps = LiquidityDetector(swing_lookback=5, wick_atr_mult=0.5).detect_sweeps(self.df)

        # Track which zones have already fired a signal
        self._fvg_signaled: set[int] = set()
        self._ob_signaled:  set[int] = set()

    def get_entry_signals(self) -> List[dict]:
        """Scan every candle and return all SMC entry signals."""
        signals: List[dict] = []
        last_idx = -self.min_separation

        for i in range(20, len(self.df)):
            if i - last_idx < self.min_separation:
                continue

            candle  = self.df.iloc[i]
            atr_val = float(self._atr.iloc[i])
            if atr_val <= 0:
                continue

            sig = (
                self._check_fvg(i, candle, atr_val) or
                self._check_ob(i, candle, atr_val)   or
                self._check_sweep(i, candle, atr_val)
            )

            if sig:
                signals.append(sig)
                last_idx = i

        return signals

    # ------------------------------------------------------------------
    # Trigger 1 — FVG retracement
    # ------------------------------------------------------------------

    def _check_fvg(self, i: int, candle, atr: float) -> dict | None:
        low  = float(candle["low"])
        high = float(candle["high"])

        for fvg in self._fvgs:
            if fvg.index >= i or fvg.index in self._fvg_signaled:
                continue

            if fvg.direction == "bull" and low <= fvg.zone_high and high >= fvg.zone_low:
                sl = fvg.zone_low - self.atr_sl_mult * atr
                tp = float(candle["close"]) + self.rr_ratio * (float(candle["close"]) - sl)
                self._fvg_signaled.add(fvg.index)
                return self._signal("BUY", "FVG Retracement", candle, sl, tp, _CONF_FVG)

            if fvg.direction == "bear" and high >= fvg.zone_low and low <= fvg.zone_high:
                sl = fvg.zone_high + self.atr_sl_mult * atr
                tp = float(candle["close"]) - self.rr_ratio * (sl - float(candle["close"]))
                self._fvg_signaled.add(fvg.index)
                return self._signal("SELL", "FVG Retracement", candle, sl, tp, _CONF_FVG)

        return None

    # ------------------------------------------------------------------
    # Trigger 2 — OB retracement
    # ------------------------------------------------------------------

    def _check_ob(self, i: int, candle, atr: float) -> dict | None:
        low  = float(candle["low"])
        high = float(candle["high"])

        for ob in self._obs:
            if ob.index >= i or ob.index in self._ob_signaled:
                continue

            if ob.direction == "bull" and low <= ob.zone_high and high >= ob.zone_low:
                sl = ob.zone_low - self.atr_sl_mult * atr
                tp = float(candle["close"]) + self.rr_ratio * (float(candle["close"]) - sl)
                self._ob_signaled.add(ob.index)
                return self._signal("BUY", "OB Retracement", candle, sl, tp, _CONF_OB)

            if ob.direction == "bear" and high >= ob.zone_low and low <= ob.zone_high:
                sl = ob.zone_high + self.atr_sl_mult * atr
                tp = float(candle["close"]) - self.rr_ratio * (sl - float(candle["close"]))
                self._ob_signaled.add(ob.index)
                return self._signal("SELL", "OB Retracement", candle, sl, tp, _CONF_OB)

        return None

    # ------------------------------------------------------------------
    # Trigger 3 — Liquidity sweep
    # ------------------------------------------------------------------

    def _check_sweep(self, i: int, candle, atr: float) -> dict | None:
        for sweep in self._sweeps:
            if sweep.index != i:
                continue

            entry = float(candle["close"])
            if sweep.direction == "bull":
                sl = float(candle["low"]) - self.atr_sl_mult * atr
                tp = entry + self.rr_ratio * (entry - sl)
                return self._signal("BUY", "Liquidity Sweep", candle, sl, tp, _CONF_SWEEP)

            if sweep.direction == "bear":
                sl = float(candle["high"]) + self.atr_sl_mult * atr
                tp = entry - self.rr_ratio * (sl - entry)
                return self._signal("SELL", "Liquidity Sweep", candle, sl, tp, _CONF_SWEEP)

        return None

    # ------------------------------------------------------------------

    def _signal(self, side, reason, candle, sl, tp, conf) -> dict:
        return {
            "type":        side,
            "reason":      reason,
            "timestamp":   candle["timestamp"],
            "price":       float(candle["close"]),
            "confidence":  conf,
            "stop_loss":   sl,
            "take_profit": tp,
        }

    def _compute_atr(self, period: int = 14) -> pd.Series:
        df   = self.df
        high = df["high"]
        low  = df["low"]
        prev = df["close"].shift(1)
        tr   = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
