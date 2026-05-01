"""
Liquidity detector: Equal Highs/Lows and Wick Sweeps.

Equal Highs/Lows (EQH/EQL):
    Two or more swing highs/lows within tolerance of each other.
    These cluster points mark resting liquidity (retail stop losses).

Wick Sweep (stop hunt):
    A candle whose wick exceeds a recent swing high/low by >= wick_atr_mult * ATR,
    but whose *close* snaps back inside.

        Bull sweep: wick below swing low, close above it  → reversal UP likely
        Bear sweep: wick above swing high, close below it → reversal DOWN likely
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
import numpy as np


@dataclass
class EqualLevel:
    index:       int             # index of the confirming (second+) touch
    timestamp:   pd.Timestamp
    direction:   str             # "high" | "low"
    level:       float           # price level of the cluster
    touch_count: int             # number of swings at this level


@dataclass
class LiquiditySweep:
    index:        int
    timestamp:    pd.Timestamp
    direction:    str            # "bull" (swept lows → reversal up) | "bear" (swept highs → reversal down)
    swept_level:  float          # the swing high/low that was swept
    wick_size:    float          # size of the wick beyond the swept level


class LiquidityDetector:
    """
    Args:
        swing_lookback:  half-window for swing high/low detection (default 5 = 5 candles each side).
        wick_atr_mult:   minimum wick size beyond the level, as ATR multiple (default 0.5).
        eq_atr_mult:     tolerance for two swings to be "equal", as ATR multiple (default 0.1).
        atr_period:      period for ATR (default 14).
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        wick_atr_mult:  float = 0.5,
        eq_atr_mult:    float = 0.1,
        atr_period:     int = 14,
    ):
        self.swing_lookback = swing_lookback
        self.wick_atr_mult  = wick_atr_mult
        self.eq_atr_mult    = eq_atr_mult
        self.atr_period     = atr_period

    def detect_equal_levels(self, df: pd.DataFrame) -> List[EqualLevel]:
        """Find Equal Highs and Equal Lows (resting liquidity clusters)."""
        atr   = self._atr(df)
        sh_idx = self._swing_highs(df)
        sl_idx = self._swing_lows(df)
        levels: List[EqualLevel] = []

        levels.extend(self._cluster(df, sh_idx, atr, direction="high"))
        levels.extend(self._cluster(df, sl_idx, atr, direction="low"))
        return sorted(levels, key=lambda x: x.index)

    def detect_sweeps(self, df: pd.DataFrame) -> List[LiquiditySweep]:
        """Find wick sweeps of recent swing highs/lows."""
        atr    = self._atr(df)
        sh_idx = self._swing_highs(df)
        sl_idx = self._swing_lows(df)
        sweeps: List[LiquiditySweep] = []

        # Build lookup: for each candle, what is the most recent swing high/low before it?
        sh_levels = self._rolling_recent_level(df["high"], sh_idx, len(df))
        sl_levels = self._rolling_recent_level(df["low"],  sl_idx, len(df))

        for i in range(self.swing_lookback + 1, len(df)):
            min_wick = self.wick_atr_mult * atr.iloc[i]
            candle_high  = df["high"].iloc[i]
            candle_low   = df["low"].iloc[i]
            candle_close = df["close"].iloc[i]

            # Bear sweep: wick above recent swing high, close back below it
            swing_h = sh_levels[i]
            if swing_h and candle_high > swing_h:
                wick = candle_high - swing_h
                if wick >= min_wick and candle_close < swing_h:
                    sweeps.append(LiquiditySweep(
                        index=i,
                        timestamp=df["timestamp"].iloc[i],
                        direction="bear",
                        swept_level=swing_h,
                        wick_size=wick,
                    ))

            # Bull sweep: wick below recent swing low, close back above it
            swing_l = sl_levels[i]
            if swing_l and candle_low < swing_l:
                wick = swing_l - candle_low
                if wick >= min_wick and candle_close > swing_l:
                    sweeps.append(LiquiditySweep(
                        index=i,
                        timestamp=df["timestamp"].iloc[i],
                        direction="bull",
                        swept_level=swing_l,
                        wick_size=wick,
                    ))

        return sweeps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        high  = df["high"]
        low   = df["low"]
        close = df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low  - close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_period, min_periods=1).mean()

    def _swing_highs(self, df: pd.DataFrame) -> List[int]:
        """Indices where high[i] is the local max over ±swing_lookback candles."""
        n = self.swing_lookback
        highs = df["high"].values
        result = []
        for i in range(n, len(highs) - n):
            window = highs[i - n: i + n + 1]
            if highs[i] == window.max():
                result.append(i)
        return result

    def _swing_lows(self, df: pd.DataFrame) -> List[int]:
        n = self.swing_lookback
        lows = df["low"].values
        result = []
        for i in range(n, len(lows) - n):
            window = lows[i - n: i + n + 1]
            if lows[i] == window.min():
                result.append(i)
        return result

    def _cluster(
        self, df: pd.DataFrame, swing_idx: List[int], atr: pd.Series, direction: str
    ) -> List[EqualLevel]:
        """Group swings that are within eq_atr_mult * ATR of each other."""
        col = "high" if direction == "high" else "low"
        levels: List[EqualLevel] = []
        used: set[int] = set()

        for i, idx in enumerate(swing_idx):
            if idx in used:
                continue
            level_price = df[col].iloc[idx]
            tol = self.eq_atr_mult * atr.iloc[idx]
            cluster = [idx]
            for j in range(i + 1, len(swing_idx)):
                other = swing_idx[j]
                if abs(df[col].iloc[other] - level_price) <= tol:
                    cluster.append(other)
                    used.add(other)
            if len(cluster) >= 2:
                levels.append(EqualLevel(
                    index=cluster[-1],
                    timestamp=df["timestamp"].iloc[cluster[-1]],
                    direction=direction,
                    level=level_price,
                    touch_count=len(cluster),
                ))
        return levels

    def _rolling_recent_level(
        self, price_series: pd.Series, swing_idx: List[int], n: int
    ) -> list:
        """For each candle index, store the price at the most recent swing before it."""
        result: list = [None] * n
        ptr = 0
        swing_set = {i: price_series.iloc[i] for i in swing_idx}
        last = None
        for i in range(n):
            if i in swing_set:
                last = swing_set[i]
            result[i] = last
        return result
