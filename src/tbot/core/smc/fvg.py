"""
Fair Value Gap (FVG) detector.

A FVG is a 3-candle imbalance where price moved so fast it left a gap
that the market is likely to revisit (fill).

    Bullish FVG:  C1.high < C3.low   → gap above C1, below C3
    Bearish FVG:  C1.low  > C3.high  → gap below C1, above C3

Only gaps >= min_atr_mult × ATR(14) are kept — filters out noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class FVG:
    index:     int            # index of C2 (the middle candle)
    timestamp: pd.Timestamp
    direction: str            # "bull" | "bear"
    zone_high: float
    zone_low:  float
    is_filled: bool = False   # True once price trades inside the zone

    @property
    def size(self) -> float:
        return self.zone_high - self.zone_low


class FVGDetector:
    """
    Scans a candle DataFrame and returns all Fair Value Gaps.

    Args:
        min_atr_mult: minimum gap size as a multiple of ATR(14).
                      Default 0.3 — rejects micro-gaps that are just spread noise.
        atr_period:   period for ATR calculation.
    """

    def __init__(self, min_atr_mult: float = 0.3, atr_period: int = 14):
        self.min_atr_mult = min_atr_mult
        self.atr_period   = atr_period

    def detect(self, df: pd.DataFrame) -> List[FVG]:
        """
        Scan the full DataFrame and return all FVGs (filled and unfilled).

        Args:
            df: candles with columns [timestamp, open, high, low, close]

        Returns:
            List of FVG objects sorted by index ascending.
        """
        atr = self._atr(df)
        fvgs: List[FVG] = []

        # Need at least 3 candles; C1=i-1, C2=i, C3=i+1
        for i in range(1, len(df) - 1):
            c1_high = df["high"].iloc[i - 1]
            c1_low  = df["low"].iloc[i - 1]
            c3_high = df["high"].iloc[i + 1]
            c3_low  = df["low"].iloc[i + 1]
            min_size = self.min_atr_mult * atr.iloc[i]

            # Bullish FVG
            if c1_high < c3_low:
                size = c3_low - c1_high
                if size >= min_size:
                    fvgs.append(FVG(
                        index=i,
                        timestamp=df["timestamp"].iloc[i],
                        direction="bull",
                        zone_high=c3_low,
                        zone_low=c1_high,
                    ))

            # Bearish FVG
            elif c1_low > c3_high:
                size = c1_low - c3_high
                if size >= min_size:
                    fvgs.append(FVG(
                        index=i,
                        timestamp=df["timestamp"].iloc[i],
                        direction="bear",
                        zone_high=c1_low,
                        zone_low=c3_high,
                    ))

        return fvgs

    def mark_filled(self, fvgs: List[FVG], df: pd.DataFrame) -> List[FVG]:
        """
        For each FVG, check if subsequent candles have traded inside the zone.
        Updates fvg.is_filled in-place and returns the list.
        """
        for fvg in fvgs:
            if fvg.is_filled:
                continue
            # Look at candles after the FVG's C3 (index + 2 onward)
            future = df.iloc[fvg.index + 2 :]
            filled = (
                (future["low"]  <= fvg.zone_high) &
                (future["high"] >= fvg.zone_low)
            ).any()
            fvg.is_filled = bool(filled)
        return fvgs

    def active(self, fvgs: List[FVG]) -> List[FVG]:
        """Return only unfilled FVGs."""
        return [f for f in fvgs if not f.is_filled]

    # ------------------------------------------------------------------

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        """True Range → rolling mean ATR(period)."""
        high  = df["high"]
        low   = df["low"]
        close = df["close"].shift(1)

        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low  - close).abs(),
        ], axis=1).max(axis=1)

        return tr.rolling(self.atr_period, min_periods=1).mean()
