"""
Order Block (OB) detector.

An Order Block is the last opposite-color candle before a confirmed BOS.
It marks where institutional orders drove price, creating a zone that
price tends to retrace into before continuing in the BOS direction.

    Bullish OB: last bearish candle before a bullish BOS
    Bearish OB: last bullish candle before a bearish BOS

BOS detection here is intentionally simple (rolling max/min break) so
this detector works standalone without the full MarketStructureAnalyzer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class OrderBlock:
    index:        int             # candle index of the OB
    timestamp:    pd.Timestamp
    direction:    str             # "bull" | "bear"
    zone_high:    float           # top of OB zone  (candle open for bull, candle high for bear)
    zone_low:     float           # bottom of OB zone (candle low for bull, candle open for bear)
    bos_index:    int             # index of the BOS candle that created this OB
    is_mitigated: bool = False    # True once price retraces into the zone

    @property
    def size(self) -> float:
        return self.zone_high - self.zone_low


class OrderBlockDetector:
    """
    Args:
        swing_lookback: rolling window for detecting BOS (break of N-bar high/low).
        ob_lookback:    how many candles before the BOS to search for the OB candle.
    """

    def __init__(self, swing_lookback: int = 20, ob_lookback: int = 10):
        self.swing_lookback = swing_lookback
        self.ob_lookback    = ob_lookback

    def detect(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Scan df and return all Order Blocks (mitigated and active).

        Args:
            df: candles with columns [timestamp, open, high, low, close]
        """
        obs: List[OrderBlock] = []
        seen_bos: set[int] = set()   # one OB per BOS event

        # Rolling max/min of the *previous* swing_lookback candles
        prev_high = df["high"].shift(1).rolling(self.swing_lookback, min_periods=1).max()
        prev_low  = df["low"].shift(1).rolling(self.swing_lookback,  min_periods=1).min()

        for i in range(self.swing_lookback, len(df)):
            close = df["close"].iloc[i]
            is_bull_bos = close > prev_high.iloc[i]
            is_bear_bos = close < prev_low.iloc[i]

            if not (is_bull_bos or is_bear_bos):
                continue
            if i in seen_bos:
                continue
            seen_bos.add(i)

            start = max(0, i - self.ob_lookback)
            window = df.iloc[start:i]

            if is_bull_bos:
                ob = self._last_bearish_candle(window, bos_index=i)
            else:
                ob = self._last_bullish_candle(window, bos_index=i)

            if ob is not None:
                obs.append(ob)

        return obs

    def mark_mitigated(self, obs: List[OrderBlock], df: pd.DataFrame) -> List[OrderBlock]:
        """
        Mark each OB as mitigated if a subsequent candle retraces into the zone.
        Updates in-place and returns the list.
        """
        for ob in obs:
            if ob.is_mitigated:
                continue
            future = df.iloc[ob.bos_index + 1:]
            if ob.direction == "bull":
                # Price retraces down into the bullish OB zone
                touched = (future["low"] <= ob.zone_high).any()
            else:
                # Price retraces up into the bearish OB zone
                touched = (future["high"] >= ob.zone_low).any()
            ob.is_mitigated = bool(touched)
        return obs

    def active(self, obs: List[OrderBlock]) -> List[OrderBlock]:
        """Return only unmitigated Order Blocks."""
        return [ob for ob in obs if not ob.is_mitigated]

    # ------------------------------------------------------------------

    def _last_bearish_candle(self, window: pd.DataFrame, bos_index: int) -> OrderBlock | None:
        """Find the last bearish (red) candle in window → bullish OB."""
        bearish = window[window["close"] < window["open"]]
        if bearish.empty:
            return None
        row = bearish.iloc[-1]
        return OrderBlock(
            index=int(row.name),
            timestamp=row["timestamp"],
            direction="bull",
            zone_high=float(row["open"]),   # body top of bearish candle
            zone_low=float(row["low"]),     # full wick bottom
            bos_index=bos_index,
        )

    def _last_bullish_candle(self, window: pd.DataFrame, bos_index: int) -> OrderBlock | None:
        """Find the last bullish (green) candle in window → bearish OB."""
        bullish = window[window["close"] > window["open"]]
        if bullish.empty:
            return None
        row = bullish.iloc[-1]
        return OrderBlock(
            index=int(row.name),
            timestamp=row["timestamp"],
            direction="bear",
            zone_high=float(row["high"]),   # full wick top
            zone_low=float(row["open"]),    # body bottom of bullish candle
            bos_index=bos_index,
        )
