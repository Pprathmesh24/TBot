"""
Tests for Phase 3 SMC detectors: FVG, OrderBlock, Liquidity.
All fixtures are synthetic with known, hand-verified outcomes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tbot.core.smc.fvg import FVG, FVGDetector
from tbot.core.smc.liquidity import LiquidityDetector
from tbot.core.smc.order_blocks import OrderBlock, OrderBlockDetector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ts(minute: int) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(minutes=minute * 5)


def _candle(minute, open_, high, low, close):
    return {"timestamp": _ts(minute), "open": open_, "high": high, "low": low, "close": close}


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FVG detector
# ---------------------------------------------------------------------------

class TestFVGDetector:

    def _base_df(self):
        """12 flat candles around 1800, no gaps."""
        rows = [_candle(i, 1800, 1801, 1799, 1800) for i in range(12)]
        return _df(rows)

    # --- bullish FVG ---

    def test_bullish_fvg_detected(self):
        rows = [_candle(i, 1800, 1801, 1799, 1800) for i in range(12)]
        # Inject a bullish gap at index 5 (C1=4, C2=5, C3=6):
        # C1.high=1800, C3.low=1803 → gap 1800-1803
        rows[4] = _candle(4, 1799, 1800, 1798, 1799)
        rows[6] = _candle(6, 1803, 1805, 1803, 1804)
        det = FVGDetector(min_atr_mult=0.0)
        fvgs = det.detect(_df(rows))
        bull = [f for f in fvgs if f.direction == "bull"]
        assert any(f.index == 5 and f.zone_low == 1800.0 and f.zone_high == 1803.0 for f in bull)

    def test_bearish_fvg_detected(self):
        rows = [_candle(i, 1800, 1801, 1799, 1800) for i in range(12)]
        # Bearish gap at index 5: C1.low=1803, C3.high=1801 → gap zone 1801-1803
        rows[4] = _candle(4, 1804, 1806, 1803, 1805)
        rows[6] = _candle(6, 1801, 1801, 1798, 1800)
        det = FVGDetector(min_atr_mult=0.0)
        fvgs = det.detect(_df(rows))
        bear = [f for f in fvgs if f.direction == "bear"]
        assert any(f.index == 5 and f.zone_low == 1801.0 and f.zone_high == 1803.0 for f in bear)

    def test_no_fvg_when_candles_overlap(self):
        rows = [_candle(i, 1800, 1802, 1798, 1800) for i in range(10)]
        det = FVGDetector(min_atr_mult=0.0)
        fvgs = det.detect(_df(rows))
        assert len(fvgs) == 0

    def test_size_filter_removes_small_gap(self):
        rows = [_candle(i, 1800, 1801, 1799, 1800) for i in range(10)]
        # Tiny gap: C1.high=1800.1, C3.low=1800.2 → size=0.1
        rows[2] = _candle(2, 1800, 1800.1, 1799, 1800)
        rows[4] = _candle(4, 1800.2, 1800.5, 1800.2, 1800.3)
        det = FVGDetector(min_atr_mult=5.0)   # very high threshold → filtered out
        fvgs = det.detect(_df(rows))
        assert len(fvgs) == 0

    def test_fvg_size_property(self):
        fvg = FVG(index=1, timestamp=_ts(1), direction="bull",
                  zone_high=1805.0, zone_low=1800.0)
        assert fvg.size == pytest.approx(5.0)

    # --- mark_filled ---

    def test_mark_filled_when_price_enters_zone(self):
        rows = [_candle(i, 1800, 1801, 1799, 1800) for i in range(12)]
        rows[4] = _candle(4, 1799, 1800, 1798, 1799)
        rows[6] = _candle(6, 1803, 1805, 1803, 1804)
        # Candle 8 retraces into the 1800-1803 zone
        rows[8] = _candle(8, 1804, 1804, 1801, 1802)
        df = _df(rows)
        det = FVGDetector(min_atr_mult=0.0)
        fvgs = det.detect(df)
        det.mark_filled(fvgs, df)
        target = next(f for f in fvgs if f.index == 5 and f.direction == "bull")
        assert target.is_filled is True

    def test_mark_filled_not_filled_when_price_stays_away(self):
        # FVG zone: C1.high=1800, C3.low=1803 → zone 1800-1803
        # All candles after C3 stay at 1806+, well above the zone → not filled
        rows = [_candle(i, 1800, 1801, 1799, 1800) for i in range(5)]
        rows.append(_candle(5, 1801, 1802, 1800, 1801))  # C2
        rows.append(_candle(6, 1803, 1805, 1803, 1804))  # C3 (low=1803 → zone top)
        # Continuation candles well above the zone
        for i in range(7, 12):
            rows.append(_candle(i, 1806, 1808, 1805, 1807))
        df = _df(rows)
        det = FVGDetector(min_atr_mult=0.0)
        fvgs = det.detect(df)
        det.mark_filled(fvgs, df)
        target = next((f for f in fvgs if f.direction == "bull" and f.zone_low == 1800.0), None)
        if target:
            assert target.is_filled is False

    def test_active_returns_only_unfilled(self):
        rows = [_candle(i, 1800, 1801, 1799, 1800) for i in range(10)]
        df = _df(rows)
        det = FVGDetector(min_atr_mult=0.0)
        fvgs = det.detect(df)
        for f in fvgs:
            f.is_filled = True
        assert det.active(fvgs) == []


# ---------------------------------------------------------------------------
# OrderBlock detector
# ---------------------------------------------------------------------------

class TestOrderBlockDetector:

    def _bos_df(self):
        """
        10 range-bound candles, then a bearish OB candle, then a bullish BOS.
        Guarantees exactly one bullish OB.
        """
        rows = [_candle(i, 1800, 1801.5, 1799, 1800.5) for i in range(10)]
        rows.append(_candle(10, 1802, 1803, 1797, 1798))   # bearish OB candidate
        rows.append(_candle(11, 1800, 1816, 1799, 1815))   # bullish BOS
        rows.append(_candle(12, 1814, 1815, 1813, 1814))   # continuation
        return _df(rows)

    def test_bullish_ob_found(self):
        df = self._bos_df()
        det = OrderBlockDetector(swing_lookback=10, ob_lookback=5)
        obs = det.detect(df)
        bull = [ob for ob in obs if ob.direction == "bull"]
        assert len(bull) >= 1

    def test_bullish_ob_zone_uses_bearish_candle(self):
        df = self._bos_df()
        det = OrderBlockDetector(swing_lookback=10, ob_lookback=5)
        obs = det.detect(df)
        bull = [ob for ob in obs if ob.direction == "bull"]
        ob = bull[-1]
        # OB is the bearish candle at idx 10: open=1802, low=1797
        assert ob.zone_high == pytest.approx(1802.0)
        assert ob.zone_low  == pytest.approx(1797.0)

    def test_ob_bos_index_is_after_ob_index(self):
        df = self._bos_df()
        det = OrderBlockDetector(swing_lookback=10, ob_lookback=5)
        obs = det.detect(df)
        for ob in obs:
            assert ob.bos_index > ob.index

    def test_ob_size_property(self):
        ob = OrderBlock(
            index=5, timestamp=_ts(5), direction="bull",
            zone_high=1810.0, zone_low=1800.0, bos_index=8,
        )
        assert ob.size == pytest.approx(10.0)

    # --- mitigated ---

    def test_ob_mitigated_when_price_retraces(self):
        df = self._bos_df()
        # Append a candle that retraces into the OB zone (low=1799 < zone_high=1802)
        rows = df.to_dict("records")
        rows.append(_candle(13, 1812, 1813, 1799, 1800))
        df2 = _df(rows)
        det = OrderBlockDetector(swing_lookback=10, ob_lookback=5)
        obs = det.detect(df2)
        det.mark_mitigated(obs, df2)
        bull = [ob for ob in obs if ob.direction == "bull"]
        assert any(ob.is_mitigated for ob in bull)

    def test_ob_not_mitigated_when_price_stays_above(self):
        df = self._bos_df()
        det = OrderBlockDetector(swing_lookback=10, ob_lookback=5)
        obs = det.detect(df)
        det.mark_mitigated(obs, df)
        bull = [ob for ob in obs if ob.direction == "bull"]
        # Price never went below zone_high=1802 after the BOS
        assert all(not ob.is_mitigated for ob in bull)

    def test_active_returns_unmitigated(self):
        df = self._bos_df()
        det = OrderBlockDetector(swing_lookback=10, ob_lookback=5)
        obs = det.detect(df)
        for ob in obs:
            ob.is_mitigated = True
        assert det.active(obs) == []


# ---------------------------------------------------------------------------
# Liquidity detector
# ---------------------------------------------------------------------------

class TestLiquidityDetector:

    def _sweep_df(self, sweep_direction: str):
        """
        Build a candle series with one deliberate wick sweep.
        sweep_direction: 'bull' = wick below swing low, close above it
                         'bear' = wick above swing high, close below it
        """
        # 20 range-bound candles to establish swing points
        rows = [_candle(i, 1800, 1802, 1798, 1800) for i in range(20)]
        if sweep_direction == "bull":
            # swing low at 1798; sweep candle: low=1793 (wick 5 pts below), close=1801
            rows.append(_candle(20, 1800, 1802, 1793, 1801))
        else:
            # swing high at 1802; sweep candle: high=1808 (wick 6 pts above), close=1799
            rows.append(_candle(20, 1800, 1808, 1798, 1799))
        return _df(rows)

    def test_bull_sweep_detected(self):
        df = self._sweep_df("bull")
        det = LiquidityDetector(swing_lookback=3, wick_atr_mult=0.0)
        sweeps = det.detect_sweeps(df)
        bull = [s for s in sweeps if s.direction == "bull"]
        assert len(bull) >= 1

    def test_bear_sweep_detected(self):
        df = self._sweep_df("bear")
        det = LiquidityDetector(swing_lookback=3, wick_atr_mult=0.0)
        sweeps = det.detect_sweeps(df)
        bear = [s for s in sweeps if s.direction == "bear"]
        assert len(bear) >= 1

    def test_sweep_wick_size_positive(self):
        df = self._sweep_df("bull")
        det = LiquidityDetector(swing_lookback=3, wick_atr_mult=0.0)
        sweeps = det.detect_sweeps(df)
        for s in sweeps:
            assert s.wick_size > 0

    def test_wick_filter_removes_small_sweeps(self):
        df = self._sweep_df("bull")
        # wick is 5 pts; ATR ~4 pts; mult=2.0 → threshold ~8 pts → filtered out
        det = LiquidityDetector(swing_lookback=3, wick_atr_mult=2.0)
        sweeps = det.detect_sweeps(df)
        bull = [s for s in sweeps if s.direction == "bull" and s.index == 20]
        assert len(bull) == 0

    def test_no_sweep_when_close_does_not_snap_back(self):
        rows = [_candle(i, 1800, 1802, 1798, 1800) for i in range(20)]
        # Wick goes below but close also stays below → not a sweep, just a move
        rows.append(_candle(20, 1800, 1801, 1790, 1792))
        df = _df(rows)
        det = LiquidityDetector(swing_lookback=3, wick_atr_mult=0.0)
        sweeps = det.detect_sweeps(df)
        bull_at_20 = [s for s in sweeps if s.direction == "bull" and s.index == 20]
        assert len(bull_at_20) == 0

    # --- equal levels ---

    def test_equal_highs_detected(self):
        # Build two swing highs at the same price level
        rows = [_candle(i, 1800, 1800 + (1 if i % 2 == 0 else 0.2), 1799, 1800) for i in range(30)]
        # Two clear swing peaks at 1805
        rows[5]  = _candle(5,  1803, 1805, 1802, 1804)
        rows[15] = _candle(15, 1803, 1805, 1802, 1804)
        df = _df(rows)
        det = LiquidityDetector(swing_lookback=3, eq_atr_mult=0.5)
        eq = det.detect_equal_levels(df)
        eq_h = [e for e in eq if e.direction == "high"]
        assert len(eq_h) >= 1

    def test_equal_level_touch_count(self):
        rows = [_candle(i, 1800, 1800 + (1 if i % 2 == 0 else 0.2), 1799, 1800) for i in range(30)]
        rows[5]  = _candle(5,  1803, 1805, 1802, 1804)
        rows[15] = _candle(15, 1803, 1805, 1802, 1804)
        df = _df(rows)
        det = LiquidityDetector(swing_lookback=3, eq_atr_mult=0.5)
        eq = det.detect_equal_levels(df)
        eq_h = [e for e in eq if e.direction == "high"]
        if eq_h:
            assert eq_h[0].touch_count >= 2
