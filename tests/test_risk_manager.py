"""
Tests for RiskManager — circuit breaker, daily loss cap, equity floor,
and ATR-based position sizing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tbot.risk.manager import RiskManager, TradeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(hour: int = 12, day: int = 1) -> datetime:
    return datetime(2024, 1, day, hour, 0, 0, tzinfo=timezone.utc)


def _trade(pnl: float, hour: int = 10, day: int = 1) -> TradeRecord:
    return TradeRecord(net_pnl=pnl, exit_time=_utc(hour=hour, day=day))


def _rm(**kwargs) -> RiskManager:
    return RiskManager(**kwargs)


# ---------------------------------------------------------------------------
# Equity floor
# ---------------------------------------------------------------------------

class TestEquityFloor:
    def test_below_floor_blocked(self):
        rm = _rm(min_equity=1_000.0)
        ok, reason = rm.can_trade(equity=500.0, recent_trades=[])
        assert not ok
        assert "minimum" in reason

    def test_at_floor_allowed(self):
        rm = _rm(min_equity=1_000.0)
        ok, _ = rm.can_trade(equity=1_000.0, recent_trades=[])
        assert ok

    def test_above_floor_allowed(self):
        rm = _rm(min_equity=1_000.0)
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=[])
        assert ok


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def _losses(self, n: int, day: int = 1) -> list[TradeRecord]:
        return [_trade(-100.0, hour=i, day=day) for i in range(n)]

    def test_two_losses_allowed(self):
        rm = _rm(max_consecutive_losses=3, cooldown_hours=24)
        trades = self._losses(2)
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc(hour=12))
        assert ok

    def test_three_losses_blocked_within_cooldown(self):
        rm = _rm(max_consecutive_losses=3, cooldown_hours=24)
        trades = self._losses(3)
        now = _utc(hour=12)  # losses ended at hour 2; within 24h cooldown
        ok, reason = rm.can_trade(equity=10_000.0, recent_trades=trades, now=now)
        assert not ok
        assert "circuit breaker" in reason
        assert "cooldown" in reason

    def test_three_losses_allowed_after_cooldown(self):
        rm = _rm(max_consecutive_losses=3, cooldown_hours=24)
        trades = self._losses(3, day=1)
        # now = 25 hours after last loss (which was day=1, hour=2)
        now = datetime(2024, 1, 2, 3, 0, tzinfo=timezone.utc)
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=now)
        assert ok

    def test_win_resets_consecutive_losses(self):
        rm = _rm(max_consecutive_losses=3, cooldown_hours=24)
        trades = [
            _trade(-100.0, hour=1),
            _trade(-100.0, hour=2),
            _trade(+200.0, hour=3),  # win breaks the streak
            _trade(-100.0, hour=4),
            _trade(-100.0, hour=5),
        ]
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc(hour=12))
        assert ok  # only 2 consecutive losses after the win

    def test_ten_losses_blocked(self):
        rm = _rm(max_consecutive_losses=3, cooldown_hours=24)
        trades = self._losses(10)
        ok, reason = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc(hour=12))
        assert not ok
        assert "circuit breaker" in reason

    def test_empty_trades_allowed(self):
        rm = _rm(max_consecutive_losses=3)
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=[], now=_utc())
        assert ok

    def test_cooldown_hours_in_reason(self):
        rm = _rm(max_consecutive_losses=3, cooldown_hours=24)
        trades = self._losses(3)
        _, reason = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc(hour=3))
        assert "h" in reason  # hours remaining mentioned


# ---------------------------------------------------------------------------
# Daily loss cap
# ---------------------------------------------------------------------------

class TestDailyLossCap:
    def test_within_cap_allowed(self):
        rm = _rm(daily_loss_cap=0.03)
        # -200 on equity 10_000 = 2% → below 3% cap
        trades = [_trade(-200.0)]
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc())
        assert ok

    def test_exactly_at_cap_allowed(self):
        rm = _rm(daily_loss_cap=0.03)
        trades = [_trade(-300.0)]  # exactly 3% of 10_000
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc())
        assert ok  # cap is strict <, not <=

    def test_exceeds_cap_blocked(self):
        rm = _rm(daily_loss_cap=0.03)
        trades = [_trade(-350.0)]  # 3.5% of 10_000
        ok, reason = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc())
        assert not ok
        assert "daily loss cap" in reason

    def test_only_todays_trades_count(self):
        rm = _rm(daily_loss_cap=0.03)
        now = _utc(day=2)
        trades = [
            _trade(-500.0, day=1),  # yesterday — should not count
            _trade(-100.0, day=2),  # today — 1% of 10_000 → fine
        ]
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=now)
        assert ok

    def test_multiple_losses_today_cumulative(self):
        rm = _rm(daily_loss_cap=0.03)
        trades = [
            _trade(-150.0, hour=9),
            _trade(+10.0,  hour=10),  # tiny win resets consecutive-loss streak
            _trade(-100.0, hour=11),
            _trade(-100.0, hour=12),  # total today = -340 = 3.4% → daily cap fires
        ]
        ok, reason = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc(hour=13))
        assert not ok
        assert "daily loss cap" in reason

    def test_wins_offset_losses(self):
        rm = _rm(daily_loss_cap=0.03)
        trades = [
            _trade(-400.0, hour=9),   # -4%
            _trade(+200.0, hour=10),  # net = -2% → allowed
        ]
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc(hour=12))
        assert ok


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSize:
    def test_basic_calculation(self):
        rm = _rm(risk_pct=0.01, min_position=0.01, max_position=100.0)
        # equity=10_000, 1% risk = $100; stop distance = 5 → 20 units
        units = rm.position_size(equity=10_000.0, entry=2350.0, stop=2345.0)
        assert abs(units - 20.0) < 1e-6

    def test_wide_stop_smaller_size(self):
        rm = _rm(risk_pct=0.01)
        narrow = rm.position_size(10_000.0, 2350.0, 2345.0)  # 5-pip stop → bigger
        wide   = rm.position_size(10_000.0, 2350.0, 2330.0)  # 20-pip stop → smaller
        assert narrow > wide

    def test_zero_stop_distance_returns_min(self):
        rm = _rm(min_position=0.01)
        units = rm.position_size(equity=10_000.0, entry=2350.0, stop=2350.0)
        assert units == 0.01

    def test_clamped_to_max(self):
        rm = _rm(risk_pct=0.01, max_position=5.0)
        # Very tight stop → huge unclamped size → should hit max
        units = rm.position_size(equity=10_000.0, entry=2350.0, stop=2349.99)
        assert units == 5.0

    def test_clamped_to_min(self):
        rm = _rm(risk_pct=0.01, min_position=1.0)
        # Huge stop → tiny unclamped size → should hit min
        units = rm.position_size(equity=100.0, entry=2350.0, stop=1.0)
        assert units == 1.0

    def test_short_position_same_as_long(self):
        rm = _rm(risk_pct=0.01)
        long_size  = rm.position_size(10_000.0, entry=2350.0, stop=2345.0)
        short_size = rm.position_size(10_000.0, entry=2345.0, stop=2350.0)
        assert abs(long_size - short_size) < 1e-6

    def test_scales_with_equity(self):
        rm = _rm(risk_pct=0.01)
        small = rm.position_size(5_000.0, entry=2350.0, stop=2345.0)
        large = rm.position_size(10_000.0, entry=2350.0, stop=2345.0)
        assert abs(large - 2 * small) < 1e-6


# ---------------------------------------------------------------------------
# Combined / integration
# ---------------------------------------------------------------------------

class TestCombined:
    def test_circuit_breaker_wins_over_daily_cap(self):
        """Both rules triggered — circuit breaker message is returned (checked first)."""
        rm = _rm(max_consecutive_losses=3, cooldown_hours=24, daily_loss_cap=0.01)
        trades = [
            TradeRecord(net_pnl=-200.0, exit_time=_utc(hour=i)) for i in range(3)
        ]
        ok, reason = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc(hour=12))
        assert not ok
        assert "circuit breaker" in reason

    def test_naive_datetime_handled(self):
        """exit_time without tzinfo should not crash."""
        rm = _rm()
        trades = [TradeRecord(net_pnl=-50.0, exit_time=datetime(2024, 1, 1, 10, 0))]
        ok, _ = rm.can_trade(equity=10_000.0, recent_trades=trades, now=_utc())
        assert isinstance(ok, bool)
