"""
Tests for RiskState — DB-backed wrapper around RiskManager.
Uses an in-memory SQLite DB so no file is created on disk.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tbot.db.models import Base, BacktestRun, Trade
from tbot.risk.manager import RiskManager
from tbot.risk.state import RiskState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    return e


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def _utc(year=2024, month=1, day=1, hour=12) -> datetime:
    return datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)


def _add_trade(session, net_pnl, exit_hour, instrument="XAU_USD", run_id=None):
    t = Trade(
        instrument      = instrument,
        side            = "BUY",
        units           = 1.0,
        entry_time      = _utc(hour=exit_hour - 1),
        entry_price     = 2350.0,
        exit_time       = _utc(hour=exit_hour),
        exit_price      = 2351.0,
        exit_reason     = "TP",
        gross_pnl       = net_pnl,
        net_pnl         = net_pnl,
        backtest_run_id = run_id,
    )
    session.add(t)
    session.flush()
    return t


def _add_run(session) -> BacktestRun:
    run = BacktestRun(
        created_at  = _utc(),
        strategy    = "test",
        instrument  = "XAU_USD",
        granularity = "M5",
        start_date  = _utc(),
        end_date    = _utc(hour=23),
        config_json = "{}",
    )
    session.add(run)
    session.flush()
    return run


# ---------------------------------------------------------------------------
# Basic queries
# ---------------------------------------------------------------------------

class TestLoadTrades:
    def test_empty_db_allowed(self, session):
        rs = RiskState(session)
        ok, _ = rs.can_trade(equity=10_000.0, now=_utc())
        assert ok

    def test_loads_completed_trades(self, session):
        _add_trade(session, -200.0, exit_hour=9)
        _add_trade(session, -200.0, exit_hour=10)
        # total today = -400 = 4% → over 3% cap → blocked
        rs = RiskState(session, rm=RiskManager(daily_loss_cap=0.03))
        ok, reason = rs.can_trade(equity=10_000.0, now=_utc(hour=12))
        assert not ok
        assert "daily loss cap" in reason

    def test_open_trade_excluded(self, session):
        # Trade with no exit_time is in-flight — must not count
        t = Trade(
            instrument  = "XAU_USD",
            side        = "BUY",
            units       = 1.0,
            entry_time  = _utc(hour=9),
            entry_price = 2350.0,
            # exit_time / net_pnl intentionally omitted
        )
        session.add(t)
        session.flush()
        rs = RiskState(session)
        ok, _ = rs.can_trade(equity=10_000.0, now=_utc(hour=12))
        assert ok  # open trade should not trigger any rule

    def test_filters_by_instrument(self, session):
        _add_trade(session, -500.0, exit_hour=9, instrument="EUR_USD")
        # XAU_USD has no trades — should be allowed
        rs = RiskState(session, instrument="XAU_USD")
        ok, _ = rs.can_trade(equity=10_000.0, now=_utc(hour=12))
        assert ok

    def test_filters_by_backtest_run_id(self, session):
        run_a = _add_run(session)
        run_b = _add_run(session)
        _add_trade(session, -500.0, exit_hour=9, run_id=run_a.id)
        _add_trade(session, -500.0, exit_hour=10, run_id=run_b.id)
        # Run B only has one -500 trade = 5% → blocked
        rs_b = RiskState(session, backtest_run_id=run_b.id, rm=RiskManager(daily_loss_cap=0.03))
        ok, reason = rs_b.can_trade(equity=10_000.0, now=_utc(hour=12))
        assert not ok
        # Run A should also see only its own trade
        rs_a = RiskState(session, backtest_run_id=run_a.id, rm=RiskManager(daily_loss_cap=0.03))
        ok_a, reason_a = rs_a.can_trade(equity=10_000.0, now=_utc(hour=12))
        assert not ok_a  # also -500 = 5% of 10k

    def test_no_run_id_loads_all_instrument_trades(self, session):
        run = _add_run(session)
        _add_trade(session, -200.0, exit_hour=9, run_id=run.id)
        _add_trade(session, -200.0, exit_hour=10, run_id=None)  # live trade
        # Total = -400 = 4% → blocked
        rs = RiskState(session, rm=RiskManager(daily_loss_cap=0.03))
        ok, _ = rs.can_trade(equity=10_000.0, now=_utc(hour=12))
        assert not ok


# ---------------------------------------------------------------------------
# position_size passthrough
# ---------------------------------------------------------------------------

class TestPositionSize:
    def test_delegates_to_rm(self, session):
        rs = RiskState(session, rm=RiskManager(risk_pct=0.01))
        units = rs.position_size(equity=10_000.0, entry=2350.0, stop=2345.0)
        assert abs(units - 20.0) < 1e-6

    def test_no_db_query_needed(self, session):
        # Even with losses in DB, position_size works (pure math, no gate)
        _add_trade(session, -500.0, exit_hour=9)
        rs = RiskState(session, rm=RiskManager(risk_pct=0.02))
        units = rs.position_size(equity=5_000.0, entry=2350.0, stop=2340.0)
        assert units == pytest.approx(10.0)
