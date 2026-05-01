"""
Tests for tbot.db.models + tbot.db.session.
Uses an in-memory SQLite database so tests are fast and isolated.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tbot.db.models import (
    BacktestRun,
    Base,
    Candle,
    EquitySnapshot,
    Event,
    ModelPrediction,
    RLExperience,
    Signal,
    Trade,
)


# ---------------------------------------------------------------------------
# In-memory DB fixture — each test gets a fresh database
# ---------------------------------------------------------------------------

@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    s = factory()
    yield s
    s.close()
    Base.metadata.drop_all(engine)


def _ts(year=2024, month=1, day=1, hour=0, minute=0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Candle
# ---------------------------------------------------------------------------

class TestCandle:
    def test_insert_and_query(self, session):
        c = Candle(
            instrument="XAU_USD", granularity="M5",
            timestamp=_ts(), open=1800.0, high=1802.0, low=1799.0, close=1801.0, volume=42,
        )
        session.add(c)
        session.commit()
        result = session.query(Candle).first()
        assert result.instrument == "XAU_USD"
        assert result.close == 1801.0
        assert result.volume == 42

    def test_multiple_candles(self, session):
        for i in range(5):
            session.add(Candle(
                instrument="XAU_USD", granularity="M5",
                timestamp=_ts(minute=i * 5),
                open=1800.0 + i, high=1802.0 + i, low=1799.0 + i,
                close=1801.0 + i, volume=i + 1,
            ))
        session.commit()
        assert session.query(Candle).count() == 5


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

class TestEvent:
    def test_insert_event(self, session):
        e = Event(
            instrument="XAU_USD", granularity="M5",
            timestamp=_ts(), event_type="BOS_BULL",
            direction="BULL", price_level=1801.0, is_active=True,
        )
        session.add(e)
        session.commit()
        result = session.query(Event).first()
        assert result.event_type == "BOS_BULL"
        assert result.is_active is True

    def test_optional_fields_nullable(self, session):
        e = Event(
            instrument="XAU_USD", granularity="M5",
            timestamp=_ts(), event_type="FVG_BEAR",
        )
        session.add(e)
        session.commit()
        result = session.query(Event).first()
        assert result.zone_high is None
        assert result.zone_low is None
        assert result.extra_json is None


# ---------------------------------------------------------------------------
# BacktestRun + relationships
# ---------------------------------------------------------------------------

class TestBacktestRun:
    def _make_run(self, session) -> BacktestRun:
        run = BacktestRun(
            created_at=_ts(),
            strategy="rule_v1",
            instrument="XAU_USD",
            granularity="M5",
            start_date=_ts(2024, 1, 1),
            end_date=_ts(2024, 12, 31),
            config_json=json.dumps({"min_confidence": 0.6}),
        )
        session.add(run)
        session.commit()
        return run

    def test_insert_run(self, session):
        run = self._make_run(session)
        result = session.query(BacktestRun).first()
        assert result.strategy == "rule_v1"
        assert result.sharpe is None  # not set yet

    def test_update_metrics(self, session):
        run = self._make_run(session)
        run.sharpe = 1.23
        run.total_trades = 47
        run.win_rate = 0.55
        session.commit()
        result = session.query(BacktestRun).first()
        assert result.sharpe == pytest.approx(1.23)
        assert result.total_trades == 47

    def test_signals_relationship(self, session):
        run = self._make_run(session)
        sig = Signal(
            backtest_run_id=run.id,
            instrument="XAU_USD", granularity="M5",
            timestamp=_ts(), side="BUY",
            entry_price=1800.0, stop_loss=1790.0, take_profit=1820.0,
            source_strategy="rule_v1",
        )
        session.add(sig)
        session.commit()
        session.refresh(run)
        assert len(run.signals) == 1
        assert run.signals[0].side == "BUY"

    def test_trades_relationship(self, session):
        run = self._make_run(session)
        trade = Trade(
            backtest_run_id=run.id,
            instrument="XAU_USD", side="BUY",
            units=0.1, entry_time=_ts(), entry_price=1800.0,
        )
        session.add(trade)
        session.commit()
        session.refresh(run)
        assert len(run.trades) == 1

    def test_equity_snapshots_relationship(self, session):
        run = self._make_run(session)
        for i in range(3):
            session.add(EquitySnapshot(
                backtest_run_id=run.id,
                timestamp=_ts(minute=i * 5),
                equity=10000.0 + i * 10,
                drawdown=0.0,
            ))
        session.commit()
        session.refresh(run)
        assert len(run.equity_snapshots) == 3


# ---------------------------------------------------------------------------
# Signal → Trade relationship
# ---------------------------------------------------------------------------

class TestSignalTradeRelationship:
    def test_signal_has_one_trade(self, session):
        sig = Signal(
            instrument="XAU_USD", granularity="M5",
            timestamp=_ts(), side="SELL",
            entry_price=1800.0, stop_loss=1810.0, take_profit=1780.0,
            source_strategy="rule_v1",
        )
        session.add(sig)
        session.commit()

        trade = Trade(
            signal_id=sig.id,
            instrument="XAU_USD", side="SELL",
            units=0.1, entry_time=_ts(), entry_price=1800.0,
            exit_time=_ts(minute=30), exit_price=1780.0,
            exit_reason="TP", gross_pnl=20.0, net_pnl=19.5,
        )
        session.add(trade)
        session.commit()

        session.refresh(sig)
        assert sig.trade is not None
        assert sig.trade.exit_reason == "TP"
        assert sig.trade.net_pnl == pytest.approx(19.5)


# ---------------------------------------------------------------------------
# ModelPrediction + RLExperience (schema smoke test)
# ---------------------------------------------------------------------------

class TestPhase5And10Tables:
    def test_model_prediction_insert(self, session):
        pred = ModelPrediction(
            model_name="xgb_v1", model_version="1.0",
            timestamp=_ts(), score=0.73,
            features_json=json.dumps({"atr": 2.1, "rsi": 58.0}),
        )
        session.add(pred)
        session.commit()
        result = session.query(ModelPrediction).first()
        assert result.score == pytest.approx(0.73)

    def test_rl_experience_insert(self, session):
        exp = RLExperience(
            timestamp=_ts(),
            state_json=json.dumps([0.1] * 10),
            action=1,
            reward=0.5,
            next_state_json=json.dumps([0.2] * 10),
            done=False,
        )
        session.add(exp)
        session.commit()
        result = session.query(RLExperience).first()
        assert result.action == 1
        assert result.done is False


# ---------------------------------------------------------------------------
# Session rollback on error
# ---------------------------------------------------------------------------

class TestSessionBehavior:
    def test_rollback_on_error(self, session):
        session.add(Candle(
            instrument="XAU_USD", granularity="M5",
            timestamp=_ts(), open=1800.0, high=1802.0, low=1799.0,
            close=1801.0, volume=1,
        ))
        session.commit()
        assert session.query(Candle).count() == 1

        try:
            session.add(Candle(
                instrument="XAU_USD", granularity="M5",
                timestamp=_ts(), open=None,  # type: ignore[arg-type]  # intentionally bad
                high=None, low=None, close=None, volume=None,
            ))
            session.commit()
        except Exception:
            session.rollback()

        assert session.query(Candle).count() == 1
