"""
SQLAlchemy 2.0 ORM models — the single source of truth for all TBot data.

Every phase writes to these tables:
  Phase 1+  → candles
  Phase 3   → events
  Phase 2+  → signals, trades, equity_snapshots, backtest_runs
  Phase 5   → model_predictions
  Phase 10  → rl_experiences
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# candles
# ---------------------------------------------------------------------------

class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        Index("ix_candles_instrument_ts", "instrument", "granularity", "timestamp"),
    )

    id:          Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument:  Mapped[str]   = mapped_column(String(16), nullable=False)
    granularity: Mapped[str]   = mapped_column(String(8),  nullable=False)
    timestamp:   Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open:        Mapped[float] = mapped_column(Float, nullable=False)
    high:        Mapped[float] = mapped_column(Float, nullable=False)
    low:         Mapped[float] = mapped_column(Float, nullable=False)
    close:       Mapped[float] = mapped_column(Float, nullable=False)
    volume:      Mapped[int]   = mapped_column(Integer, nullable=False)


# ---------------------------------------------------------------------------
# events  (BOS / ChoCH / FVG / OrderBlock / LiquiditySweep)
# ---------------------------------------------------------------------------

class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        Index("ix_events_instrument_ts", "instrument", "timestamp"),
    )

    id:           Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument:   Mapped[str]  = mapped_column(String(16), nullable=False)
    granularity:  Mapped[str]  = mapped_column(String(8),  nullable=False)
    timestamp:    Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    event_type:   Mapped[str]  = mapped_column(String(32), nullable=False)   # BOS_BULL, FVG_BEAR, …
    direction:    Mapped[str | None]  = mapped_column(String(8),  nullable=True)   # BULL / BEAR
    price_level:  Mapped[float | None] = mapped_column(Float, nullable=True)
    zone_high:    Mapped[float | None] = mapped_column(Float, nullable=True)
    zone_low:     Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active:    Mapped[bool] = mapped_column(Boolean, default=True)
    extra_json:   Mapped[str | None]  = mapped_column(Text, nullable=True)   # arbitrary extra data

    signals: Mapped[list["Signal"]] = relationship("Signal", back_populates="trigger_event")


# ---------------------------------------------------------------------------
# backtest_runs
# ---------------------------------------------------------------------------

class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id:           Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    strategy:     Mapped[str]   = mapped_column(String(64), nullable=False)
    instrument:   Mapped[str]   = mapped_column(String(16), nullable=False)
    granularity:  Mapped[str]   = mapped_column(String(8),  nullable=False)
    start_date:   Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date:     Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    config_json:  Mapped[str]   = mapped_column(Text, nullable=False)       # full TBotConfig snapshot
    git_sha:      Mapped[str | None]  = mapped_column(String(40), nullable=True)

    # summary metrics (written after run completes)
    total_trades: Mapped[int | None]   = mapped_column(Integer, nullable=True)
    win_rate:     Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe:       Mapped[float | None] = mapped_column(Float, nullable=True)
    sortino:      Mapped[float | None] = mapped_column(Float, nullable=True)
    calmar:       Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_return: Mapped[float | None] = mapped_column(Float, nullable=True)

    signals:          Mapped[list["Signal"]]         = relationship("Signal",         back_populates="backtest_run")
    trades:           Mapped[list["Trade"]]          = relationship("Trade",          back_populates="backtest_run")
    equity_snapshots: Mapped[list["EquitySnapshot"]] = relationship("EquitySnapshot", back_populates="backtest_run")


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

class Signal(Base):
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signals_instrument_ts", "instrument", "timestamp"),
    )

    id:              Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("backtest_runs.id"), nullable=True)
    trigger_event_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("events.id"), nullable=True)
    instrument:      Mapped[str]   = mapped_column(String(16), nullable=False)
    granularity:     Mapped[str]   = mapped_column(String(8),  nullable=False)
    timestamp:       Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    side:            Mapped[str]   = mapped_column(String(8),  nullable=False)   # BUY / SELL
    entry_price:     Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss:       Mapped[float] = mapped_column(Float, nullable=False)
    take_profit:     Mapped[float] = mapped_column(Float, nullable=False)
    source_strategy: Mapped[str]   = mapped_column(String(64), nullable=False)  # rule_v1 / smc_v2 / …
    model_score:     Mapped[float | None] = mapped_column(Float, nullable=True)  # None until Phase 5
    features_json:   Mapped[str | None]  = mapped_column(Text, nullable=True)   # None until Phase 4
    label:           Mapped[str | None]  = mapped_column(String(8), nullable=True)  # WIN/LOSS/NEUTRAL (Phase 4)

    backtest_run:  Mapped["BacktestRun | None"] = relationship("BacktestRun", back_populates="signals")
    trigger_event: Mapped["Event | None"]       = relationship("Event",       back_populates="signals")
    trade:         Mapped["Trade | None"]        = relationship("Trade",       back_populates="signal", uselist=False)


# ---------------------------------------------------------------------------
# trades
# ---------------------------------------------------------------------------

class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trades_instrument_entry", "instrument", "entry_time"),
    )

    id:              Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("backtest_runs.id"), nullable=True)
    signal_id:       Mapped[int | None] = mapped_column(Integer, ForeignKey("signals.id"), nullable=True)
    instrument:      Mapped[str]   = mapped_column(String(16), nullable=False)
    side:            Mapped[str]   = mapped_column(String(8),  nullable=False)   # BUY / SELL
    units:           Mapped[float] = mapped_column(Float, nullable=False)
    entry_time:      Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entry_price:     Mapped[float] = mapped_column(Float, nullable=False)
    exit_time:       Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price:      Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason:     Mapped[str | None]   = mapped_column(String(16), nullable=True)  # SL / TP / MANUAL / TIMEOUT
    gross_pnl:       Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage:        Mapped[float | None] = mapped_column(Float, nullable=True)
    commission:      Mapped[float | None] = mapped_column(Float, nullable=True)
    net_pnl:         Mapped[float | None] = mapped_column(Float, nullable=True)
    oanda_order_id:  Mapped[str | None]   = mapped_column(String(32), nullable=True)  # Phase 7

    backtest_run: Mapped["BacktestRun | None"] = relationship("BacktestRun", back_populates="trades")
    signal:       Mapped["Signal | None"]      = relationship("Signal",      back_populates="trade")


# ---------------------------------------------------------------------------
# equity_snapshots
# ---------------------------------------------------------------------------

class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    __table_args__ = (
        Index("ix_equity_run_ts", "backtest_run_id", "timestamp"),
    )

    id:              Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[int]   = mapped_column(Integer, ForeignKey("backtest_runs.id"), nullable=False)
    timestamp:       Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    equity:          Mapped[float] = mapped_column(Float, nullable=False)
    drawdown:        Mapped[float] = mapped_column(Float, nullable=False)  # as fraction, e.g. 0.05 = 5%

    backtest_run: Mapped["BacktestRun"] = relationship("BacktestRun", back_populates="equity_snapshots")


# ---------------------------------------------------------------------------
# model_predictions  (Phase 5 — drift detection)
# ---------------------------------------------------------------------------

class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id:           Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id:    Mapped[int | None] = mapped_column(Integer, ForeignKey("signals.id"), nullable=True)
    model_name:   Mapped[str]   = mapped_column(String(64), nullable=False)
    model_version: Mapped[str]  = mapped_column(String(16), nullable=False)
    timestamp:    Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    score:        Mapped[float] = mapped_column(Float, nullable=False)
    features_json: Mapped[str]  = mapped_column(Text, nullable=False)


# ---------------------------------------------------------------------------
# rl_experiences  (Phase 10 — replay buffer)
# ---------------------------------------------------------------------------

class RLExperience(Base):
    __tablename__ = "rl_experiences"

    id:          Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id:    Mapped[int | None] = mapped_column(Integer, ForeignKey("trades.id"), nullable=True)
    timestamp:   Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    state_json:  Mapped[str]  = mapped_column(Text, nullable=False)
    action:      Mapped[int]  = mapped_column(Integer, nullable=False)   # 0-4 discrete
    reward:      Mapped[float] = mapped_column(Float, nullable=False)
    next_state_json: Mapped[str] = mapped_column(Text, nullable=False)
    done:        Mapped[bool] = mapped_column(Boolean, nullable=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def json_loads_safe(text: str | None) -> dict:
    if not text:
        return {}
    return json.loads(text)
