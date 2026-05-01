"""
Backtest engine: runs AITradingAgent on historical candles,
feeds signals into vectorbt, and writes all results to the DB.

Entry point:
    run_id = run_backtest(candles_df)
    # data/tbot.sqlite now has BacktestRun, Signals, Trades, EquitySnapshots
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import vectorbt as vbt

from tbot.backtest.metrics import compute_metrics
from tbot.backtest.signals_adapter import adapt
from tbot.config import cfg
from tbot.core.agent import AITradingAgent
from tbot.data.loader import candles_to_dict_list
from tbot.db.models import BacktestRun, EquitySnapshot, Signal, Trade
from tbot.db.session import get_session, init_db

# Cost model: 2-pip spread + 0.5-pip slippage for XAU/USD M5
# At ~$2000/oz: 1 pip = $0.01 → 2.5 pips ≈ $0.025 ≈ 0.00125% of price
_FEES = 0.0000125


def _get_git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def run_backtest(
    candles_df: pd.DataFrame,
    strategy_name: str = "rule_v1",
    init_cash: float = 10_000.0,
    agent=None,
) -> int:
    """
    Full backtest pipeline.

    Args:
        candles_df:    DataFrame from load_candles()
        strategy_name: label stored in backtest_runs.strategy
        init_cash:     starting equity in USD
        agent:         optional agent instance; defaults to AITradingAgent (V1).
                       Pass an AITradingAgentV2() instance to run the SMC strategy.

    Returns:
        backtest_run_id (int) — the DB row id for this run
    """
    init_db()

    # --- 1. Run agent ---
    print(f"Running {strategy_name} on {len(candles_df):,} candles …")

    if agent is None:
        _agent = AITradingAgent(config={
            "save_reports": False,
            "enable_visualization": False,
            "enable_alerts": False,
        })
        _agent.analyzer.candles = candles_to_dict_list(candles_df)
        with contextlib.redirect_stdout(io.StringIO()):
            _agent.perform_initial_analysis()
            _agent.run_batch_analysis()
    else:
        _agent = agent
        with contextlib.redirect_stdout(io.StringIO()):
            _agent.run_on_df(candles_df)

    raw_signals: list[dict] = _agent.current_signals
    print(f"Agent produced {len(raw_signals)} raw signals")

    # --- 2. Align signals to candle index ---
    arrays = adapt(raw_signals, candles_df, min_confidence=cfg.min_confidence)
    print(f"Signals after confidence filter (≥{cfg.min_confidence}): {arrays.n_signals}")

    # --- 3. Convert absolute SL/TP to relative fractions for vectorbt ---
    close_vals = candles_df["close"].values
    sl_rel = np.where(
        arrays.entries & (arrays.sl_stop > 0),
        np.abs(close_vals - arrays.sl_stop) / close_vals,
        0.0,
    )
    tp_rel = np.where(
        arrays.entries & (arrays.tp_stop > 0),
        np.abs(arrays.tp_stop - close_vals) / close_vals,
        0.0,
    )

    # --- 4. Build vectorbt close series (DatetimeIndex required) ---
    close_series = pd.Series(
        close_vals,
        index=pd.DatetimeIndex(candles_df["timestamp"]),
    )

    # --- 5. Run vectorbt portfolio ---
    portfolio = vbt.Portfolio.from_signals(
        close=close_series,
        entries=arrays.entries,
        exits=arrays.exits,
        sl_stop=sl_rel,
        tp_stop=tp_rel,
        init_cash=init_cash,
        fees=_FEES,
        freq="5min",
    )

    # --- 6. Compute metrics ---
    metrics = compute_metrics(portfolio, init_cash=init_cash)
    print(
        f"Backtest complete — {metrics['total_trades']} trades  "
        f"Sharpe={metrics['sharpe']:.3f}  "
        f"MaxDD={metrics['max_drawdown']:.1%}  "
        f"WinRate={metrics['win_rate']:.1%}"
    )

    # --- 7. Write everything to DB ---
    run_id = _write_to_db(
        candles_df=candles_df,
        raw_signals=raw_signals,
        portfolio=portfolio,
        metrics=metrics,
        strategy_name=strategy_name,
        init_cash=init_cash,
    )
    print(f"Results saved → backtest_run_id={run_id}")
    return run_id


# ---------------------------------------------------------------------------
# DB write helpers
# ---------------------------------------------------------------------------

def _write_to_db(
    candles_df: pd.DataFrame,
    raw_signals: list[dict],
    portfolio,
    metrics: dict,
    strategy_name: str,
    init_cash: float,
) -> int:
    with get_session() as session:
        run = BacktestRun(
            created_at=datetime.now(timezone.utc),
            strategy=strategy_name,
            instrument="XAU_USD",
            granularity="M5",
            start_date=candles_df["timestamp"].iloc[0].to_pydatetime(),
            end_date=candles_df["timestamp"].iloc[-1].to_pydatetime(),
            config_json=json.dumps({
                "min_confidence": cfg.min_confidence,
                "init_cash":      init_cash,
                "fees":           _FEES,
                "strategy":       strategy_name,
            }),
            git_sha=_get_git_sha(),
            total_trades=metrics["total_trades"],
            win_rate=metrics["win_rate"],
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            calmar=metrics["calmar"],
            profit_factor=metrics["profit_factor"],
            max_drawdown=metrics["max_drawdown"],
            total_return=metrics["total_return"],
        )
        session.add(run)
        session.flush()  # get run.id before committing

        # --- signals ---
        for sig in raw_signals:
            ts = pd.Timestamp(sig["timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            session.add(Signal(
                backtest_run_id=run.id,
                instrument="XAU_USD",
                granularity="M5",
                timestamp=ts.to_pydatetime(),
                side=sig.get("type", "BUY"),
                entry_price=float(sig.get("price", 0.0)),
                stop_loss=float(sig.get("stop_loss", 0.0)),
                take_profit=float(sig.get("take_profit", 0.0)),
                source_strategy=strategy_name,
                model_score=float(sig.get("confidence", 0.0)),
            ))

        # --- trades from vectorbt ---
        try:
            _write_trades(session, run.id, portfolio)
        except Exception as exc:
            print(f"  Warning: could not write trade records: {exc}")

        # --- equity curve (sampled every 100 candles) ---
        try:
            _write_equity(session, run.id, portfolio)
        except Exception as exc:
            print(f"  Warning: could not write equity snapshots: {exc}")

        run_id = run.id

    return run_id


def _write_trades(session, run_id: int, portfolio) -> None:
    try:
        trades_df = portfolio.trades.records_readable
    except Exception:
        return
    if len(trades_df) == 0:
        return

    col = {c.lower(): c for c in trades_df.columns}

    for _, row in trades_df.iterrows():
        entry_ts = _to_dt(row.get(col.get("entry timestamp") or col.get("entry idx")))
        exit_ts  = _to_dt(row.get(col.get("exit timestamp")  or col.get("exit idx")))
        pnl_col  = col.get("pnl") or col.get("profit & loss")
        pnl      = float(row[pnl_col]) if pnl_col else None

        session.add(Trade(
            backtest_run_id=run_id,
            instrument="XAU_USD",
            side="BUY",
            units=1.0,
            entry_time=entry_ts or datetime.now(timezone.utc),
            entry_price=float(row.get(col.get("entry price"), 0.0)),
            exit_time=exit_ts,
            exit_price=float(row.get(col.get("exit price"), 0.0)) if col.get("exit price") else None,
            exit_reason=str(row.get(col.get("exit type"), "")),
            gross_pnl=pnl,
            net_pnl=pnl,
        ))


def _write_equity(session, run_id: int, portfolio, sample_every: int = 100) -> None:
    equity = portfolio.value()
    peak   = equity.cummax()
    dd     = (equity - peak) / peak

    indices = range(0, len(equity), sample_every)
    for i in indices:
        session.add(EquitySnapshot(
            backtest_run_id=run_id,
            timestamp=equity.index[i].to_pydatetime(),
            equity=float(equity.iloc[i]),
            drawdown=float(dd.iloc[i]),
        ))


def _to_dt(val) -> datetime | None:
    if val is None:
        return None
    try:
        ts = pd.Timestamp(val)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()
    except Exception:
        return None
