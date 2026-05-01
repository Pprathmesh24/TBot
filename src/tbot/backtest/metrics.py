"""
Compute backtest metrics from a vectorbt Portfolio object.
All functions return plain Python scalars — no vectorbt types leak out.
"""

from __future__ import annotations

import math


def compute_metrics(portfolio, init_cash: float = 10_000.0) -> dict:
    """
    Extract standard metrics from a vectorbt Portfolio.

    Returns a dict with keys:
        total_trades, total_return, max_drawdown, sharpe, sortino,
        calmar, win_rate, profit_factor, expectancy
    """
    total_return = float(portfolio.total_return())
    max_dd       = float(portfolio.max_drawdown())

    try:
        sharpe = float(portfolio.sharpe_ratio())
        if math.isnan(sharpe):
            sharpe = 0.0
    except Exception:
        sharpe = 0.0

    try:
        sortino = float(portfolio.sortino_ratio())
        if math.isnan(sortino):
            sortino = 0.0
    except Exception:
        sortino = 0.0

    calmar = (total_return / abs(max_dd)) if max_dd != 0 else 0.0

    # --- trade-level stats ---
    n_trades = win_rate = profit_factor = expectancy = 0.0
    try:
        trades_df = portfolio.trades.records_readable
        n_trades = len(trades_df)
        if n_trades > 0:
            pnl_col = _find_pnl_col(trades_df)
            pnl = trades_df[pnl_col]
            wins   = pnl[pnl > 0]
            losses = pnl[pnl <= 0]

            win_rate      = len(wins) / n_trades
            gross_win     = float(wins.sum())
            gross_loss    = abs(float(losses.sum()))
            profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
            avg_win       = float(wins.mean())   if len(wins)   > 0 else 0.0
            avg_loss      = abs(float(losses.mean())) if len(losses) > 0 else 0.0
            expectancy    = win_rate * avg_win - (1 - win_rate) * avg_loss
    except Exception:
        pass

    return {
        "total_trades":  int(n_trades),
        "total_return":  total_return,
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "calmar":        calmar,
        "win_rate":      float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy":    float(expectancy),
    }


def _find_pnl_col(df) -> str:
    """Find the PnL column regardless of vectorbt version naming."""
    for candidate in ("PnL", "pnl", "Profit & Loss", "profit_and_loss"):
        if candidate in df.columns:
            return candidate
    raise KeyError(f"No PnL column found. Available: {list(df.columns)}")
