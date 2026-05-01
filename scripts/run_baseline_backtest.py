"""
Run the full 5-year baseline backtest and print the results.

Usage:
    python scripts/run_baseline_backtest.py
    python scripts/run_baseline_backtest.py --start 2023-01-01 --end 2024-12-31

This is the bar every later phase (SMC, ML, RL) must beat.
Results are written to data/tbot.sqlite.
"""

from __future__ import annotations

import argparse

from tbot.backtest.engine import run_backtest
from tbot.data.loader import load_candles
from tbot.db.models import BacktestRun
from tbot.db.session import get_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TBot baseline backtest")
    parser.add_argument("--start", default=None, help="Start date e.g. 2020-01-01")
    parser.add_argument("--end",   default=None, help="End date   e.g. 2025-12-31")
    args = parser.parse_args()

    print("=" * 60)
    print("TBot Baseline Backtest — rule_v1")
    print("=" * 60)

    df = load_candles(start=args.start, end=args.end)
    print(f"Loaded {len(df):,} candles  "
          f"({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})\n")

    run_id = run_backtest(df, strategy_name="rule_v1_baseline")

    # Pretty-print the results from DB
    with get_session() as s:
        run = s.query(BacktestRun).filter_by(id=run_id).one()
        print("\n" + "=" * 60)
        print("BASELINE RESULTS")
        print("=" * 60)
        print(f"  Strategy   : {run.strategy}")
        print(f"  Period     : {run.start_date.date()} → {run.end_date.date()}")
        print(f"  Total trades: {run.total_trades}")
        print(f"  Win rate   : {run.win_rate:.1%}")
        print(f"  Total return: {run.total_return:.2%}")
        print(f"  Max drawdown: {run.max_drawdown:.2%}")
        print(f"  Sharpe     : {run.sharpe:.3f}  ← baseline to beat")
        print(f"  Sortino    : {run.sortino:.3f}")
        print(f"  Calmar     : {run.calmar:.3f}")
        print(f"  Profit factor: {run.profit_factor:.2f}")
        print(f"  DB run_id  : {run.id}")
        print("=" * 60)


if __name__ == "__main__":
    main()
