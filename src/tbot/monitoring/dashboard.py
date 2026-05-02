"""
TBot Streamlit monitoring dashboard — simple read-only view of SQLite.

Run with:
    streamlit run src/tbot/monitoring/dashboard.py
"""

from __future__ import annotations

import sqlite3
import time

import pandas as pd
import streamlit as st

from tbot.config import cfg

DB_PATH   = cfg.db_path
REFRESH_S = 30

st.set_page_config(page_title="TBot Dashboard", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _query(sql: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
            return pd.read_sql_query(sql, conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_S)
def load_trades() -> pd.DataFrame:
    return _query("""
        SELECT side, entry_time, entry_price, exit_time, exit_price,
               exit_reason, net_pnl, units, oanda_order_id
        FROM trades
        ORDER BY entry_time DESC
        LIMIT 50
    """)


@st.cache_data(ttl=REFRESH_S)
def load_signals() -> pd.DataFrame:
    return _query("""
        SELECT timestamp, side, model_score, source_strategy
        FROM signals
        ORDER BY timestamp DESC
        LIMIT 1000
    """)


@st.cache_data(ttl=REFRESH_S)
def load_backtest_runs() -> pd.DataFrame:
    return _query("""
        SELECT strategy, start_date, end_date,
               total_trades, win_rate, sharpe, max_drawdown, total_return
        FROM backtest_runs
        ORDER BY id DESC
        LIMIT 10
    """)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("📈 TBot — XAU/USD Paper Trading")
st.caption(f"DB: `{DB_PATH}` · refreshes every {REFRESH_S}s")

if st.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

trades  = load_trades()
signals = load_signals()

# --- Top metrics ---
total_trades = len(trades)
wins         = int((trades["net_pnl"] > 0).sum()) if not trades.empty and "net_pnl" in trades.columns else 0
win_rate     = wins / total_trades if total_trades > 0 else 0.0
total_pnl    = float(trades["net_pnl"].sum()) if not trades.empty and "net_pnl" in trades.columns else 0.0
open_trades  = int(trades["exit_time"].isna().sum()) if not trades.empty and "exit_time" in trades.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Trades",  total_trades)
c2.metric("Win Rate",      f"{win_rate:.1%}")
c3.metric("Net P&L",       f"${total_pnl:+,.2f}")
c4.metric("Open Trades",   open_trades)
c5.metric("Signals (1k)",  len(signals))

st.divider()

# --- Recent Trades ---
st.subheader("Recent Trades")
if trades.empty:
    st.info("No trades yet — start `run_paper_live.py` to populate this.")
else:
    # Rename columns for display
    display = trades.rename(columns={
        "entry_time":     "Entry Time",
        "entry_price":    "Entry",
        "exit_time":      "Exit Time",
        "exit_price":     "Exit",
        "exit_reason":    "Reason",
        "net_pnl":        "Net P&L",
        "units":          "Units",
        "oanda_order_id": "OANDA ID",
        "side":           "Side",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

st.divider()

# --- Signals ---
st.subheader("Recent Signals")
if signals.empty:
    st.info("No signals yet.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        counts = signals["side"].value_counts().reset_index()
        counts.columns = ["Side", "Count"]
        st.dataframe(counts, use_container_width=True, hide_index=True)
    with col_b:
        if "model_score" in signals.columns and signals["model_score"].notna().any():
            score_stats = signals["model_score"].describe().rename("Confidence Score")
            st.dataframe(score_stats.to_frame(), use_container_width=True)

st.divider()

# --- Backtest Runs ---
st.subheader("Backtest History")
runs = load_backtest_runs()
if runs.empty:
    st.info("No backtest runs recorded.")
else:
    st.dataframe(runs, use_container_width=True, hide_index=True)

# --- Auto-refresh ---
st.caption(f"Auto-refreshing every {REFRESH_S}s")
time.sleep(REFRESH_S)
st.rerun()
