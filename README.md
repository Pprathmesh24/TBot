# TBot — AI Gold Trading System

> **Status: Active development — currently in paper trading phase.**
> Phase 10 (Reinforcement Learning execution agent) begins after 4 weeks of live paper trade data is collected.

A production-grade algorithmic trading system for XAU/USD (Gold) on the M5 timeframe.
Built from scratch: SMC market structure detection → XGBoost ML confidence filter → OANDA paper broker → live monitoring.

**Paper trading only. No real capital is used.**

### Roadmap

| Phase | Description | Status |
|---|---|---|
| 0–8 | Core system: data, SMC, ML, risk, broker, monitoring | ✅ Complete |
| 9 | Macro data pipeline (DXY, VIX, TIPS yield, CFTC COT) | ✅ Complete |
| 10 | RL execution agent (PPO via Stable-Baselines3) | ⏳ Pending — needs 4 weeks paper trade data |
| 11 | Portfolio polish: architecture docs, CI/CD, final README | ⏳ Pending |

---

## Results

### Model validation (historical backtest — not live results)

| Metric | Value | What it means |
|---|---|---|
| Walk-forward AUC (22 folds, 5 years) | **0.6354** | Model's ability to rank winning vs losing signals on unseen data |
| Calibrated win rate at threshold 0.60 | **68.7%** on 7% of signals | Of historical signals scored ≥0.60, 68.7% hit TP before SL |
| Training signals labeled | 82,849 | Triple-barrier labeled signals over 5 years |
| Candle history | 448,782 M5 bars (2020 → present) | |
| Baseline Sharpe (rule-only, no ML) | -0.656 | The bar the ML model must beat in live trading |

> **Note:** These are pre-live estimates from backtested data. Live P&L, live win rate, and live Sharpe will be reported here after the paper trading phase completes (Phase 10 prerequisite).

---

## Architecture

```
OANDA streaming
      │
      ▼
EnrichedMarketStructureAnalyzer
  ├── BOS / ChoCH detection
  ├── Fair Value Gaps (FVG)
  ├── Order Blocks
  └── Liquidity Sweeps
      │
      ▼
Feature Engine (34 features)
  ├── Volatility: ATR, Bollinger bands
  ├── Momentum: RSI, MACD, EMA slopes
  ├── SMC context: distance to FVG/OB zones
  └── Time: session flags, hour, day-of-week
      │
      ▼
XGBoost confidence model
  └── Isotonic calibration (0.60 threshold → 68.7% win rate)
      │
      ▼
RiskManager gate
  ├── 1% equity per trade
  ├── ATR-based dynamic stops
  ├── Circuit breaker: 3 consecutive losses → 24h cooldown
  └── Daily loss cap: 3% of equity
      │
      ▼
OANDA paper order
  └── SQLite DB → Streamlit dashboard + Slack alerts
```

---

## Tech Stack

| Concern | Choice |
|---|---|
| ML model | XGBoost + isotonic calibration |
| Validation | Purged + embargoed walk-forward CV (López de Prado) |
| Broker | OANDA demo via `oandapyV20` |
| Database | SQLite + SQLAlchemy 2.0 ORM |
| Config | pydantic-settings (`.env` → typed config) |
| Macro data | yfinance (DXY, VIX) + FRED API (10Y TIPS yield) + CFTC COT |
| Dashboard | Streamlit |
| Logging | structlog → JSON |
| Package layout | `src/` + `pyproject.toml` + `uv` |

---

## Quick Start

### 1. Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- OANDA demo account (free at [oanda.com](https://www.oanda.com)) — needed for live data + paper trading
- FRED API key (free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)) — needed for macro data

### 2. Install

```bash
git clone https://github.com/your-username/TBot.git
cd TBot
uv venv
uv pip install -e ".[all]"
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and fill in:
#   OANDA__ACCOUNT_ID
#   OANDA__API_TOKEN
#   FRED_API_KEY
#   SLACK_WEBHOOK_URL   (optional — for trade alerts)
```

### 4. Fetch historical data

```bash
# 448k M5 candles, 2020 → present (~2 min)
.venv/bin/python src/tbot/data/fetch_oanda_history.py
```

### 5. Run a backtest

```bash
.venv/bin/python scripts/run_baseline_backtest.py
```

### 6. Build the ML training dataset

```bash
# Runs SMC analyzer + triple-barrier labeling (~20 min on 448k candles)
.venv/bin/python scripts/build_training_dataset.py
```

### 7. Train the XGBoost confidence model

```bash
# Walk-forward training, 22 folds
.venv/bin/python scripts/train_model.py

# Calibrate probabilities
.venv/bin/python scripts/calibrate_model.py
```

### 8. Refresh macro data (optional)

```bash
.venv/bin/python scripts/refresh_macro_data.py
```

### 9. Run paper trading (live)

```bash
# Streams XAU/USD M5 candles, places paper orders on OANDA demo
.venv/bin/python scripts/run_paper_live.py
```

### 10. Monitor

```bash
streamlit run src/tbot/monitoring/dashboard.py
# Open http://localhost:8501
```

---

## Run Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

---

## Project Structure

```
TBot/
├── src/tbot/
│   ├── config.py                    # pydantic-settings typed config
│   ├── core/
│   │   ├── market_structure.py      # BOS / ChoCH detector
│   │   ├── agent_v2.py              # SMC + ML signal agent
│   │   └── smc/
│   │       ├── fvg.py               # Fair Value Gap detector
│   │       ├── order_blocks.py      # Order Block detector
│   │       ├── liquidity.py         # Liquidity sweep detector
│   │       └── structure_v2.py      # Enriched analyzer (combines all)
│   ├── data/
│   │   ├── loader.py                # Parquet loader + validation
│   │   ├── fetch_oanda_history.py   # Historical candle downloader
│   │   └── macro/
│   │       ├── dxy.py               # DXY via yfinance
│   │       ├── vix.py               # VIX via yfinance
│   │       ├── yields.py            # 10Y TIPS real yield via FRED
│   │       └── cot.py               # CFTC COT gold positioning
│   ├── features/
│   │   ├── builder.py               # 34-feature snapshot per signal
│   │   ├── labeler.py               # Triple-barrier labeling
│   │   └── macro_features.py        # Macro feature lookup (batch API)
│   ├── ml/
│   │   ├── walk_forward.py          # Purged + embargoed CV splitter
│   │   ├── train.py                 # XGBoost walk-forward training
│   │   ├── calibration.py           # Isotonic probability calibration
│   │   └── predict.py               # score_signal() inference
│   ├── backtest/
│   │   ├── engine.py                # vectorbt-based backtest engine
│   │   ├── signals_adapter.py       # Signals → vbt arrays
│   │   └── metrics.py               # Sharpe, Sortino, Calmar, etc.
│   ├── risk/
│   │   ├── manager.py               # RiskManager: sizing + circuit breaker
│   │   └── state.py                 # DB-backed risk state
│   ├── broker/
│   │   ├── oanda_client.py          # OANDA REST client
│   │   ├── stream.py                # Tick → M5 candle aggregator
│   │   └── executor.py              # Signal → paper order + DB write
│   ├── live/
│   │   └── runner.py                # Full integration loop
│   ├── monitoring/
│   │   ├── logger.py                # structlog → JSON
│   │   ├── alerts.py                # Slack webhook alerts
│   │   └── dashboard.py             # Streamlit dashboard
│   └── db/
│       ├── models.py                # SQLAlchemy ORM (8 tables)
│       └── session.py               # Engine + session factory
├── scripts/                         # Runnable entry points
├── tests/                           # pytest test suite
├── models/                          # Trained model artifacts
│   ├── xgb_v1.pkl
│   └── calibrator_v1.pkl
├── data/
│   ├── raw/                         # Parquet files (git-ignored)
│   └── features/                    # Labeled training dataset
├── deploy/
│   └── launchd/                     # macOS service config
├── docs/
│   └── PROGRESS.md                  # Build journal
└── pyproject.toml
```

---

## Key Design Decisions

**Walk-forward validation with purging + embargo** — prevents label leakage that inflates backtest AUC. Each fold has a 1-week embargo gap between train and test. Based on López de Prado's *Advances in Financial Machine Learning*.

**Triple-barrier labeling** — signals labeled WIN/LOSS/NEUTRAL based on which barrier (TP, SL, or 48-candle timeout) is hit first. More honest than fixed-horizon labeling for M5 trades.

**Isotonic calibration** — raw XGBoost scores are overconfident. Calibration maps them to true win probabilities so "0.65" actually means 65% historical win rate.

**SQLite central database** — every signal, trade, ML prediction, and equity snapshot written to `data/tbot.sqlite`. Single source of truth for backtest, live trading, and future RL training.

**ATR-based position sizing** — stop distance varies with volatility. Larger ATR = smaller position. Keeps risk per trade at 1% of equity regardless of market conditions.

---

## Macro Data

Four macro series are fetched and stored as daily parquets:

| Series | Source | Purpose |
|---|---|---|
| DXY (Dollar Index) | yfinance | Gold's primary inverse driver |
| VIX (Volatility Index) | yfinance | Risk-on / risk-off regime |
| 10Y TIPS Real Yield | FRED `DFII10` | Strongest gold fundamental driver |
| CFTC COT Gold | cftc.gov | Managed money positioning (weekly) |

**Finding:** At M5 granularity, macro features showed ~1% model importance (daily data repeats 288× per day). Used as regime context rather than ML features.

---

## Disclaimer

This project is for educational and research purposes. Paper trading only — no real capital is used. Past performance does not guarantee future results.
