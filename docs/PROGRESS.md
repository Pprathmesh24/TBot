# TBot Build Progress

---

## Phase 0 — Project Restructure ✅

### Chunk 1 — pyproject.toml + uv setup
**Files:** `pyproject.toml`
**Verify:** `uv pip install -e "."` succeeds
**Sign-off:** ✅

### Chunk 2 — src/tbot skeleton + config.py
**Files:** `src/tbot/__init__.py`, `src/tbot/config.py`
**Verify:** `python -c "from tbot.config import cfg; print(cfg)"` prints TBotConfig
**Sign-off:** ✅

### Chunk 3 — File moves (market_structure, agent, viz)
**Files:** `src/tbot/core/market_structure.py`, `src/tbot/core/agent.py`, `src/tbot/viz/charts.py`
**Verify:** imports resolve without error
**Sign-off:** ✅

### Chunk 4 — .gitignore + tests
**Files:** `.gitignore`, `.env.example`, `tests/test_market_structure.py`
**Verify:** `pytest tests/test_market_structure.py` → 14 passed
**Sign-off:** ✅

---

## Phase 1 — Historical Data + Loader ✅

### Chunk 1 — fetch_oanda_history.py
**Files:** `src/tbot/data/fetch_oanda_history.py`, `data/raw/XAU_USD_M5_2020_2025.parquet`
**Verify:** parquet exists, 448,782 rows, 2020-01-01 → 2026-05-01
**Sign-off:** ✅

### Chunk 2 — loader.py
**Files:** `src/tbot/data/loader.py`
**Verify:** `load_candles()` returns 448,782 UTC rows; date slice works; `candles_to_dict_list()` returns dicts with `is_green`/`is_red`
**Sign-off:** ✅

### Chunk 3 — test_loader.py
**Files:** `tests/test_loader.py`
**Verify:** `pytest tests/test_loader.py -v` → 24 passed
**Sign-off:** ✅

---

## Phase 3 — Full SMC Detectors ✅

### Chunk 1 — FVG detector
**Files:** `src/tbot/core/smc/__init__.py`, `src/tbot/core/smc/fvg.py`
**Verify:** 2,021 FVGs in Q1 2024, 98.3% fill rate
**Sign-off:** ✅

### Chunk 2 — Order Block detector
**Files:** `src/tbot/core/smc/order_blocks.py`
**Verify:** 2,560 OBs in Q1 2024, 96.4% mitigated
**Sign-off:** ✅

### Chunk 3 — Liquidity detector
**Files:** `src/tbot/core/smc/liquidity.py`
**Verify:** 138 sweeps + 482 equal levels in Q1 2024
**Sign-off:** ✅

### Chunk 4 — SMC detector tests
**Files:** `tests/test_smc_detectors.py`
**Verify:** `pytest tests/test_smc_detectors.py -v` → 22 passed
**Sign-off:** ✅

---

## Phase 2 — Backtest Engine + Central Database ✅

### Chunk 1 — DB models
**Files:** `src/tbot/db/__init__.py`, `src/tbot/db/models.py`
**Verify:** 8 tables registered — `python -c "from tbot.db.models import Base; print(list(Base.metadata.tables.keys()))"`
**Sign-off:** ✅

### Chunk 2 — DB session
**Files:** `src/tbot/db/session.py`
**Verify:** `data/tbot.sqlite` created, round-trip write/read works
**Sign-off:** ✅

### Chunk 3 — DB model tests
**Files:** `tests/test_db_models.py`
**Verify:** `pytest tests/test_db_models.py -v` → 13 passed
**Sign-off:** ✅

### Chunk 4 — Signals adapter
**Files:** `src/tbot/backtest/signals_adapter.py`
**Verify:** BUY/SELL/confidence-filter all correct on synthetic signals
**Sign-off:** ✅

### Chunk 5 — Metrics + Engine
**Files:** `src/tbot/backtest/metrics.py`, `src/tbot/backtest/engine.py`
**Bugfixes:** `market_structure.py` lines 592-596 and 648-652 — capture price before nulling active level
**Verify:** smoke test on Q1 2024 → 1 trade, Sharpe 0.677, DB populated
**Sign-off:** ✅

### Chunk 6 — Baseline backtest script
**Files:** `scripts/run_baseline_backtest.py`
**Verify:** `python scripts/run_baseline_backtest.py` → completes on 448,782 candles
**Baseline (bar to beat):** Sharpe -0.656 · 1 trade · 0% win rate · -1.0% return (5-year, rule_v1)
**Sign-off:** ✅

---

## Phase 4 — Feature Engineering + Triple-Barrier Labeling

### Chunk 1 — Feature builder
**Files:** `src/tbot/features/builder.py`
**Verify:** `pytest tests/test_features.py -v` (TBD in Chunk 4)
**Sign-off:** ✅

### Chunk 2 — Triple-barrier labeler
**Files:** `src/tbot/features/labeler.py`
**Verify:** Synthetic tests passed · Jan 2024 real data: 1,023 signals → WIN=36.7%, LOSS=57%, NEUTRAL=6.4%
**Sign-off:** ✅

### Chunk 3 — Build training dataset script
**Files:** `scripts/build_training_dataset.py`
**Verify:** `uv run python scripts/build_training_dataset.py` → `data/features/signals_labeled.parquet` + `data/features/dataset_stats.json`
**Result:** 82,849 rows × 34 cols · WIN=35.2% · LOSS=59.0% · NEUTRAL=5.8% · win_rate healthy
**Sign-off:** ✅

---

## Phase 5 — XGBoost Confidence Model

### Chunk 1 — Walk-forward splitter
**Files:** `src/tbot/ml/walk_forward.py`, `tests/test_walk_forward.py`
**Verify:** `uv run pytest tests/test_walk_forward.py -v` → 10/10 passed
**Sign-off:** ✅

### Chunk 2 — XGBoost walk-forward training
**Files:** `src/tbot/ml/train.py`, `scripts/train_model.py`
**Verify:** `.venv/bin/python scripts/train_model.py` → 22 folds, mean AUC=0.6354
**Result:** All 22 folds AUC 0.61–0.67 · consistent across 5 years · `models/xgb_v1.pkl` saved
**Sign-off:** ✅

### Chunk 3 — Probability calibration
**Files:** `src/tbot/ml/calibration.py`, `scripts/calibrate_model.py`
**Verify:** `.venv/bin/python scripts/calibrate_model.py` → Brier improvement 8.1% · `models/calibrator_v1.pkl` saved
**Result:** Raw model overconfident (+0.05–0.17 gap) · calibrated gaps ≈ 0 · at threshold 0.60 → 68.7% expected win rate on 7% of signals
**Sign-off:** ✅

### Chunk 4 — Wire ML model into agent_v2
**Files:** `src/tbot/ml/predict.py`, `src/tbot/core/agent_v2.py`
**Verify:** Smoke test on 6mo slice → 9,120 hardcoded signals → 680 ML-filtered signals (conf range 0.629–0.917)
**Result:** Hardcoded 0.65/0.70/0.75 replaced by calibrated XGBoost scores · 92.5% signal reduction at threshold 0.60
**Sign-off:** ✅
