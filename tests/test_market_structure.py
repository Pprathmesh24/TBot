import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from tbot.core.market_structure import (
    MarketStructureAnalyzer,
    TrendDirection,
    EventType,
    StructureLevel,
)
from tbot.core.agent import AITradingAgent


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return MarketStructureAnalyzer(lookback_period=100)


@pytest.fixture
def ohlcv_csv(tmp_path):
    """Write a 200-row OHLCV CSV and return its path."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=200, freq="5min")
    price = 2000.0
    rows = []
    for i, date in enumerate(dates):
        price += rng.normal(0, 2.0) + (0.3 if 50 <= i < 150 else -0.5 if i < 50 else 0)
        o = price + rng.normal(0, 0.5)
        c = price + rng.normal(0, 0.5)
        rows.append({
            "Date": date, "Open": o,
            "High": max(o, c) + abs(rng.normal(0, 1)),
            "Low":  min(o, c) - abs(rng.normal(0, 1)),
            "Close": c, "Volume": int(rng.integers(100, 1000)),
        })
    path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def agent():
    return AITradingAgent({
        "lookback_period": 50,
        "enable_alerts": True,
        "enable_visualization": False,
        "save_reports": False,
        "analysis_batch_size": 10,
    })


# ---------------------------------------------------------------------------
# MarketStructureAnalyzer
# ---------------------------------------------------------------------------

class TestMarketStructureAnalyzer:

    def test_initialization(self, analyzer):
        assert analyzer.lookback_period == 100
        assert analyzer.current_trend == TrendDirection.NEUTRAL
        assert len(analyzer.candles) == 0
        assert len(analyzer.events) == 0

    def test_data_loading(self, analyzer, ohlcv_csv):
        assert analyzer.load_data(ohlcv_csv)
        assert len(analyzer.candles) == 200
        first = analyzer.candles[0]
        for key in ("open", "high", "low", "close", "timestamp"):
            assert key in first

    def test_initial_trend_detection(self, analyzer, ohlcv_csv):
        analyzer.load_data(ohlcv_csv)
        trend = analyzer.determine_initial_trend()
        assert trend in (TrendDirection.BULLISH, TrendDirection.BEARISH)
        assert analyzer.current_trend == trend

    def test_candle_classification(self, analyzer):
        bull = {"open": 100, "close": 105, "high": 106, "low": 99}
        bear = {"open": 105, "close": 100, "high": 106, "low": 99}
        assert analyzer.is_bullish_candle(bull)
        assert not analyzer.is_bearish_candle(bull)
        assert analyzer.is_bearish_candle(bear)
        assert not analyzer.is_bullish_candle(bear)

    def test_three_candle_bullish_pattern(self, analyzer):
        analyzer.candles = [
            {"open": 100, "close": 105, "high": 106, "low": 99},   # green
            {"open": 105, "close": 102, "high": 106, "low": 101},  # red
            {"open": 102, "close": 100, "high": 103, "low": 99},   # red, close < c2 low
        ]
        pattern = analyzer.find_three_candle_pattern_bullish(0)
        assert pattern == (0, 1, 2)

    def test_structure_level_dataclass(self, analyzer, ohlcv_csv):
        analyzer.load_data(ohlcv_csv)
        level = StructureLevel(price=2000.0, timestamp=datetime.now(),
                               level_type="CH1", candle_index=50)
        assert level.price == 2000.0
        assert level.level_type == "CH1"
        assert not level.is_broken

    def test_get_current_state_keys(self, analyzer):
        state = analyzer.get_current_state()
        for key in ("current_trend", "trend_start_index", "active_levels",
                    "total_events", "total_alerts", "waiting_for_bos",
                    "statistics", "data_loaded", "total_candles"):
            assert key in state


# ---------------------------------------------------------------------------
# AITradingAgent
# ---------------------------------------------------------------------------

class TestAITradingAgent:

    def test_initialization(self, agent):
        assert agent.analyzer is not None
        assert agent.config["lookback_period"] == 50
        assert len(agent.current_signals) == 0

    def test_custom_config(self):
        a = AITradingAgent({"lookback_period": 200, "enable_alerts": False})
        assert a.config["lookback_period"] == 200
        assert not a.config["enable_alerts"]

    def test_data_loading(self, agent, ohlcv_csv):
        assert agent.load_market_data(ohlcv_csv)
        assert len(agent.analyzer.candles) == 200

    def test_initial_analysis_keys(self, agent, ohlcv_csv):
        agent.load_market_data(ohlcv_csv)
        results = agent.perform_initial_analysis()
        for key in ("initial_trend", "total_candles", "analysis_period"):
            assert key in results

    def test_stop_loss_direction(self, agent):
        entry = 2000.0
        assert agent._calculate_stop_loss(entry, "BUY") < entry
        assert agent._calculate_stop_loss(entry, "SELL") > entry

    def test_take_profit_direction(self, agent):
        entry = 2000.0
        assert agent._calculate_take_profit(entry, "BUY") > entry
        assert agent._calculate_take_profit(entry, "SELL") < entry


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_workflow(self, ohlcv_csv):
        agent = AITradingAgent({
            "lookback_period": 100,
            "enable_visualization": False,
            "save_reports": False,
            "analysis_batch_size": 20,
        })
        assert agent.load_market_data(ohlcv_csv)
        assert isinstance(agent.perform_initial_analysis(), dict)
        batch = agent.run_batch_analysis()
        assert isinstance(batch, dict)
        report = agent.generate_comprehensive_report(save_to_file=False)
        assert "performance_analysis" in report
        assert "trading_signals" in report
