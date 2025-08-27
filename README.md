# AI Trading Agent - Sophisticated Market Structure Analysis

🤖 **A comprehensive Python-based AI trading agent that implements sophisticated market structure analysis using two-candle retracement patterns, Break of Structure (BOS), and Change of Character (ChoCH) concepts.**

## 🎯 Overview

This AI trading agent is designed for professional traders who want to implement institutional-level market structure analysis. The system identifies high-probability trading opportunities by analyzing candlestick patterns, market structure breaks, and trend changes using advanced algorithmic techniques.

### ✨ Key Features

- **📊 Advanced Market Structure Analysis**: Identifies key support/resistance levels using sophisticated pattern recognition
- **🎯 Two-Candle Retracement Patterns**: Detects institutional-level entry patterns with high accuracy
- **💥 Break of Structure (BOS) Detection**: Automatically identifies trend continuation signals
- **🔄 Change of Character (ChoCH) Analysis**: Detects major trend reversals with high confidence
- **🚨 Real-time Alert System**: Instant notifications for trading opportunities
- **📈 Comprehensive Visualization**: Professional-grade charts with structure levels and signals
- **📊 Performance Tracking**: Detailed analytics and reporting capabilities
- **⚙️ Risk Management**: Built-in stop-loss and take-profit calculations
- **🔧 Highly Configurable**: Customizable parameters for different trading styles

## 🏗️ Architecture

The system consists of four main components:

### 1. Market Structure Analyzer (`market_structure_analyzer.py`)
- **Core analysis engine**
- Initial trend detection using 1000-candle lookback
- Three-candle retracement pattern recognition
- BOS and ChoCH event detection
- Structure level management (CH1, CL1, CH2, CL2)

### 2. Visualizer (`visualizer.py`)
- **Professional charting system**
- Candlestick charts with structure overlays
- Real-time performance dashboards
- Event timeline visualization
- Alert and notification displays

### 3. AI Trading Agent (`ai_trading_agent.py`)
- **Main application orchestrator**
- Trading signal generation
- Risk management calculations
- Performance reporting
- Configuration management

### 4. Test Suite (`test_trading_agent.py`)
- **Comprehensive validation**
- Unit tests for all components
- Integration testing
- Performance benchmarking

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run comprehensive analysis on your data
python ai_trading_agent.py --data XAU_5m_data.csv --mode batch

# Real-time analysis mode
python ai_trading_agent.py --data XAU_5m_data.csv --mode realtime

# Full analysis with visualizations
python ai_trading_agent.py --data XAU_5m_data.csv --mode both

# Custom configuration
python ai_trading_agent.py --data XAU_5m_data.csv --config config.json
```

### Demo Mode

```bash
# Run comprehensive demonstration
python demo_trading_agent.py
```

## 📊 Trading Strategy

### Core Methodology

The AI trading agent implements a sophisticated market structure analysis strategy based on institutional trading concepts:

#### 1. **Initial Trend Detection**
- Analyzes the last 1000 candles to find absolute highest and lowest points
- Determines initial trend direction based on chronological order:
  - **Bullish**: If lowest point occurs before highest point
  - **Bearish**: If highest point occurs before lowest point

#### 2. **Three-Candle Retracement Patterns**

**Bullish Pattern (in bullish trend):**
```
Candle 1: 🟢 Green (bullish impulse)
Candle 2: 🔴 Red (correction begins)
Candle 3: 🔴 Red (correction continues) AND close < Candle 2's low
→ Creates CH1 (Confirm High Level 1) - Resistance
```

**Bearish Pattern (in bearish trend):**
```
Candle 1: 🔴 Red (bearish impulse)
Candle 2: 🟢 Green (correction begins)
Candle 3: 🟢 Green (correction continues) AND close > Candle 2's high
→ Creates CL1 (Confirm Low Level 1) - Support
```

#### 3. **Break of Structure (BOS)**

**Bullish BOS:**
- Wait for green candle to close above CH1
- Creates CL2 (lowest point between CH1 and BOS candle)
- Generates **BUY signal**

**Bearish BOS:**
- Wait for red candle to close below CL1
- Creates CH2 (highest point between CL1 and BOS candle)
- Generates **SELL signal**

#### 4. **Change of Character (ChoCH)**

**Bullish ChoCH:**
- Two consecutive candle closes above CH2
- Trend changes from bearish to bullish
- Generates high-confidence **BUY signal**

**Bearish ChoCH:**
- Two consecutive candle closes below CL2
- Trend changes from bullish to bearish
- Generates high-confidence **SELL signal**

## 📈 Example Results

### Recent Analysis on XAU (Gold) 5-Minute Data

```
🎯 ANALYSIS RESULTS:
📊 Current Trend: BEARISH
📈 Total Candles: 1,411,862
🔍 Patterns found: 7
💥 BOS events: 6
🔄 ChoCH events: 0
🎯 Trading signals: 6

📍 ACTIVE LEVELS:
   CL1: $3,282.64 (Support)
   CH2: $3,297.76 (Resistance)

🎯 LATEST SIGNAL: SELL - Bearish BOS Confirmed
   Price: $3,284.12 | Confidence: 80.0%
   Stop Loss: $3,314.55 | Take Profit: $3,225.37
   Risk/Reward: 2.0:1
```

## ⚙️ Configuration

The system is highly configurable through the `config.json` file:

```json
{
  "lookback_period": 1000,
  "enable_alerts": true,
  "enable_visualization": true,
  "save_reports": true,
  "risk_management": {
    "position_size_percent": 2.0,
    "stop_loss_percent": 1.0,
    "take_profit_percent": 2.0
  }
}
```

### Key Parameters

- **`lookback_period`**: Number of candles for initial trend analysis (default: 1000)
- **`analysis_batch_size`**: Candles processed per batch (default: 100)
- **`stop_loss_percent`**: Stop loss as percentage of entry price (default: 1.0%)
- **`take_profit_percent`**: Take profit as percentage of entry price (default: 2.0%)

## 📁 File Structure

```
TBot/
├── market_structure_analyzer.py    # Core analysis engine
├── visualizer.py                   # Visualization components
├── ai_trading_agent.py            # Main application
├── demo_trading_agent.py          # Comprehensive demo
├── test_trading_agent.py          # Test suite
├── config.json                    # Configuration file
├── requirements.txt               # Dependencies
├── XAU_5m_data.csv               # Sample data (Gold 5-min)
└── reports/                       # Generated reports and charts
    ├── trading_report_*.json      # Analysis reports
    ├── market_analysis_*.png      # Chart images
    └── alerts.log                 # Alert logs
```

## 📊 Data Format

The system expects CSV data with the following columns:

```csv
Date,Open,High,Low,Close,Volume
2024-01-01 09:00,2000.50,2005.25,1998.75,2003.20,1250
2024-01-01 09:05,2003.20,2008.10,2001.50,2006.80,980
...
```

## 🎨 Visualization Features

The system generates professional-grade visualizations including:

- **📈 Candlestick Charts**: With structure levels and BOS lines
- **📊 Performance Dashboards**: Real-time statistics and metrics
- **⏰ Event Timelines**: Chronological view of BOS and ChoCH events
- **🚨 Alert Panels**: Recent notifications and signals
- **📈 Live Charts**: Real-time updating for active trading

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_trading_agent.py

# Basic validation only
python -c "from test_trading_agent import run_basic_validation; run_basic_validation()"
```

### Test Coverage

- ✅ Market structure analyzer components
- ✅ Pattern recognition algorithms
- ✅ Signal generation logic
- ✅ Risk management calculations
- ✅ Integration testing
- ✅ Performance benchmarking

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Analysis Statistics
- **Event Frequency**: Percentage of candles generating events
- **Pattern Recognition Accuracy**: Success rate of pattern detection
- **Signal Quality**: Confidence levels and success rates
- **Trend Analysis**: Duration and consistency of trends

### Trading Performance
- **Signal Generation Rate**: Signals per time period
- **Risk/Reward Ratios**: Automatic calculation for all signals
- **Drawdown Analysis**: Maximum adverse movement tracking
- **Win Rate Tracking**: Success rate of generated signals

## 🔧 Advanced Usage

### Custom Pattern Recognition

```python
from market_structure_analyzer import MarketStructureAnalyzer

# Initialize with custom parameters
analyzer = MarketStructureAnalyzer(lookback_period=500)

# Load your data
analyzer.load_data("your_data.csv")

# Determine initial trend
initial_trend = analyzer.determine_initial_trend()

# Analyze specific candle
result = analyzer.analyze_candle(candle_index)
```

### Real-time Integration

```python
from ai_trading_agent import AITradingAgent

# Initialize for real-time trading
config = {
    'real_time_mode': True,
    'enable_alerts': True,
    'risk_management': {
        'position_size_percent': 1.0,
        'stop_loss_percent': 0.5
    }
}

agent = AITradingAgent(config)

# Process new candle data
latest_signals = agent.get_latest_signals(1)
```

## 🚨 Risk Management

### Built-in Safety Features

- **📊 Position Sizing**: Automatic calculation based on account percentage
- **🛡️ Stop Loss**: Automatic stop loss placement for all signals
- **🎯 Take Profit**: 2:1 risk/reward ratio enforcement
- **⚠️ Maximum Drawdown**: Configurable loss limits
- **🔄 Trend Confirmation**: Multiple confirmation requirements

### Risk Parameters

```json
{
  "risk_management": {
    "max_consecutive_losses": 3,
    "position_size_percent": 2.0,
    "stop_loss_percent": 1.0,
    "take_profit_percent": 2.0
  }
}
```

## 📞 Support & Contributing

### Getting Help

1. Check the demo script: `python demo_trading_agent.py`
2. Review the test suite for examples
3. Examine the configuration options in `config.json`

### Contributing

This is a complete, production-ready trading system. For enhancements:

1. Fork the repository
2. Create feature branches
3. Add comprehensive tests
4. Update documentation
5. Submit pull requests

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes. Trading financial instruments involves risk, and past performance does not guarantee future results. Always:

- 📊 Backtest strategies thoroughly
- 🎯 Use proper risk management
- 💰 Never risk more than you can afford to lose
- 📚 Understand the markets you're trading
- 🧪 Test in demo environments first

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

🎉 **Congratulations!** You now have a sophisticated AI trading agent capable of institutional-level market structure analysis. Happy trading! 🚀