#!/usr/bin/env python3
"""
AI Trading Agent - Comprehensive Demo
=====================================

This demo script showcases the complete functionality of the AI trading agent
including pattern recognition, market structure analysis, and trading signals.

Author: AI Trading Agent
Version: 1.0
"""

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import our custom modules
from market_structure_analyzer import MarketStructureAnalyzer, TrendDirection
from visualizer import MarketStructureVisualizer, create_performance_report
from ai_trading_agent import AITradingAgent


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🎯 {title}")
    print("="*80)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n📊 {title}")
    print("-" * 60)


def demo_basic_functionality():
    """Demonstrate basic functionality of the trading agent"""
    print_header("BASIC FUNCTIONALITY DEMO")
    
    # Initialize the agent
    config = {
        'lookback_period': 500,
        'enable_visualization': False,  # Disable for console demo
        'save_reports': True,
        'analysis_batch_size': 50
    }
    
    agent = AITradingAgent(config)
    print("✅ AI Trading Agent initialized successfully!")
    
    # Load data
    data_file = "XAU_5m_data.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file {data_file} not found!")
        return False
    
    print(f"📈 Loading market data from {data_file}...")
    success = agent.load_market_data(data_file)
    
    if not success:
        print("❌ Failed to load market data!")
        return False
    
    print(f"✅ Successfully loaded {len(agent.analyzer.candles):,} candles")
    
    # Show data range
    first_candle = agent.analyzer.candles[0]
    last_candle = agent.analyzer.candles[-1]
    print(f"📅 Data range: {first_candle['timestamp']} to {last_candle['timestamp']}")
    
    return agent


def demo_market_structure_analysis(agent: AITradingAgent):
    """Demonstrate market structure analysis"""
    print_header("MARKET STRUCTURE ANALYSIS DEMO")
    
    # Perform initial analysis
    print_section("Initial Trend Detection")
    initial_results = agent.perform_initial_analysis()
    
    print(f"🎯 Initial Trend: {initial_results['initial_trend'].upper()}")
    print(f"📊 Analysis Period: {initial_results['analysis_period']} candles")
    print(f"🚀 Analysis Start Index: {initial_results['analysis_start_index']}")
    
    # Show recent price action
    print_section("Recent Price Action (Last 10 Candles)")
    recent_candles = agent.analyzer.candles[-10:]
    
    print("| Index | Timestamp           | Open     | High     | Low      | Close    | Type   |")
    print("|-------|---------------------|----------|----------|----------|----------|--------|")
    
    for i, candle in enumerate(recent_candles):
        candle_type = "🟢 Bull" if candle['close'] > candle['open'] else "🔴 Bear"
        idx = len(agent.analyzer.candles) - 10 + i
        print(f"| {idx:5d} | {candle['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
              f"{candle['open']:8.2f} | {candle['high']:8.2f} | {candle['low']:8.2f} | "
              f"{candle['close']:8.2f} | {candle_type} |")


def demo_pattern_recognition(agent: AITradingAgent):
    """Demonstrate pattern recognition capabilities"""
    print_header("PATTERN RECOGNITION DEMO")
    
    # Run analysis on recent data
    print_section("Running Pattern Analysis")
    
    # Analyze last 200 candles for demo
    start_index = max(0, len(agent.analyzer.candles) - 200)
    end_index = len(agent.analyzer.candles) - 1
    
    print(f"🔍 Analyzing candles {start_index} to {end_index} for patterns...")
    
    batch_results = agent.run_batch_analysis(start_index, end_index)
    
    print(f"✅ Analysis completed in {batch_results.get('duration', 0):.2f} seconds")
    print(f"🔍 Patterns found: {batch_results.get('patterns_found', 0)}")
    print(f"💥 BOS events: {batch_results.get('bos_events', 0)}")
    print(f"🔄 ChoCH events: {batch_results.get('choch_events', 0)}")
    print(f"🎯 Trading signals: {batch_results.get('signals_generated', 0)}")


def demo_trading_signals(agent: AITradingAgent):
    """Demonstrate trading signal generation"""
    print_header("TRADING SIGNALS DEMO")
    
    # Get latest signals
    latest_signals = agent.get_latest_signals(5)
    
    if not latest_signals:
        print("📭 No trading signals generated yet.")
        return
    
    print_section(f"Latest {len(latest_signals)} Trading Signals")
    
    for i, signal in enumerate(latest_signals, 1):
        print(f"\n🎯 Signal #{i}:")
        print(f"   Type: {signal['type']} ({'🟢 BUY' if signal['type'] == 'BUY' else '🔴 SELL'})")
        print(f"   Reason: {signal['reason']}")
        print(f"   Price: ${signal['price']:.2f}")
        print(f"   Confidence: {signal['confidence']:.1%}")
        print(f"   Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"   Take Profit: ${signal['take_profit']:.2f}")
        print(f"   Risk/Reward: {signal['risk_reward_ratio']:.1f}:1")
        print(f"   Timestamp: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")


def demo_market_structure_levels(agent: AITradingAgent):
    """Demonstrate market structure levels"""
    print_header("MARKET STRUCTURE LEVELS DEMO")
    
    state = agent.analyzer.get_current_state()
    
    print_section("Current Market State")
    print(f"🎯 Current Trend: {state['current_trend'].upper()}")
    print(f"⏰ Status: {'🔍 Waiting for BOS' if state['waiting_for_bos'] else '🔄 Searching for patterns'}")
    
    if state['active_levels']:
        print_section("Active Structure Levels")
        for level_name, level_data in state['active_levels'].items():
            level_type = ""
            if 'CH' in level_name:
                level_type = "🔴 Resistance"
            elif 'CL' in level_name:
                level_type = "🟢 Support"
            
            print(f"   {level_name}: ${level_data['price']:.2f} ({level_type})")
            print(f"   └─ Created: {level_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("📭 No active structure levels currently.")
    
    # Show all historical levels
    print_section("Historical Structure Events")
    
    if agent.analyzer.events:
        recent_events = agent.analyzer.events[-5:]  # Last 5 events
        
        for i, event in enumerate(recent_events, 1):
            event_icon = ""
            if event.event_type.value == "bos_bullish":
                event_icon = "🚀"
            elif event.event_type.value == "bos_bearish":
                event_icon = "🔻"
            elif event.event_type.value == "choch_bullish":
                event_icon = "🔄🟢"
            elif event.event_type.value == "choch_bearish":
                event_icon = "🔄🔴"
            
            print(f"   {event_icon} {event.event_type.value.replace('_', ' ').title()}")
            print(f"   └─ Price: ${event.price:.2f} at {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   └─ {event.description}")
    else:
        print("📭 No historical events recorded yet.")


def demo_performance_metrics(agent: AITradingAgent):
    """Demonstrate performance metrics and reporting"""
    print_header("PERFORMANCE METRICS DEMO")
    
    # Generate comprehensive report
    print_section("Generating Performance Report")
    report = agent.generate_comprehensive_report(save_to_file=True)
    
    # Display key metrics
    perf_analysis = report['performance_analysis']
    signal_summary = report['trading_signals']
    
    print_section("Analysis Summary")
    print(f"📊 Total Candles Analyzed: {perf_analysis['analysis_summary']['total_candles_analyzed']:,}")
    print(f"🎯 Current Trend: {perf_analysis['analysis_summary']['current_trend'].upper()}")
    print(f"⏱️  Current Trend Duration: {perf_analysis['analysis_summary']['trend_duration']} candles")
    
    print_section("Event Statistics")
    print(f"📊 Total Events: {perf_analysis['event_statistics']['total_events']}")
    print(f"💥 BOS Events: {perf_analysis['event_statistics']['bos_events']}")
    print(f"🔄 ChoCH Events: {perf_analysis['event_statistics']['choch_events']}")
    print(f"📈 Event Frequency: {perf_analysis['event_statistics']['event_frequency_percent']:.4f}%")
    
    print_section("Trading Signal Summary")
    print(f"🎯 Total Signals: {signal_summary['total_signals']}")
    print(f"🟢 Buy Signals: {signal_summary['buy_signals']}")
    print(f"🔴 Sell Signals: {signal_summary['sell_signals']}")
    if signal_summary['total_signals'] > 0:
        print(f"⭐ Average Confidence: {signal_summary['average_confidence']:.1%}")
        print(f"🏆 High Confidence Signals: {signal_summary['high_confidence_signals']}")
    
    print_section("Trend Analysis")
    trend_analysis = perf_analysis['trend_analysis']
    print(f"📈 Bullish Periods: {trend_analysis['bullish_periods']}")
    print(f"📉 Bearish Periods: {trend_analysis['bearish_periods']}")
    print(f"🎯 Trend Consistency: {trend_analysis['trend_consistency']}")


def demo_alerts_and_notifications(agent: AITradingAgent):
    """Demonstrate alerts and notification system"""
    print_header("ALERTS & NOTIFICATIONS DEMO")
    
    # Show recent alerts
    recent_alerts = agent.analyzer.alerts[-10:] if agent.analyzer.alerts else []
    
    if recent_alerts:
        print_section(f"Recent Alerts (Last {len(recent_alerts)})")
        
        for i, alert in enumerate(recent_alerts, 1):
            print(f"   🚨 Alert #{i}: {alert}")
    else:
        print("📭 No alerts generated yet.")
    
    print_section("Alert System Features")
    print("✅ Real-time pattern detection alerts")
    print("✅ BOS (Break of Structure) confirmation alerts")
    print("✅ ChoCH (Change of Character) alerts")
    print("✅ Trading signal generation alerts")
    print("✅ Risk management notifications")


def demo_strategy_explanation():
    """Explain the trading strategy being implemented"""
    print_header("TRADING STRATEGY EXPLANATION")
    
    print_section("Core Strategy Components")
    print("""
🎯 TWO-CANDLE RETRACEMENT PATTERN STRATEGY

This AI trading agent implements a sophisticated market structure analysis strategy 
based on two-candle retracement patterns and institutional trading concepts.

📊 KEY CONCEPTS:

1. INITIAL TREND DETECTION
   • Analyzes last 1000 candles to find absolute highest/lowest points
   • Determines trend direction based on chronological order
   • Sets foundation for subsequent pattern analysis

2. THREE-CANDLE RETRACEMENT PATTERNS
   
   Bullish Pattern (in bullish trend):
   • Candle 1: Green (bullish impulse)
   • Candle 2: Red (correction begins)
   • Candle 3: Red (correction continues) AND close < Candle 2's low
   → Creates CH1 (Resistance level)
   
   Bearish Pattern (in bearish trend):
   • Candle 1: Red (bearish impulse)
   • Candle 2: Green (correction begins)
   • Candle 3: Green (correction continues) AND close > Candle 2's high
   → Creates CL1 (Support level)

3. BREAK OF STRUCTURE (BOS)
   • Bullish BOS: Green candle closes above CH1
   • Bearish BOS: Red candle closes below CL1
   • Creates secondary levels (CL2/CH2) for continued analysis

4. CHANGE OF CHARACTER (ChoCH)
   • Bullish ChoCH: Two consecutive closes above CH2 (trend reversal)
   • Bearish ChoCH: Two consecutive closes below CL2 (trend reversal)
   • Signals major trend changes

🎯 TRADING SIGNALS:
   • BOS confirmations generate entry signals
   • ChoCH events generate high-confidence trend reversal signals
   • Risk management with automatic stop-loss and take-profit levels
   • 2:1 risk-reward ratio for optimal trade management
""")


def main():
    """Main demo function"""
    print_header("AI TRADING AGENT - COMPREHENSIVE DEMO")
    print("Welcome to the AI Trading Agent demonstration!")
    print("This demo will showcase all features of the sophisticated market structure analysis system.")
    
    # Run demos in sequence
    try:
        # 1. Basic functionality
        agent = demo_basic_functionality()
        if not agent:
            print("❌ Demo failed during initialization!")
            return
        
        # 2. Strategy explanation
        demo_strategy_explanation()
        
        # 3. Market structure analysis
        demo_market_structure_analysis(agent)
        
        # 4. Pattern recognition
        demo_pattern_recognition(agent)
        
        # 5. Market structure levels
        demo_market_structure_levels(agent)
        
        # 6. Trading signals
        demo_trading_signals(agent)
        
        # 7. Performance metrics
        demo_performance_metrics(agent)
        
        # 8. Alerts and notifications
        demo_alerts_and_notifications(agent)
        
        # Final summary
        print_header("DEMO COMPLETED SUCCESSFULLY")
        agent.print_summary()
        
        print("\n🎉 CONGRATULATIONS!")
        print("You now have a fully functional AI trading agent with:")
        print("✅ Sophisticated market structure analysis")
        print("✅ Advanced pattern recognition")
        print("✅ Automated trading signal generation")
        print("✅ Comprehensive risk management")
        print("✅ Real-time alerts and notifications")
        print("✅ Performance tracking and reporting")
        
        print(f"\n📊 USAGE INSTRUCTIONS:")
        print(f"1. Run full analysis: python ai_trading_agent.py --data XAU_5m_data.csv --mode batch")
        print(f"2. Real-time mode: python ai_trading_agent.py --data XAU_5m_data.csv --mode realtime")
        print(f"3. Custom config: python ai_trading_agent.py --data XAU_5m_data.csv --config config.json")
        print(f"4. Visual reports: python ai_trading_agent.py --data XAU_5m_data.csv --mode both")
        
        print(f"\n📁 Check the './reports' directory for detailed analysis reports and charts!")
        
    except Exception as e:
        print(f"❌ Demo encountered an error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()