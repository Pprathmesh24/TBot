#!/usr/bin/env python3
"""
AI Trading Agent - Example Usage
===============================

This script demonstrates practical usage of the AI trading agent
for real trading scenarios.

Author: AI Trading Agent
Version: 1.0
"""

from ai_trading_agent import AITradingAgent
from datetime import datetime
import json

def example_basic_analysis():
    """Example: Basic market structure analysis"""
    print("🔍 Example 1: Basic Market Structure Analysis")
    print("-" * 50)
    
    # Initialize with conservative settings
    config = {
        'lookback_period': 500,
        'enable_alerts': True,
        'enable_visualization': False,
        'save_reports': True
    }
    
    agent = AITradingAgent(config)
    
    # Load data and analyze
    if agent.load_market_data("XAU_5m_data.csv"):
        # Perform analysis
        initial_results = agent.perform_initial_analysis()
        batch_results = agent.run_batch_analysis()
        
        # Get latest signals
        signals = agent.get_latest_signals(3)
        
        print(f"✅ Analysis completed!")
        print(f"🎯 Initial trend: {initial_results['initial_trend']}")
        print(f"📊 Patterns found: {batch_results.get('patterns_found', 0)}")
        print(f"🎯 Latest signals: {len(signals)}")
        
        if signals:
            latest = signals[-1]
            print(f"🚨 Latest signal: {latest['type']} at ${latest['price']:.2f}")
        
        return agent
    return None

def example_custom_risk_management():
    """Example: Custom risk management settings"""
    print("\n💰 Example 2: Custom Risk Management")
    print("-" * 50)
    
    # Conservative risk settings
    config = {
        'lookback_period': 1000,
        'risk_management': {
            'position_size_percent': 1.0,  # 1% position size
            'stop_loss_percent': 0.5,      # 0.5% stop loss
            'take_profit_percent': 1.5,    # 1.5% take profit (3:1 RR)
            'max_consecutive_losses': 2
        },
        'enable_visualization': False,
        'save_reports': True
    }
    
    agent = AITradingAgent(config)
    
    if agent.load_market_data("XAU_5m_data.csv"):
        agent.perform_initial_analysis()
        batch_results = agent.run_batch_analysis()
        
        signals = agent.get_latest_signals(1)
        if signals:
            signal = signals[0]
            print(f"🎯 Signal: {signal['type']}")
            print(f"💰 Entry: ${signal['price']:.2f}")
            print(f"🛡️  Stop Loss: ${signal['stop_loss']:.2f}")
            print(f"🎯 Take Profit: ${signal['take_profit']:.2f}")
            print(f"📊 Risk/Reward: {signal['risk_reward_ratio']:.1f}:1")
    
    return agent

def example_real_time_monitoring():
    """Example: Real-time monitoring setup"""
    print("\n⏰ Example 3: Real-time Monitoring Setup")
    print("-" * 50)
    
    config = {
        'lookback_period': 200,
        'analysis_batch_size': 10,
        'enable_alerts': True,
        'real_time_mode': True,
        'alerts': {
            'enable_console': True,
            'enable_file_logging': True
        }
    }
    
    agent = AITradingAgent(config)
    
    if agent.load_market_data("XAU_5m_data.csv"):
        print("🔴 Setting up real-time monitoring...")
        print("📊 Analyzing recent market structure...")
        
        # Analyze recent data
        total_candles = len(agent.analyzer.candles)
        start_index = max(0, total_candles - 100)  # Last 100 candles
        
        batch_results = agent.run_batch_analysis(start_index)
        
        state = agent.analyzer.get_current_state()
        print(f"🎯 Current trend: {state['current_trend']}")
        print(f"⚠️  Active alerts: {len(agent.analyzer.alerts)}")
        
        if state['active_levels']:
            print("📍 Watching levels:")
            for level, data in state['active_levels'].items():
                print(f"   {level}: ${data['price']:.2f}")
    
    return agent

def example_performance_analysis():
    """Example: Performance analysis and reporting"""
    print("\n📊 Example 4: Performance Analysis")
    print("-" * 50)
    
    config = {
        'lookback_period': 1000,
        'save_reports': True,
        'report_directory': './reports'
    }
    
    agent = AITradingAgent(config)
    
    if agent.load_market_data("XAU_5m_data.csv"):
        # Run comprehensive analysis
        agent.perform_initial_analysis()
        batch_results = agent.run_batch_analysis()
        
        # Generate detailed report
        report = agent.generate_comprehensive_report()
        
        # Extract key metrics
        perf = report['performance_analysis']
        signals = report['trading_signals']
        
        print("📈 PERFORMANCE SUMMARY")
        print(f"📊 Total events: {perf['event_statistics']['total_events']}")
        print(f"🎯 Trading signals: {signals['total_signals']}")
        print(f"📈 Buy signals: {signals['buy_signals']}")
        print(f"📉 Sell signals: {signals['sell_signals']}")
        
        if signals['total_signals'] > 0:
            print(f"⭐ Avg confidence: {signals['average_confidence']:.1%}")
        
        print(f"🎯 Trend consistency: {perf['trend_analysis']['trend_consistency']}")
    
    return agent

def example_strategy_backtest():
    """Example: Simple strategy backtesting"""
    print("\n🧪 Example 5: Strategy Backtesting")
    print("-" * 50)
    
    config = {
        'lookback_period': 500,
        'enable_visualization': False,
        'save_reports': False
    }
    
    agent = AITradingAgent(config)
    
    if agent.load_market_data("XAU_5m_data.csv"):
        # Analyze a specific period
        total_candles = len(agent.analyzer.candles)
        
        # Test on last 1000 candles
        start_index = max(0, total_candles - 1000)
        
        print(f"🔍 Backtesting on {total_candles - start_index} candles...")
        
        agent.perform_initial_analysis()
        batch_results = agent.run_batch_analysis(start_index)
        
        # Calculate simple metrics
        signals = agent.current_signals
        
        if signals:
            buy_signals = [s for s in signals if s['type'] == 'BUY']
            sell_signals = [s for s in signals if s['type'] == 'SELL']
            
            avg_confidence = sum(s['confidence'] for s in signals) / len(signals)
            high_confidence = [s for s in signals if s['confidence'] >= 0.8]
            
            print(f"📊 BACKTEST RESULTS:")
            print(f"🎯 Total signals: {len(signals)}")
            print(f"📈 Buy signals: {len(buy_signals)}")
            print(f"📉 Sell signals: {len(sell_signals)}")
            print(f"⭐ Average confidence: {avg_confidence:.1%}")
            print(f"🏆 High confidence signals: {len(high_confidence)}")
            print(f"📈 Signal frequency: {len(signals)/(total_candles-start_index)*100:.3f}%")
        else:
            print("📭 No signals generated in backtest period")
    
    return agent

def main():
    """Run all examples"""
    print("🤖 AI TRADING AGENT - PRACTICAL EXAMPLES")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_analysis()
        example_custom_risk_management()
        example_real_time_monitoring()
        example_performance_analysis()
        example_strategy_backtest()
        
        print("\n🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("\n📚 NEXT STEPS:")
        print("1. Customize config.json for your trading style")
        print("2. Run: python ai_trading_agent.py --data your_data.csv --mode batch")
        print("3. Use --mode realtime for live monitoring")
        print("4. Check ./reports/ for detailed analysis")
        
    except Exception as e:
        print(f"❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()