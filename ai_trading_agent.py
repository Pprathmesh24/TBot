#!/usr/bin/env python3
"""
AI Trading Agent - Main Application
===================================

Complete AI trading agent that implements sophisticated market structure analysis
based on two-candle retracement patterns, BOS, and ChoCH concepts.

Features:
- Real-time market structure analysis
- Pattern recognition and signal generation
- Comprehensive visualization
- Performance tracking and reporting
- Alert system for trading opportunities

Author: AI Trading Agent
Version: 1.0
"""

import os
import sys
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Import our custom modules
from market_structure_analyzer import MarketStructureAnalyzer, TrendDirection, EventType
from visualizer import MarketStructureVisualizer, create_performance_report


class AITradingAgent:
    """
    Main AI Trading Agent class that orchestrates market structure analysis
    and provides trading signals based on sophisticated pattern recognition.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AI Trading Agent
        
        Args:
            config (Optional[Dict]): Configuration dictionary for customization
        """
        # Default configuration
        self.config = {
            'lookback_period': 1000,
            'min_pattern_separation': 3,
            'enable_alerts': True,
            'enable_visualization': True,
            'save_reports': True,
            'report_directory': './reports',
            'analysis_batch_size': 100,
            'real_time_mode': False,
            'risk_management': {
                'max_consecutive_losses': 3,
                'position_size_percent': 2.0,
                'stop_loss_percent': 1.0,
                'take_profit_percent': 2.0
            }
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        # Initialize core components
        self.analyzer = MarketStructureAnalyzer(self.config['lookback_period'])
        self.visualizer = None
        
        # Trading state
        self.current_signals = []
        self.trade_history = []
        self.performance_metrics = {}
        
        # Ensure report directory exists
        if self.config['save_reports']:
            os.makedirs(self.config['report_directory'], exist_ok=True)
        
        print("🤖 AI Trading Agent initialized successfully!")
        print(f"📊 Lookback period: {self.config['lookback_period']} candles")
        print(f"🎯 Analysis mode: {'Real-time' if self.config['real_time_mode'] else 'Batch'}")

    def load_market_data(self, file_path: str) -> bool:
        """
        Load market data from CSV file
        
        Args:
            file_path (str): Path to the CSV file containing OHLCV data
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        print(f"📈 Loading market data from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found at {file_path}")
            return False
        
        success = self.analyzer.load_data(file_path)
        
        if success:
            # Initialize visualizer after data is loaded
            if self.config['enable_visualization']:
                self.visualizer = MarketStructureVisualizer(self.analyzer)
            
            print("✅ Market data loaded successfully!")
            return True
        else:
            print("❌ Failed to load market data!")
            return False

    def perform_initial_analysis(self) -> Dict:
        """
        Perform initial market structure analysis to determine trend direction
        
        Returns:
            Dict: Initial analysis results
        """
        print("\\n🔍 Performing initial market structure analysis...")
        
        # Determine initial trend
        initial_trend = self.analyzer.determine_initial_trend()
        
        # Get current state
        state = self.analyzer.get_current_state()
        
        results = {
            'initial_trend': initial_trend.value,
            'analysis_start_index': self.analyzer.trend_start_index,
            'total_candles': len(self.analyzer.candles),
            'analysis_period': min(self.config['lookback_period'], len(self.analyzer.candles)),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"🎯 Initial trend determined: {initial_trend.value.upper()}")
        print(f"📊 Analysis period: {results['analysis_period']} candles")
        print(f"🚀 Starting analysis from index: {results['analysis_start_index']}")
        
        return results

    def run_batch_analysis(self, start_index: Optional[int] = None, 
                          end_index: Optional[int] = None) -> Dict:
        """
        Run comprehensive batch analysis on historical data
        
        Args:
            start_index (Optional[int]): Starting candle index for analysis
            end_index (Optional[int]): Ending candle index for analysis
            
        Returns:
            Dict: Comprehensive analysis results
        """
        print("\\n🚀 Starting batch analysis...")
        
        if not self.analyzer.candles:
            print("❌ No data loaded for analysis!")
            return {}
        
        # Set analysis range
        if start_index is None:
            start_index = max(0, self.analyzer.trend_start_index)
        if end_index is None:
            end_index = len(self.analyzer.candles) - 1
        
        print(f"📊 Analyzing candles {start_index} to {end_index} ({end_index - start_index + 1} candles)")
        
        # Track analysis progress
        analysis_results = {
            'patterns_found': 0,
            'bos_events': 0,
            'choch_events': 0,
            'signals_generated': 0,
            'analysis_range': (start_index, end_index),
            'start_time': datetime.now()
        }
        
        # Process candles in batches for better performance
        batch_size = self.config['analysis_batch_size']
        total_candles = end_index - start_index + 1
        processed = 0
        
        for batch_start in range(start_index, end_index + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_index)
            
            # Analyze each candle in the batch
            for candle_index in range(batch_start, batch_end + 1):
                result = self.analyzer.analyze_candle(candle_index)
                
                # Update counters
                if result.get('pattern_found'):
                    analysis_results['patterns_found'] += 1
                if result.get('bos_detected'):
                    analysis_results['bos_events'] += 1
                if result.get('choch_detected'):
                    analysis_results['choch_events'] += 1
                
                # Generate trading signals
                signals = self._generate_trading_signals(result)
                if signals:
                    analysis_results['signals_generated'] += len(signals)
                    self.current_signals.extend(signals)
                
                processed += 1
            
            # Show progress
            progress = (processed / total_candles) * 100
            if processed % (batch_size * 2) == 0:  # Update every 2 batches
                print(f"📈 Progress: {progress:.1f}% ({processed}/{total_candles} candles)")
        
        analysis_results['end_time'] = datetime.now()
        analysis_results['duration'] = (analysis_results['end_time'] - analysis_results['start_time']).total_seconds()
        
        print(f"\\n✅ Batch analysis completed!")
        print(f"⏱️  Duration: {analysis_results['duration']:.2f} seconds")
        print(f"🔍 Patterns found: {analysis_results['patterns_found']}")
        print(f"💥 BOS events: {analysis_results['bos_events']}")
        print(f"🔄 ChoCH events: {analysis_results['choch_events']}")
        print(f"🎯 Trading signals: {analysis_results['signals_generated']}")
        
        return analysis_results

    def _generate_trading_signals(self, analysis_result: Dict) -> List[Dict]:
        """
        Generate trading signals based on analysis results
        
        Args:
            analysis_result (Dict): Results from candle analysis
            
        Returns:
            List[Dict]: List of trading signals
        """
        signals = []
        
        if not analysis_result or 'candle_index' not in analysis_result:
            return signals
        
        candle_index = analysis_result['candle_index']
        candle = self.analyzer.candles[candle_index]
        
        # Signal generation for BOS events
        if analysis_result.get('bos_detected'):
            if 'bullish_bos' in analysis_result.get('events', []):
                signal = {
                    'type': 'BUY',
                    'reason': 'Bullish BOS Confirmed',
                    'timestamp': candle['timestamp'],
                    'price': candle['close'],
                    'candle_index': candle_index,
                    'confidence': 0.8,
                    'stop_loss': self._calculate_stop_loss(candle['close'], 'BUY'),
                    'take_profit': self._calculate_take_profit(candle['close'], 'BUY'),
                    'risk_reward_ratio': self.config['risk_management']['take_profit_percent'] / self.config['risk_management']['stop_loss_percent']
                }
                signals.append(signal)
                
            elif 'bearish_bos' in analysis_result.get('events', []):
                signal = {
                    'type': 'SELL',
                    'reason': 'Bearish BOS Confirmed',
                    'timestamp': candle['timestamp'],
                    'price': candle['close'],
                    'candle_index': candle_index,
                    'confidence': 0.8,
                    'stop_loss': self._calculate_stop_loss(candle['close'], 'SELL'),
                    'take_profit': self._calculate_take_profit(candle['close'], 'SELL'),
                    'risk_reward_ratio': self.config['risk_management']['take_profit_percent'] / self.config['risk_management']['stop_loss_percent']
                }
                signals.append(signal)
        
        # Signal generation for ChoCH events (trend change confirmation)
        if analysis_result.get('choch_detected'):
            if 'bullish_choch' in analysis_result.get('events', []):
                signal = {
                    'type': 'BUY',
                    'reason': 'Bullish ChoCH - Trend Change',
                    'timestamp': candle['timestamp'],
                    'price': candle['close'],
                    'candle_index': candle_index,
                    'confidence': 0.9,  # Higher confidence for trend changes
                    'stop_loss': self._calculate_stop_loss(candle['close'], 'BUY'),
                    'take_profit': self._calculate_take_profit(candle['close'], 'BUY'),
                    'risk_reward_ratio': self.config['risk_management']['take_profit_percent'] / self.config['risk_management']['stop_loss_percent']
                }
                signals.append(signal)
                
            elif 'bearish_choch' in analysis_result.get('events', []):
                signal = {
                    'type': 'SELL',
                    'reason': 'Bearish ChoCH - Trend Change',
                    'timestamp': candle['timestamp'],
                    'price': candle['close'],
                    'candle_index': candle_index,
                    'confidence': 0.9,  # Higher confidence for trend changes
                    'stop_loss': self._calculate_stop_loss(candle['close'], 'SELL'),
                    'take_profit': self._calculate_take_profit(candle['close'], 'SELL'),
                    'risk_reward_ratio': self.config['risk_management']['take_profit_percent'] / self.config['risk_management']['stop_loss_percent']
                }
                signals.append(signal)
        
        # Add alerts for new signals
        if signals and self.config['enable_alerts']:
            for signal in signals:
                alert_msg = f"{signal['type']} Signal: {signal['reason']} at {signal['price']:.2f} (Confidence: {signal['confidence']:.1%})"
                self.analyzer.add_alert(alert_msg)
        
        return signals

    def _calculate_stop_loss(self, entry_price: float, signal_type: str) -> float:
        """Calculate stop loss price based on risk management settings"""
        stop_loss_percent = self.config['risk_management']['stop_loss_percent'] / 100
        
        if signal_type == 'BUY':
            return entry_price * (1 - stop_loss_percent)
        else:  # SELL
            return entry_price * (1 + stop_loss_percent)

    def _calculate_take_profit(self, entry_price: float, signal_type: str) -> float:
        """Calculate take profit price based on risk management settings"""
        take_profit_percent = self.config['risk_management']['take_profit_percent'] / 100
        
        if signal_type == 'BUY':
            return entry_price * (1 + take_profit_percent)
        else:  # SELL
            return entry_price * (1 - take_profit_percent)

    def generate_comprehensive_report(self, save_to_file: bool = True) -> Dict:
        """
        Generate a comprehensive analysis and performance report
        
        Args:
            save_to_file (bool): Whether to save the report to a file
            
        Returns:
            Dict: Comprehensive report data
        """
        print("\\n📊 Generating comprehensive report...")
        
        # Get performance report
        performance_report = create_performance_report(self.analyzer)
        
        # Add trading signals summary
        signal_summary = {
            'total_signals': len(self.current_signals),
            'buy_signals': len([s for s in self.current_signals if s['type'] == 'BUY']),
            'sell_signals': len([s for s in self.current_signals if s['type'] == 'SELL']),
            'average_confidence': np.mean([s['confidence'] for s in self.current_signals]) if self.current_signals else 0,
            'high_confidence_signals': len([s for s in self.current_signals if s['confidence'] >= 0.8])
        }
        
        # Combine all report data
        comprehensive_report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'agent_version': '1.0',
                'data_source': 'XAU_5m_data.csv',
                'analysis_config': self.config
            },
            'performance_analysis': performance_report,
            'trading_signals': signal_summary,
            'recent_signals': self.current_signals[-10:] if self.current_signals else [],
            'market_state': self.analyzer.get_current_state()
        }
        
        # Save to file if requested
        if save_to_file and self.config['save_reports']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.config['report_directory'], f"trading_report_{timestamp}.json")
            
            try:
                with open(report_file, 'w') as f:
                    json.dump(comprehensive_report, f, indent=2, default=str)
                print(f"💾 Report saved to: {report_file}")
            except Exception as e:
                print(f"❌ Error saving report: {str(e)}")
        
        return comprehensive_report

    def create_visual_report(self, save_chart: bool = True) -> None:
        """
        Create and display visual analysis report
        
        Args:
            save_chart (bool): Whether to save the chart to a file
        """
        if not self.visualizer:
            print("❌ Visualizer not available. Enable visualization in config.")
            return
        
        print("\\n🎨 Creating visual analysis report...")
        
        save_path = None
        if save_chart and self.config['save_reports']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.config['report_directory'], f"market_analysis_{timestamp}.png")
        
        self.visualizer.create_comprehensive_report(save_path)

    def run_real_time_analysis(self, update_interval: int = 5) -> None:
        """
        Run real-time analysis mode (simulation using historical data)
        
        Args:
            update_interval (int): Update interval in seconds
        """
        print("\\n🔴 Starting real-time analysis mode...")
        print("Press Ctrl+C to stop real-time analysis")
        
        if not self.visualizer:
            print("❌ Visualizer not available for real-time mode.")
            return
        
        try:
            self.visualizer.create_live_chart(update_interval=update_interval)
        except KeyboardInterrupt:
            print("\\n🛑 Real-time analysis stopped by user.")

    def get_latest_signals(self, count: int = 5) -> List[Dict]:
        """
        Get the latest trading signals
        
        Args:
            count (int): Number of latest signals to return
            
        Returns:
            List[Dict]: Latest trading signals
        """
        return self.current_signals[-count:] if self.current_signals else []

    def print_summary(self) -> None:
        """Print a summary of the current analysis state"""
        state = self.analyzer.get_current_state()
        
        print("\\n" + "="*60)
        print("🤖 AI TRADING AGENT - ANALYSIS SUMMARY")
        print("="*60)
        print(f"📊 Current Trend: {state['current_trend'].upper()}")
        print(f"📈 Total Candles: {state['total_candles']:,}")
        print(f"🎯 Total Events: {state['total_events']}")
        print(f"🚨 Total Alerts: {state['total_alerts']}")
        print(f"💼 Trading Signals: {len(self.current_signals)}")
        print(f"⏰ Status: {'Waiting for BOS' if state['waiting_for_bos'] else 'Searching for patterns'}")
        
        if state['active_levels']:
            print("\\n📍 ACTIVE LEVELS:")
            for level_name, level_data in state['active_levels'].items():
                print(f"   {level_name}: {level_data['price']:.2f}")
        
        if self.current_signals:
            latest_signal = self.current_signals[-1]
            print(f"\\n🎯 LATEST SIGNAL: {latest_signal['type']} - {latest_signal['reason']}")
            print(f"   Price: {latest_signal['price']:.2f} | Confidence: {latest_signal['confidence']:.1%}")
        
        print("="*60)


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='AI Trading Agent - Market Structure Analysis')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--mode', choices=['batch', 'realtime', 'both'], default='batch',
                       help='Analysis mode')
    parser.add_argument('--no-visual', action='store_true', help='Disable visualization')
    parser.add_argument('--no-reports', action='store_true', help='Disable report generation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file: {str(e)}")
    
    # Apply command line overrides
    if args.no_visual:
        config['enable_visualization'] = False
    if args.no_reports:
        config['save_reports'] = False
        
    # Initialize the trading agent
    agent = AITradingAgent(config)
    
    # Load market data
    if not agent.load_market_data(args.data):
        sys.exit(1)
    
    # Perform initial analysis
    initial_results = agent.perform_initial_analysis()
    
    # Run analysis based on mode
    if args.mode in ['batch', 'both']:
        batch_results = agent.run_batch_analysis()
        
        # Generate reports
        report = agent.generate_comprehensive_report()
        
        # Create visual report
        if not args.no_visual:
            agent.create_visual_report()
        
        # Print summary
        agent.print_summary()
    
    if args.mode in ['realtime', 'both']:
        agent.run_real_time_analysis()


if __name__ == "__main__":
    main()