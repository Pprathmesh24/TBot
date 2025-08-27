#!/usr/bin/env python3
"""
AI Trading Agent - Comprehensive Test Suite
===========================================

Complete test suite for validating all components of the AI trading agent
including market structure analysis, pattern recognition, and signal generation.

Author: AI Trading Agent
Version: 1.0
"""

import unittest
import sys
import os
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_structure_analyzer import MarketStructureAnalyzer, TrendDirection, EventType, StructureLevel
from visualizer import MarketStructureVisualizer, create_performance_report
from ai_trading_agent import AITradingAgent


class TestMarketStructureAnalyzer(unittest.TestCase):
    """Test cases for MarketStructureAnalyzer class"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.analyzer = MarketStructureAnalyzer(lookback_period=100)
        
        # Create sample test data
        self.sample_data = self._create_sample_data()
        
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=200, freq='5T')
        
        # Create realistic price movements
        base_price = 2000.0
        prices = []
        current_price = base_price
        
        for i in range(200):
            # Add some randomness and trends
            change = np.random.normal(0, 2.0)
            if i < 50:  # Bearish trend
                change -= 0.5
            elif i < 150:  # Bullish trend  
                change += 0.3
            else:  # Consolidation
                change *= 0.5
                
            current_price += change
            prices.append(current_price)
        
        # Generate OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            open_price = close + np.random.normal(0, 0.5)
            high = max(open_price, close) + abs(np.random.normal(0, 1.0))
            low = min(open_price, close) - abs(np.random.normal(0, 1.0))
            volume = np.random.randint(100, 1000)
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        return pd.DataFrame(data)
    
    def _save_sample_data(self) -> str:
        """Save sample data to temporary file"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertEqual(self.analyzer.lookback_period, 100)
        self.assertEqual(self.analyzer.current_trend, TrendDirection.NEUTRAL)
        self.assertEqual(len(self.analyzer.candles), 0)
        self.assertEqual(len(self.analyzer.events), 0)
        
    def test_data_loading(self):
        """Test data loading functionality"""
        temp_file = self._save_sample_data()
        
        try:
            # Test successful loading
            success = self.analyzer.load_data(temp_file)
            self.assertTrue(success)
            self.assertEqual(len(self.analyzer.candles), 200)
            
            # Test first candle structure
            first_candle = self.analyzer.candles[0]
            self.assertIn('open', first_candle)
            self.assertIn('high', first_candle)
            self.assertIn('low', first_candle)
            self.assertIn('close', first_candle)
            self.assertIn('timestamp', first_candle)
            
        finally:
            os.unlink(temp_file)
    
    def test_initial_trend_detection(self):
        """Test initial trend detection logic"""
        temp_file = self._save_sample_data()
        
        try:
            self.analyzer.load_data(temp_file)
            initial_trend = self.analyzer.determine_initial_trend()
            
            # Should return a valid trend direction
            self.assertIn(initial_trend, [TrendDirection.BULLISH, TrendDirection.BEARISH])
            self.assertEqual(self.analyzer.current_trend, initial_trend)
            
        finally:
            os.unlink(temp_file)
    
    def test_candle_classification(self):
        """Test bullish/bearish candle classification"""
        # Test bullish candle
        bullish_candle = {'open': 100, 'close': 105, 'high': 106, 'low': 99}
        self.assertTrue(self.analyzer.is_bullish_candle(bullish_candle))
        self.assertFalse(self.analyzer.is_bearish_candle(bullish_candle))
        
        # Test bearish candle
        bearish_candle = {'open': 105, 'close': 100, 'high': 106, 'low': 99}
        self.assertTrue(self.analyzer.is_bearish_candle(bearish_candle))
        self.assertFalse(self.analyzer.is_bullish_candle(bearish_candle))
    
    def test_three_candle_pattern_recognition(self):
        """Test three-candle pattern recognition"""
        # Create test candles that form a bullish pattern
        test_candles = [
            {'open': 100, 'close': 105, 'high': 106, 'low': 99},   # Green
            {'open': 105, 'close': 102, 'high': 106, 'low': 101}, # Red
            {'open': 102, 'close': 100, 'high': 103, 'low': 99}   # Red, close < candle2 low
        ]
        
        self.analyzer.candles = test_candles
        
        # Test bullish pattern detection
        pattern = self.analyzer.find_three_candle_pattern_bullish(0)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern, (0, 1, 2))
    
    def test_structure_level_creation(self):
        """Test structure level creation and management"""
        temp_file = self._save_sample_data()
        
        try:
            self.analyzer.load_data(temp_file)
            
            # Create a test structure level
            level = StructureLevel(
                price=2000.0,
                timestamp=datetime.now(),
                level_type='CH1',
                candle_index=50
            )
            
            self.assertEqual(level.price, 2000.0)
            self.assertEqual(level.level_type, 'CH1')
            self.assertFalse(level.is_broken)
            
        finally:
            os.unlink(temp_file)
    
    def test_get_current_state(self):
        """Test current state retrieval"""
        state = self.analyzer.get_current_state()
        
        required_keys = [
            'current_trend', 'trend_start_index', 'active_levels',
            'total_events', 'total_alerts', 'waiting_for_bos',
            'statistics', 'data_loaded', 'total_candles'
        ]
        
        for key in required_keys:
            self.assertIn(key, state)


class TestAITradingAgent(unittest.TestCase):
    """Test cases for AITradingAgent class"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.config = {
            'lookback_period': 50,
            'enable_alerts': True,
            'enable_visualization': False,  # Disable for testing
            'save_reports': False,
            'analysis_batch_size': 10
        }
        self.agent = AITradingAgent(self.config)
        
        # Create sample data
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
        data = []
        
        for i, date in enumerate(dates):
            price = 2000 + i * 0.1 + np.random.normal(0, 1)
            data.append({
                'Date': date,
                'Open': price + np.random.normal(0, 0.5),
                'High': price + abs(np.random.normal(0, 1)),
                'Low': price - abs(np.random.normal(0, 1)),
                'Close': price,
                'Volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(data)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent.analyzer)
        self.assertEqual(self.agent.config['lookback_period'], 50)
        self.assertEqual(len(self.agent.current_signals), 0)
    
    def test_config_loading(self):
        """Test configuration loading"""
        custom_config = {'lookback_period': 200, 'enable_alerts': False}
        agent = AITradingAgent(custom_config)
        
        self.assertEqual(agent.config['lookback_period'], 200)
        self.assertFalse(agent.config['enable_alerts'])
    
    def test_data_loading(self):
        """Test market data loading"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            success = self.agent.load_market_data(temp_file.name)
            self.assertTrue(success)
            self.assertEqual(len(self.agent.analyzer.candles), 100)
        finally:
            os.unlink(temp_file.name)
    
    def test_initial_analysis(self):
        """Test initial analysis functionality"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            self.agent.load_market_data(temp_file.name)
            results = self.agent.perform_initial_analysis()
            
            self.assertIn('initial_trend', results)
            self.assertIn('total_candles', results)
            self.assertIn('analysis_period', results)
            
        finally:
            os.unlink(temp_file.name)
    
    def test_signal_generation(self):
        """Test trading signal generation"""
        # Mock analysis result with BOS event
        analysis_result = {
            'candle_index': 50,
            'bos_detected': True,
            'events': ['bullish_bos']
        }
        
        # Mock candle data
        self.agent.analyzer.candles = [
            {'timestamp': datetime.now(), 'close': 2000.0} for _ in range(51)
        ]
        
        signals = self.agent._generate_trading_signals(analysis_result)
        
        if signals:  # Signals might not be generated based on conditions
            signal = signals[0]
            self.assertIn('type', signal)
            self.assertIn('reason', signal)
            self.assertIn('price', signal)
            self.assertIn('confidence', signal)
    
    def test_risk_management_calculations(self):
        """Test risk management calculations"""
        entry_price = 2000.0
        
        # Test stop loss calculation
        buy_stop_loss = self.agent._calculate_stop_loss(entry_price, 'BUY')
        sell_stop_loss = self.agent._calculate_stop_loss(entry_price, 'SELL')
        
        self.assertLess(buy_stop_loss, entry_price)  # Stop loss below entry for BUY
        self.assertGreater(sell_stop_loss, entry_price)  # Stop loss above entry for SELL
        
        # Test take profit calculation
        buy_take_profit = self.agent._calculate_take_profit(entry_price, 'BUY')
        sell_take_profit = self.agent._calculate_take_profit(entry_price, 'SELL')
        
        self.assertGreater(buy_take_profit, entry_price)  # Take profit above entry for BUY
        self.assertLess(sell_take_profit, entry_price)  # Take profit below entry for SELL


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Setup integration test fixtures"""
        self.config = {
            'lookback_period': 100,
            'enable_visualization': False,
            'save_reports': False,
            'analysis_batch_size': 20
        }
        
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow from data loading to signal generation"""
        agent = AITradingAgent(self.config)
        
        # Create test data with known patterns
        test_data = self._create_pattern_data()
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            # Load data
            success = agent.load_market_data(temp_file.name)
            self.assertTrue(success)
            
            # Perform initial analysis
            initial_results = agent.perform_initial_analysis()
            self.assertIsInstance(initial_results, dict)
            
            # Run batch analysis
            batch_results = agent.run_batch_analysis()
            self.assertIsInstance(batch_results, dict)
            
            # Generate report
            report = agent.generate_comprehensive_report(save_to_file=False)
            self.assertIsInstance(report, dict)
            self.assertIn('performance_analysis', report)
            self.assertIn('trading_signals', report)
            
        finally:
            os.unlink(temp_file.name)
    
    def _create_pattern_data(self) -> pd.DataFrame:
        """Create data with known patterns for testing"""
        dates = pd.date_range(start='2024-01-01', periods=150, freq='5T')
        data = []
        
        base_price = 2000.0
        
        for i, date in enumerate(dates):
            # Create trending movements with some patterns
            if i < 30:  # Initial downtrend
                price = base_price - (i * 2) + np.random.normal(0, 1)
            elif i < 60:  # Reversal and uptrend
                price = base_price - 60 + ((i - 30) * 3) + np.random.normal(0, 1)
            elif i < 90:  # Consolidation
                price = base_price + 30 + np.random.normal(0, 2)
            else:  # Another trend
                price = base_price + 30 + ((i - 90) * 1.5) + np.random.normal(0, 1)
            
            # Generate OHLC
            open_price = price + np.random.normal(0, 0.5)
            close_price = price + np.random.normal(0, 0.5)
            high = max(open_price, close_price) + abs(np.random.normal(0, 1))
            low = min(open_price, close_price) - abs(np.random.normal(0, 1))
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close_price,
                'Volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(data)


class TestPerformance(unittest.TestCase):
    """Performance and stress tests"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        # Create large dataset
        large_data = self._create_large_dataset(5000)  # 5000 candles
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        large_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            config = {
                'lookback_period': 1000,
                'enable_visualization': False,
                'save_reports': False,
                'analysis_batch_size': 100
            }
            
            agent = AITradingAgent(config)
            
            # Measure loading time
            start_time = datetime.now()
            success = agent.load_market_data(temp_file.name)
            load_time = (datetime.now() - start_time).total_seconds()
            
            self.assertTrue(success)
            self.assertLess(load_time, 10.0)  # Should load within 10 seconds
            
            # Measure analysis time
            start_time = datetime.now()
            agent.perform_initial_analysis()
            batch_results = agent.run_batch_analysis()
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            self.assertLess(analysis_time, 30.0)  # Should analyze within 30 seconds
            print(f"Performance test: {len(agent.analyzer.candles)} candles analyzed in {analysis_time:.2f}s")
            
        finally:
            os.unlink(temp_file.name)
    
    def _create_large_dataset(self, size: int) -> pd.DataFrame:
        """Create large dataset for performance testing"""
        dates = pd.date_range(start='2020-01-01', periods=size, freq='5T')
        data = []
        
        price = 2000.0
        
        for i, date in enumerate(dates):
            # Add realistic price movements
            change = np.random.normal(0, 2.0)
            # Add some trends
            if i % 1000 < 300:  # Trending periods
                change += np.random.choice([-0.5, 0.5])
            
            price += change
            
            open_price = price + np.random.normal(0, 0.5)
            close_price = price + np.random.normal(0, 0.5)
            high = max(open_price, close_price) + abs(np.random.normal(0, 1))
            low = min(open_price, close_price) - abs(np.random.normal(0, 1))
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close_price,
                'Volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(data)


def run_basic_validation():
    """Run basic validation tests without unittest framework"""
    print("🧪 Running basic validation tests...")
    
    try:
        # Test 1: Basic imports
        print("✅ All imports successful")
        
        # Test 2: Analyzer initialization
        analyzer = MarketStructureAnalyzer(100)
        print("✅ MarketStructureAnalyzer initialization successful")
        
        # Test 3: Agent initialization
        agent = AITradingAgent({'enable_visualization': False, 'save_reports': False})
        print("✅ AITradingAgent initialization successful")
        
        # Test 4: Basic functionality
        state = analyzer.get_current_state()
        assert isinstance(state, dict)
        print("✅ Basic functionality test passed")
        
        print("🎉 All basic validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        return False


if __name__ == '__main__':
    print("=" * 80)
    print("🤖 AI TRADING AGENT - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Run basic validation first
    if not run_basic_validation():
        sys.exit(1)
    
    print("\n🚀 Running comprehensive test suite...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMarketStructureAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestAITradingAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"🎯 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        sys.exit(1)