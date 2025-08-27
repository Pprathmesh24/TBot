"""
AI Trading Agent - Market Structure Analyzer
============================================

A sophisticated market structure analysis system based on two-candle retracement patterns
and break of structure (BOS) / change of character (ChoCH) concepts.

Author: AI Trading Agent
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class TrendDirection(Enum):
    """Enum for trend direction states"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class EventType(Enum):
    """Enum for market structure events"""
    BOS_BULLISH = "bos_bullish"
    BOS_BEARISH = "bos_bearish"
    CHOCH_BULLISH = "choch_bullish"
    CHOCH_BEARISH = "choch_bearish"


@dataclass
class StructureLevel:
    """Data class for storing structure levels (CH1, CL1, CH2, CL2)"""
    price: float
    timestamp: datetime
    level_type: str  # 'CH1', 'CL1', 'CH2', 'CL2'
    candle_index: int
    is_broken: bool = False
    break_timestamp: Optional[datetime] = None
    break_candle_index: Optional[int] = None


@dataclass
class MarketEvent:
    """Data class for storing market structure events"""
    event_type: EventType
    timestamp: datetime
    candle_index: int
    price: float
    from_level: Optional[StructureLevel] = None
    to_level: Optional[StructureLevel] = None
    description: str = ""


class MarketStructureAnalyzer:
    """
    Main class for analyzing market structure using two-candle retracement patterns,
    BOS (Break of Structure), and ChoCH (Change of Character) concepts.
    """
    
    def __init__(self, lookback_period: int = 1000):
        """
        Initialize the Market Structure Analyzer
        
        Args:
            lookback_period (int): Number of candles to analyze for initial trend detection
        """
        self.lookback_period = lookback_period
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.candles: List[Dict] = []
        
        # Current state
        self.current_trend: TrendDirection = TrendDirection.NEUTRAL
        self.trend_start_index: int = 0
        
        # Structure levels storage
        self.ch1_levels: List[StructureLevel] = []  # Confirm High Level 1 (Resistance)
        self.cl1_levels: List[StructureLevel] = []  # Confirm Low Level 1 (Support) 
        self.ch2_levels: List[StructureLevel] = []  # Confirm High Level 2 (Resistance)
        self.cl2_levels: List[StructureLevel] = []  # Confirm Low Level 2 (Support)
        
        # Active levels (current working levels)
        self.active_ch1: Optional[StructureLevel] = None
        self.active_cl1: Optional[StructureLevel] = None
        self.active_ch2: Optional[StructureLevel] = None
        self.active_cl2: Optional[StructureLevel] = None
        
        # Events and signals
        self.events: List[MarketEvent] = []
        self.alerts: List[str] = []
        
        # Pattern tracking
        self.waiting_for_bos: bool = False
        self.last_pattern_index: int = -1
        self.consecutive_closes_count: int = 0
        self.consecutive_direction: Optional[str] = None
        
        # Performance tracking
        self.analysis_stats = {
            'total_bos_events': 0,
            'total_choch_events': 0,
            'bullish_periods': 0,
            'bearish_periods': 0,
            'current_trend_duration': 0
        }

    def load_data(self, file_path: str) -> bool:
        """
        Load OHLCV data from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            # Load the CSV file
            self.data = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Need: {required_columns}")
            
            # Convert Date column to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            
            # Sort by date to ensure chronological order
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            
            # Convert to list of dictionaries for easier access
            self.candles = []
            for idx, row in self.data.iterrows():
                candle = {
                    'index': idx,
                    'timestamp': row['Date'],
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'is_green': float(row['Close']) > float(row['Open']),
                    'is_red': float(row['Close']) < float(row['Open'])
                }
                self.candles.append(candle)
            
            print(f"✅ Successfully loaded {len(self.candles)} candles from {file_path}")
            print(f"📊 Data range: {self.candles[0]['timestamp']} to {self.candles[-1]['timestamp']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def determine_initial_trend(self) -> TrendDirection:
        """
        Analyze the last 1000 candles to find absolute highest and lowest points
        and determine initial trend direction based on which occurs first chronologically.
        
        Returns:
            TrendDirection: Initial trend direction based on historical analysis
        """
        if not self.candles or len(self.candles) < 10:
            return TrendDirection.NEUTRAL
        
        # Use lookback period or all available data if less than lookback
        analysis_period = min(self.lookback_period, len(self.candles))
        start_index = max(0, len(self.candles) - analysis_period)
        analysis_candles = self.candles[start_index:]
        
        if len(analysis_candles) < 10:
            return TrendDirection.NEUTRAL
        
        # Find absolute highest and lowest points in the analysis period
        highest_price = float('-inf')
        lowest_price = float('inf')
        highest_index = -1
        lowest_index = -1
        
        for i, candle in enumerate(analysis_candles):
            if candle['high'] > highest_price:
                highest_price = candle['high']
                highest_index = start_index + i
                
            if candle['low'] < lowest_price:
                lowest_price = candle['low']
                lowest_index = start_index + i
        
        # Determine which occurs first chronologically
        if lowest_index < highest_index:
            # Lowest comes first → Start in bullish trend mode
            initial_trend = TrendDirection.BULLISH
            self.trend_start_index = lowest_index
            trend_reasoning = f"Lowest point ({lowest_price:.2f}) at index {lowest_index} occurs before highest point ({highest_price:.2f}) at index {highest_index}"
        else:
            # Highest comes first → Start in bearish trend mode  
            initial_trend = TrendDirection.BEARISH
            self.trend_start_index = highest_index
            trend_reasoning = f"Highest point ({highest_price:.2f}) at index {highest_index} occurs before lowest point ({lowest_price:.2f}) at index {lowest_index}"
        
        self.current_trend = initial_trend
        
        print(f"🔍 Initial Trend Analysis (Last {analysis_period} candles):")
        print(f"📈 Highest: {highest_price:.2f} at index {highest_index} ({self.candles[highest_index]['timestamp']})")
        print(f"📉 Lowest: {lowest_price:.2f} at index {lowest_index} ({self.candles[lowest_index]['timestamp']})")
        print(f"🎯 Initial Trend: {initial_trend.value.upper()}")
        print(f"💡 Reasoning: {trend_reasoning}")
        
        return initial_trend

    def is_bullish_candle(self, candle: Dict) -> bool:
        """Check if candle is bullish (green)"""
        return candle['close'] > candle['open']
    
    def is_bearish_candle(self, candle: Dict) -> bool:
        """Check if candle is bearish (red)"""
        return candle['close'] < candle['open']

    def find_three_candle_pattern_bullish(self, start_index: int) -> Optional[Tuple[int, int, int]]:
        """
        Find three-candle bullish retracement pattern:
        - Candle 1: Green (bullish impulse)
        - Candle 2: Red (correction begins) 
        - Candle 3: Red (correction continues) AND Candle 3's close < Candle 2's low
        
        Args:
            start_index (int): Index to start searching from
            
        Returns:
            Optional[Tuple[int, int, int]]: Indices of the three candles if pattern found, None otherwise
        """
        if start_index + 2 >= len(self.candles):
            return None
            
        candle1 = self.candles[start_index]
        candle2 = self.candles[start_index + 1] 
        candle3 = self.candles[start_index + 2]
        
        # Check pattern conditions
        if (self.is_bullish_candle(candle1) and 
            self.is_bearish_candle(candle2) and 
            self.is_bearish_candle(candle3) and
            candle3['close'] < candle2['low']):
            
            return (start_index, start_index + 1, start_index + 2)
        
        return None

    def find_three_candle_pattern_bearish(self, start_index: int) -> Optional[Tuple[int, int, int]]:
        """
        Find three-candle bearish retracement pattern:
        - Candle 1: Red (bearish impulse)
        - Candle 2: Green (correction begins)
        - Candle 3: Green (correction continues) AND Candle 3's close > Candle 2's high
        
        Args:
            start_index (int): Index to start searching from
            
        Returns:
            Optional[Tuple[int, int, int]]: Indices of the three candles if pattern found, None otherwise
        """
        if start_index + 2 >= len(self.candles):
            return None
            
        candle1 = self.candles[start_index]
        candle2 = self.candles[start_index + 1]
        candle3 = self.candles[start_index + 2]
        
        # Check pattern conditions
        if (self.is_bearish_candle(candle1) and 
            self.is_bullish_candle(candle2) and 
            self.is_bullish_candle(candle3) and
            candle3['close'] > candle2['high']):
            
            return (start_index, start_index + 1, start_index + 2)
        
        return None

    def get_highest_point_in_range(self, start_index: int, end_index: int) -> Tuple[float, int]:
        """Get the highest price and its index within a range of candles"""
        highest_price = float('-inf')
        highest_index = start_index
        
        for i in range(start_index, min(end_index + 1, len(self.candles))):
            if self.candles[i]['high'] > highest_price:
                highest_price = self.candles[i]['high']
                highest_index = i
                
        return highest_price, highest_index

    def get_lowest_point_in_range(self, start_index: int, end_index: int) -> Tuple[float, int]:
        """Get the lowest price and its index within a range of candles"""
        lowest_price = float('inf')
        lowest_index = start_index
        
        for i in range(start_index, min(end_index + 1, len(self.candles))):
            if self.candles[i]['low'] < lowest_price:
                lowest_price = self.candles[i]['low']
                lowest_index = i
                
        return lowest_price, lowest_index

    def add_alert(self, message: str) -> None:
        """Add an alert message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = f"[{timestamp}] {message}"
        self.alerts.append(alert)
        print(f"🚨 ALERT: {message}")

    def reset_state(self) -> None:
        """Reset the analyzer state for fresh analysis"""
        self.current_trend = TrendDirection.NEUTRAL
        self.trend_start_index = 0
        self.ch1_levels.clear()
        self.cl1_levels.clear() 
        self.ch2_levels.clear()
        self.cl2_levels.clear()
        self.active_ch1 = None
        self.active_cl1 = None
        self.active_ch2 = None
        self.active_cl2 = None
        self.events.clear()
        self.alerts.clear()
        self.waiting_for_bos = False
        self.last_pattern_index = -1
        self.consecutive_closes_count = 0
        self.consecutive_direction = None
        
        # Reset stats
        self.analysis_stats = {
            'total_bos_events': 0,
            'total_choch_events': 0,
            'bullish_periods': 0,
            'bearish_periods': 0,
            'current_trend_duration': 0
        }

    def process_bullish_pattern(self, pattern_indices: Tuple[int, int, int]) -> Optional[StructureLevel]:
        """
        Process a bullish three-candle retracement pattern and create CH1 level
        
        Args:
            pattern_indices: Tuple of (candle1_index, candle2_index, candle3_index)
            
        Returns:
            Optional[StructureLevel]: Created CH1 level if successful
        """
        idx1, idx2, idx3 = pattern_indices
        
        # Find the highest point among the three candles
        highest_price, highest_index = self.get_highest_point_in_range(idx1, idx3)
        
        # Create CH1 level (Confirm High Level 1 - Resistance)
        ch1_level = StructureLevel(
            price=highest_price,
            timestamp=self.candles[highest_index]['timestamp'],
            level_type='CH1',
            candle_index=highest_index
        )
        
        # Store the level
        self.ch1_levels.append(ch1_level)
        self.active_ch1 = ch1_level
        self.waiting_for_bos = True
        self.last_pattern_index = idx3
        
        print(f"📊 Bullish Pattern Detected: CH1 Level created at {highest_price:.2f} (Index: {highest_index})")
        self.add_alert(f"Bullish 3-candle pattern: CH1 resistance at {highest_price:.2f}")
        
        return ch1_level

    def process_bearish_pattern(self, pattern_indices: Tuple[int, int, int]) -> Optional[StructureLevel]:
        """
        Process a bearish three-candle retracement pattern and create CL1 level
        
        Args:
            pattern_indices: Tuple of (candle1_index, candle2_index, candle3_index)
            
        Returns:
            Optional[StructureLevel]: Created CL1 level if successful
        """
        idx1, idx2, idx3 = pattern_indices
        
        # Find the lowest point among the three candles
        lowest_price, lowest_index = self.get_lowest_point_in_range(idx1, idx3)
        
        # Create CL1 level (Confirm Low Level 1 - Support)
        cl1_level = StructureLevel(
            price=lowest_price,
            timestamp=self.candles[lowest_index]['timestamp'],
            level_type='CL1',
            candle_index=lowest_index
        )
        
        # Store the level
        self.cl1_levels.append(cl1_level)
        self.active_cl1 = cl1_level
        self.waiting_for_bos = True
        self.last_pattern_index = idx3
        
        print(f"📊 Bearish Pattern Detected: CL1 Level created at {lowest_price:.2f} (Index: {lowest_index})")
        self.add_alert(f"Bearish 3-candle pattern: CL1 support at {lowest_price:.2f}")
        
        return cl1_level

    def check_bullish_bos(self, current_index: int) -> bool:
        """
        Check for bullish Break of Structure (BOS)
        Wait for a green candle to close above CH1
        
        Args:
            current_index (int): Current candle index to check
            
        Returns:
            bool: True if BOS confirmed, False otherwise
        """
        if not self.active_ch1 or not self.waiting_for_bos:
            return False
            
        current_candle = self.candles[current_index]
        
        # Check if current candle is green and closes above CH1
        if (self.is_bullish_candle(current_candle) and 
            current_candle['close'] > self.active_ch1.price):
            
            # Mark CH1 as broken
            self.active_ch1.is_broken = True
            self.active_ch1.break_timestamp = current_candle['timestamp']
            self.active_ch1.break_candle_index = current_index
            
            # Create BOS event
            bos_event = MarketEvent(
                event_type=EventType.BOS_BULLISH,
                timestamp=current_candle['timestamp'],
                candle_index=current_index,
                price=current_candle['close'],
                from_level=self.active_ch1,
                description=f"Bullish BOS: Green candle closed above CH1 ({self.active_ch1.price:.2f})"
            )
            
            self.events.append(bos_event)
            self.analysis_stats['total_bos_events'] += 1
            
            # Find CL2 level (lowest point between CH1 and BOS candle)
            lowest_price, lowest_index = self.get_lowest_point_in_range(
                self.active_ch1.candle_index, current_index
            )
            
            cl2_level = StructureLevel(
                price=lowest_price,
                timestamp=self.candles[lowest_index]['timestamp'],
                level_type='CL2',
                candle_index=lowest_index
            )
            
            self.cl2_levels.append(cl2_level)
            self.active_cl2 = cl2_level
            
            print(f"🚀 BULLISH BOS CONFIRMED! CH1 ({self.active_ch1.price:.2f}) broken by close at {current_candle['close']:.2f}")
            print(f"📉 CL2 Support level created at {cl2_level.price:.2f}")
            
            self.add_alert(f"Bullish BOS confirmed: {current_candle['close']:.2f} > CH1 {self.active_ch1.price:.2f}")
            self.add_alert(f"CL2 support established at {cl2_level.price:.2f}")
            
            # Reset pattern search state
            self.waiting_for_bos = False
            
            return True
            
        return False

    def check_bearish_bos(self, current_index: int) -> bool:
        """
        Check for bearish Break of Structure (BOS)
        Wait for a red candle to close below CL1
        
        Args:
            current_index (int): Current candle index to check
            
        Returns:
            bool: True if BOS confirmed, False otherwise
        """
        if not self.active_cl1 or not self.waiting_for_bos:
            return False
            
        current_candle = self.candles[current_index]
        
        # Check if current candle is red and closes below CL1
        if (self.is_bearish_candle(current_candle) and 
            current_candle['close'] < self.active_cl1.price):
            
            # Mark CL1 as broken
            self.active_cl1.is_broken = True
            self.active_cl1.break_timestamp = current_candle['timestamp']
            self.active_cl1.break_candle_index = current_index
            
            # Create BOS event
            bos_event = MarketEvent(
                event_type=EventType.BOS_BEARISH,
                timestamp=current_candle['timestamp'],
                candle_index=current_index,
                price=current_candle['close'],
                from_level=self.active_cl1,
                description=f"Bearish BOS: Red candle closed below CL1 ({self.active_cl1.price:.2f})"
            )
            
            self.events.append(bos_event)
            self.analysis_stats['total_bos_events'] += 1
            
            # Find CH2 level (highest point between CL1 and BOS candle)
            highest_price, highest_index = self.get_highest_point_in_range(
                self.active_cl1.candle_index, current_index
            )
            
            ch2_level = StructureLevel(
                price=highest_price,
                timestamp=self.candles[highest_index]['timestamp'],
                level_type='CH2',
                candle_index=highest_index
            )
            
            self.ch2_levels.append(ch2_level)
            self.active_ch2 = ch2_level
            
            print(f"🔻 BEARISH BOS CONFIRMED! CL1 ({self.active_cl1.price:.2f}) broken by close at {current_candle['close']:.2f}")
            print(f"📈 CH2 Resistance level created at {ch2_level.price:.2f}")
            
            self.add_alert(f"Bearish BOS confirmed: {current_candle['close']:.2f} < CL1 {self.active_cl1.price:.2f}")
            self.add_alert(f"CH2 resistance established at {ch2_level.price:.2f}")
            
            # Reset pattern search state
            self.waiting_for_bos = False
            
            return True
            
        return False

    def check_bullish_choch(self, current_index: int) -> bool:
        """
        Check for Bullish Change of Character (ChoCH)
        After bearish trend with CL1 and CH2 levels, if:
        - New pattern fails to break below new CL1
        - AND previous CH2 is broken with TWO consecutive candle closes above CH2
        
        Args:
            current_index (int): Current candle index to check
            
        Returns:
            bool: True if ChoCH confirmed, False otherwise
        """
        if (self.current_trend != TrendDirection.BEARISH or 
            not self.active_ch2 or 
            current_index < 1):
            return False
            
        current_candle = self.candles[current_index]
        previous_candle = self.candles[current_index - 1]
        
        # Check for two consecutive closes above CH2
        if (current_candle['close'] > self.active_ch2.price and 
            previous_candle['close'] > self.active_ch2.price):
            
            # Confirm Bullish ChoCH
            choch_event = MarketEvent(
                event_type=EventType.CHOCH_BULLISH,
                timestamp=current_candle['timestamp'],
                candle_index=current_index,
                price=current_candle['close'],
                from_level=self.active_ch2,
                description=f"Bullish ChoCH: Two consecutive closes above CH2 ({self.active_ch2.price:.2f})"
            )
            
            self.events.append(choch_event)
            self.analysis_stats['total_choch_events'] += 1
            self.analysis_stats['bullish_periods'] += 1
            
            # Switch to bullish trend
            self.current_trend = TrendDirection.BULLISH
            self.trend_start_index = current_index
            
            # Reset active levels for new trend
            self.active_cl1 = None
            self.active_ch2 = None
            
            print(f"🔄 BULLISH CHOCH CONFIRMED! Trend switched to BULLISH")
            print(f"📊 Two consecutive closes above CH2 ({self.active_ch2.price:.2f})")
            
            self.add_alert(f"Bullish ChoCH: Trend changed to BULLISH at {current_candle['close']:.2f}")
            
            return True
            
        return False

    def check_bearish_choch(self, current_index: int) -> bool:
        """
        Check for Bearish Change of Character (ChoCH)
        After bullish trend with CH1 and CL2 levels, if:
        - New pattern fails to break above new CH1
        - AND previous CL2 is broken with TWO consecutive candle closes below CL2
        
        Args:
            current_index (int): Current candle index to check
            
        Returns:
            bool: True if ChoCH confirmed, False otherwise
        """
        if (self.current_trend != TrendDirection.BULLISH or 
            not self.active_cl2 or 
            current_index < 1):
            return False
            
        current_candle = self.candles[current_index]
        previous_candle = self.candles[current_index - 1]
        
        # Check for two consecutive closes below CL2
        if (current_candle['close'] < self.active_cl2.price and 
            previous_candle['close'] < self.active_cl2.price):
            
            # Confirm Bearish ChoCH
            choch_event = MarketEvent(
                event_type=EventType.CHOCH_BEARISH,
                timestamp=current_candle['timestamp'],
                candle_index=current_index,
                price=current_candle['close'],
                from_level=self.active_cl2,
                description=f"Bearish ChoCH: Two consecutive closes below CL2 ({self.active_cl2.price:.2f})"
            )
            
            self.events.append(choch_event)
            self.analysis_stats['total_choch_events'] += 1
            self.analysis_stats['bearish_periods'] += 1
            
            # Switch to bearish trend
            self.current_trend = TrendDirection.BEARISH
            self.trend_start_index = current_index
            
            # Reset active levels for new trend
            self.active_ch1 = None
            self.active_cl2 = None
            
            print(f"🔄 BEARISH CHOCH CONFIRMED! Trend switched to BEARISH")
            print(f"📊 Two consecutive closes below CL2 ({self.active_cl2.price:.2f})")
            
            self.add_alert(f"Bearish ChoCH: Trend changed to BEARISH at {current_candle['close']:.2f}")
            
            return True
            
        return False

    def analyze_candle(self, candle_index: int) -> Dict:
        """
        Analyze a single candle for patterns, BOS, and ChoCH
        
        Args:
            candle_index (int): Index of candle to analyze
            
        Returns:
            Dict: Analysis results for this candle
        """
        if candle_index >= len(self.candles):
            return {'error': 'Invalid candle index'}
            
        results = {
            'candle_index': candle_index,
            'timestamp': self.candles[candle_index]['timestamp'],
            'price': self.candles[candle_index]['close'],
            'pattern_found': False,
            'bos_detected': False,
            'choch_detected': False,
            'events': [],
            'new_levels': []
        }
        
        # Skip analysis if we don't have enough candles
        if candle_index < self.trend_start_index + 3:
            return results
        
        # Check for ChoCH first (higher priority)
        if self.check_bullish_choch(candle_index):
            results['choch_detected'] = True
            results['events'].append('bullish_choch')
        elif self.check_bearish_choch(candle_index):
            results['choch_detected'] = True
            results['events'].append('bearish_choch')
        
        # Check for BOS if waiting for one
        if self.waiting_for_bos:
            if self.current_trend == TrendDirection.BULLISH:
                if self.check_bullish_bos(candle_index):
                    results['bos_detected'] = True
                    results['events'].append('bullish_bos')
            elif self.current_trend == TrendDirection.BEARISH:
                if self.check_bearish_bos(candle_index):
                    results['bos_detected'] = True
                    results['events'].append('bearish_bos')
        
        # Look for new patterns only after confirmed BOS or at trend start
        if not self.waiting_for_bos and candle_index > self.last_pattern_index + 3:
            
            if self.current_trend == TrendDirection.BULLISH:
                # Look for bullish retracement pattern
                pattern = self.find_three_candle_pattern_bullish(candle_index - 2)
                if pattern:
                    self.process_bullish_pattern(pattern)
                    results['pattern_found'] = True
                    results['events'].append('bullish_pattern')
                    results['new_levels'].append('CH1')
            
            elif self.current_trend == TrendDirection.BEARISH:
                # Look for bearish retracement pattern  
                pattern = self.find_three_candle_pattern_bearish(candle_index - 2)
                if pattern:
                    self.process_bearish_pattern(pattern)
                    results['pattern_found'] = True
                    results['events'].append('bearish_pattern')
                    results['new_levels'].append('CL1')
        
        return results

    def get_current_state(self) -> Dict:
        """
        Get current analyzer state and statistics
        
        Returns:
            Dict: Current state information
        """
        active_levels = {}
        if self.active_ch1:
            active_levels['CH1'] = {'price': self.active_ch1.price, 'timestamp': self.active_ch1.timestamp}
        if self.active_cl1:
            active_levels['CL1'] = {'price': self.active_cl1.price, 'timestamp': self.active_cl1.timestamp}
        if self.active_ch2:
            active_levels['CH2'] = {'price': self.active_ch2.price, 'timestamp': self.active_ch2.timestamp}
        if self.active_cl2:
            active_levels['CL2'] = {'price': self.active_cl2.price, 'timestamp': self.active_cl2.timestamp}
        
        return {
            'current_trend': self.current_trend.value,
            'trend_start_index': self.trend_start_index,
            'active_levels': active_levels,
            'total_events': len(self.events),
            'total_alerts': len(self.alerts),
            'waiting_for_bos': self.waiting_for_bos,
            'statistics': self.analysis_stats.copy(),
            'data_loaded': self.data is not None,
            'total_candles': len(self.candles) if self.candles else 0
        }