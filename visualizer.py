"""
AI Trading Agent - Market Structure Visualizer
==============================================

Visualization components for displaying market structure analysis including
levels, BOS lines, ChoCH events, and real-time analysis results.

Author: AI Trading Agent
Version: 1.0
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from market_structure_analyzer import MarketStructureAnalyzer, TrendDirection, EventType

# Set style for better looking plots
plt.style.use('dark_background')
sns.set_palette("husl")


class MarketStructureVisualizer:
    """
    Comprehensive visualization class for market structure analysis
    """
    
    def __init__(self, analyzer: MarketStructureAnalyzer):
        """
        Initialize the visualizer with a market structure analyzer
        
        Args:
            analyzer (MarketStructureAnalyzer): The analyzer instance to visualize
        """
        self.analyzer = analyzer
        self.fig = None
        self.axes = None
        
        # Color scheme
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444', 
            'neutral': '#888888',
            'ch1': '#ff6b6b',      # Resistance levels
            'cl1': '#4ecdc4',      # Support levels  
            'ch2': '#ff9999',      # Secondary resistance
            'cl2': '#74d4aa',      # Secondary support
            'bos_line': '#ffeb3b', # BOS lines
            'choch': '#e91e63',    # ChoCH events
            'candle_up': '#26a69a',
            'candle_down': '#ef5350',
            'background': '#1e1e1e'
        }

    def setup_plot(self, figsize: Tuple[int, int] = (16, 10)) -> None:
        """
        Setup the main plotting environment
        
        Args:
            figsize (Tuple[int, int]): Figure size (width, height)
        """
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('AI Trading Agent - Market Structure Analysis', 
                         fontsize=16, fontweight='bold', color='white')
        
        # Configure subplot layout
        self.axes = self.axes.flatten()
        
        # Set background color
        self.fig.patch.set_facecolor(self.colors['background'])
        for ax in self.axes:
            ax.set_facecolor(self.colors['background'])

    def plot_candlestick_chart(self, start_index: int = -500, end_index: int = -1) -> None:
        """
        Plot candlestick chart with market structure overlays
        
        Args:
            start_index (int): Starting candle index (negative for recent data)
            end_index (int): Ending candle index (negative for recent data)
        """
        if not self.analyzer.candles:
            return
            
        ax = self.axes[0]
        ax.clear()
        
        # Prepare data slice
        if start_index < 0:
            start_index = max(0, len(self.analyzer.candles) + start_index)
        if end_index < 0:
            end_index = len(self.analyzer.candles) + end_index + 1
            
        candles_slice = self.analyzer.candles[start_index:end_index]
        
        if not candles_slice:
            return
            
        # Extract data
        timestamps = [c['timestamp'] for c in candles_slice]
        opens = [c['open'] for c in candles_slice]
        highs = [c['high'] for c in candles_slice]
        lows = [c['low'] for c in candles_slice]
        closes = [c['close'] for c in candles_slice]
        
        # Plot candlesticks
        for i, (ts, o, h, l, c) in enumerate(zip(timestamps, opens, highs, lows, closes)):
            color = self.colors['candle_up'] if c > o else self.colors['candle_down']
            
            # Draw wick
            ax.plot([i, i], [l, h], color=color, linewidth=1, alpha=0.8)
            
            # Draw body
            body_height = abs(c - o)
            body_bottom = min(o, c)
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, alpha=0.8, edgecolor=color)
            ax.add_patch(rect)
        
        # Add structure levels
        self._add_structure_levels(ax, start_index, end_index)
        
        # Add BOS lines
        self._add_bos_lines(ax, start_index, end_index)
        
        # Add ChoCH markers
        self._add_choch_markers(ax, start_index, end_index)
        
        # Format axes
        ax.set_title(f'Market Structure Analysis - {self.analyzer.current_trend.value.upper()} Trend', 
                    fontweight='bold', color='white')
        ax.set_ylabel('Price', color='white')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis with time labels (simplified for better performance)
        if len(timestamps) > 50:
            step = len(timestamps) // 10
            tick_indices = range(0, len(timestamps), step)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([timestamps[i].strftime('%m/%d %H:%M') for i in tick_indices], 
                              rotation=45, color='white')
        
        ax.tick_params(colors='white')

    def _add_structure_levels(self, ax, start_index: int, end_index: int) -> None:
        """Add horizontal lines for structure levels"""
        
        # Add active levels
        if self.analyzer.active_ch1:
            if start_index <= self.analyzer.active_ch1.candle_index <= end_index:
                level_x = self.analyzer.active_ch1.candle_index - start_index
                ax.axhline(y=self.analyzer.active_ch1.price, color=self.colors['ch1'], 
                          linestyle='--', linewidth=2, alpha=0.8, label='CH1 (Resistance)')
                ax.text(level_x, self.analyzer.active_ch1.price, 
                       f'CH1: {self.analyzer.active_ch1.price:.2f}', 
                       color=self.colors['ch1'], fontweight='bold')
        
        if self.analyzer.active_cl1:
            if start_index <= self.analyzer.active_cl1.candle_index <= end_index:
                level_x = self.analyzer.active_cl1.candle_index - start_index
                ax.axhline(y=self.analyzer.active_cl1.price, color=self.colors['cl1'], 
                          linestyle='--', linewidth=2, alpha=0.8, label='CL1 (Support)')
                ax.text(level_x, self.analyzer.active_cl1.price, 
                       f'CL1: {self.analyzer.active_cl1.price:.2f}', 
                       color=self.colors['cl1'], fontweight='bold')
        
        if self.analyzer.active_ch2:
            if start_index <= self.analyzer.active_ch2.candle_index <= end_index:
                level_x = self.analyzer.active_ch2.candle_index - start_index
                ax.axhline(y=self.analyzer.active_ch2.price, color=self.colors['ch2'], 
                          linestyle='--', linewidth=2, alpha=0.8, label='CH2 (Resistance)')
                ax.text(level_x, self.analyzer.active_ch2.price, 
                       f'CH2: {self.analyzer.active_ch2.price:.2f}', 
                       color=self.colors['ch2'], fontweight='bold')
        
        if self.analyzer.active_cl2:
            if start_index <= self.analyzer.active_cl2.candle_index <= end_index:
                level_x = self.analyzer.active_cl2.candle_index - start_index
                ax.axhline(y=self.analyzer.active_cl2.price, color=self.colors['cl2'], 
                          linestyle='--', linewidth=2, alpha=0.8, label='CL2 (Support)')
                ax.text(level_x, self.analyzer.active_cl2.price, 
                       f'CL2: {self.analyzer.active_cl2.price:.2f}', 
                       color=self.colors['cl2'], fontweight='bold')

    def _add_bos_lines(self, ax, start_index: int, end_index: int) -> None:
        """Add BOS lines connecting levels to break points"""
        
        for event in self.analyzer.events:
            if event.event_type in [EventType.BOS_BULLISH, EventType.BOS_BEARISH]:
                if (event.from_level and 
                    start_index <= event.candle_index <= end_index and
                    start_index <= event.from_level.candle_index <= end_index):
                    
                    x1 = event.from_level.candle_index - start_index
                    x2 = event.candle_index - start_index
                    y1 = event.from_level.price
                    y2 = event.price
                    
                    color = self.colors['bullish'] if event.event_type == EventType.BOS_BULLISH else self.colors['bearish']
                    
                    ax.plot([x1, x2], [y1, y2], color=self.colors['bos_line'], 
                           linewidth=3, alpha=0.9, linestyle='-')
                    
                    # Add arrow at break point
                    arrow_props = dict(arrowstyle='->', color=color, lw=2)
                    ax.annotate('', xy=(x2, y2), xytext=(x2-5, y2), arrowprops=arrow_props)
                    
                    # Add label
                    label = 'Bullish BOS' if event.event_type == EventType.BOS_BULLISH else 'Bearish BOS'
                    ax.text(x2, y2, label, color=color, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    def _add_choch_markers(self, ax, start_index: int, end_index: int) -> None:
        """Add ChoCH event markers"""
        
        for event in self.analyzer.events:
            if event.event_type in [EventType.CHOCH_BULLISH, EventType.CHOCH_BEARISH]:
                if start_index <= event.candle_index <= end_index:
                    x = event.candle_index - start_index
                    y = event.price
                    
                    color = self.colors['bullish'] if event.event_type == EventType.CHOCH_BULLISH else self.colors['bearish']
                    
                    # Add star marker
                    ax.scatter(x, y, s=200, c=self.colors['choch'], marker='*', 
                              edgecolors=color, linewidth=2, alpha=0.9, zorder=10)
                    
                    # Add label
                    label = 'ChoCH ↑' if event.event_type == EventType.CHOCH_BULLISH else 'ChoCH ↓'
                    ax.text(x, y + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, label, 
                           color=self.colors['choch'], fontweight='bold', ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

    def plot_statistics_summary(self) -> None:
        """Plot analysis statistics and current state"""
        
        ax = self.axes[1]
        ax.clear()
        
        # Get current state
        state = self.analyzer.get_current_state()
        stats = state['statistics']
        
        # Create text summary
        summary_text = f"""
MARKET STRUCTURE ANALYSIS SUMMARY

Current Trend: {state['current_trend'].upper()}
Total Candles Analyzed: {state['total_candles']:,}
Trend Start Index: {state['trend_start_index']}

EVENTS SUMMARY:
• Total BOS Events: {stats['total_bos_events']}
• Total ChoCH Events: {stats['total_choch_events']}
• Bullish Periods: {stats['bullish_periods']}
• Bearish Periods: {stats['bearish_periods']}

ACTIVE LEVELS:
"""
        
        # Add active levels info
        for level_name, level_data in state['active_levels'].items():
            summary_text += f"• {level_name}: {level_data['price']:.2f}\n"
        
        if not state['active_levels']:
            summary_text += "• No active levels\n"
            
        summary_text += f"\nStatus: {'Waiting for BOS' if state['waiting_for_bos'] else 'Searching for patterns'}"
        summary_text += f"\nTotal Alerts: {state['total_alerts']}"
        
        # Display text
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', color='white', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        ax.set_title('Analysis Summary', fontweight='bold', color='white')
        ax.axis('off')

    def plot_events_timeline(self) -> None:
        """Plot timeline of BOS and ChoCH events"""
        
        ax = self.axes[2]
        ax.clear()
        
        if not self.analyzer.events:
            ax.text(0.5, 0.5, 'No events detected yet', transform=ax.transAxes,
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_title('Events Timeline', fontweight='bold', color='white')
            ax.axis('off')
            return
        
        # Prepare event data
        events_data = []
        for i, event in enumerate(self.analyzer.events):
            events_data.append({
                'index': i,
                'timestamp': event.timestamp,
                'type': event.event_type.value,
                'price': event.price,
                'description': event.description
            })
        
        if not events_data:
            return
            
        # Create timeline plot
        event_types = [e['type'] for e in events_data]
        event_indices = [e['index'] for e in events_data]
        
        # Color mapping for events
        event_colors = []
        for event_type in event_types:
            if 'bullish' in event_type:
                event_colors.append(self.colors['bullish'])
            else:
                event_colors.append(self.colors['bearish'])
        
        # Create scatter plot
        ax.scatter(event_indices, [1] * len(event_indices), c=event_colors, 
                  s=100, alpha=0.8, edgecolors='white', linewidth=1)
        
        # Add labels
        for i, event in enumerate(events_data):
            ax.annotate(event['type'].replace('_', ' ').title(), 
                       (i, 1), xytext=(0, 20), textcoords='offset points',
                       ha='center', rotation=45, color='white', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_ylim(0.5, 1.5)
        ax.set_xlim(-0.5, len(events_data) - 0.5)
        ax.set_xlabel('Event Sequence', color='white')
        ax.set_title('Market Structure Events Timeline', fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        
        # Remove y-axis ticks
        ax.set_yticks([])

    def plot_recent_alerts(self) -> None:
        """Plot recent alerts and notifications"""
        
        ax = self.axes[3]
        ax.clear()
        
        # Get recent alerts (last 10)
        recent_alerts = self.analyzer.alerts[-10:] if self.analyzer.alerts else []
        
        if not recent_alerts:
            ax.text(0.5, 0.5, 'No alerts generated yet', transform=ax.transAxes,
                   ha='center', va='center', color='white', fontsize=12)
            ax.set_title('Recent Alerts', fontweight='bold', color='white')
            ax.axis('off')
            return
        
        # Display alerts
        alert_text = "RECENT ALERTS:\n\n"
        for i, alert in enumerate(reversed(recent_alerts)):
            alert_text += f"{i+1:2d}. {alert}\n"
        
        ax.text(0.05, 0.95, alert_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', color='yellow', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8))
        
        ax.set_title('Recent Alerts & Notifications', fontweight='bold', color='white')
        ax.axis('off')

    def create_comprehensive_report(self, save_path: Optional[str] = None) -> None:
        """
        Create and display a comprehensive market structure analysis report
        
        Args:
            save_path (Optional[str]): Path to save the plot, if None will display only
        """
        # Setup the plot
        self.setup_plot(figsize=(20, 12))
        
        # Create all subplot components
        self.plot_candlestick_chart()
        self.plot_statistics_summary()
        self.plot_events_timeline()
        self.plot_recent_alerts()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
            print(f"📊 Report saved to: {save_path}")
        
        plt.show()

    def create_live_chart(self, update_interval: int = 1, window_size: int = 200) -> None:
        """
        Create a live updating chart (for real-time analysis)
        
        Args:
            update_interval (int): Update interval in seconds
            window_size (int): Number of candles to display
        """
        plt.ion()  # Turn on interactive mode
        
        fig, ax = plt.subplots(figsize=(15, 8))
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        print("🔴 Live chart started. Close the window to stop.")
        
        try:
            while True:
                ax.clear()
                
                # Get recent data
                if len(self.analyzer.candles) >= window_size:
                    start_idx = len(self.analyzer.candles) - window_size
                    end_idx = len(self.analyzer.candles)
                    
                    # Plot recent candlesticks
                    self._plot_candlesticks_simple(ax, start_idx, end_idx)
                    self._add_structure_levels(ax, start_idx, end_idx)
                    self._add_bos_lines(ax, start_idx, end_idx)
                    self._add_choch_markers(ax, start_idx, end_idx)
                    
                    ax.set_title(f'Live Market Structure - {self.analyzer.current_trend.value.upper()}', 
                                color='white', fontweight='bold')
                
                plt.pause(update_interval)
                
        except KeyboardInterrupt:
            print("🛑 Live chart stopped.")
        finally:
            plt.ioff()
            plt.close()

    def _plot_candlesticks_simple(self, ax, start_index: int, end_index: int) -> None:
        """Simplified candlestick plotting for live updates"""
        candles_slice = self.analyzer.candles[start_index:end_index]
        
        for i, candle in enumerate(candles_slice):
            color = self.colors['candle_up'] if candle['close'] > candle['open'] else self.colors['candle_down']
            
            # Draw wick
            ax.plot([i, i], [candle['low'], candle['high']], color=color, linewidth=1)
            
            # Draw body
            body_height = abs(candle['close'] - candle['open'])
            body_bottom = min(candle['open'], candle['close'])
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                           facecolor=color, alpha=0.8)
            ax.add_patch(rect)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')


def create_performance_report(analyzer: MarketStructureAnalyzer) -> Dict:
    """
    Create a detailed performance report of the market structure analysis
    
    Args:
        analyzer (MarketStructureAnalyzer): The analyzer to report on
        
    Returns:
        Dict: Performance metrics and analysis results
    """
    state = analyzer.get_current_state()
    stats = state['statistics']
    
    # Calculate additional metrics
    total_events = stats['total_bos_events'] + stats['total_choch_events']
    trend_changes = stats['total_choch_events']
    
    # Event frequency
    if state['total_candles'] > 0:
        event_frequency = (total_events / state['total_candles']) * 100
    else:
        event_frequency = 0
    
    # Trend analysis
    total_trend_periods = stats['bullish_periods'] + stats['bearish_periods']
    
    report = {
        'analysis_summary': {
            'total_candles_analyzed': state['total_candles'],
            'current_trend': state['current_trend'],
            'trend_duration': state['total_candles'] - state['trend_start_index'],
            'data_loaded': state['data_loaded']
        },
        'event_statistics': {
            'total_events': total_events,
            'bos_events': stats['total_bos_events'],
            'choch_events': stats['total_choch_events'],
            'event_frequency_percent': round(event_frequency, 4),
            'trend_changes': trend_changes
        },
        'trend_analysis': {
            'bullish_periods': stats['bullish_periods'],
            'bearish_periods': stats['bearish_periods'],
            'total_periods': total_trend_periods,
            'trend_consistency': 'High' if total_trend_periods > 0 and trend_changes / total_trend_periods < 0.3 else 'Moderate'
        },
        'active_levels': state['active_levels'],
        'system_status': {
            'waiting_for_bos': state['waiting_for_bos'],
            'total_alerts': state['total_alerts'],
            'last_analysis': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    return report