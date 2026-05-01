"""
AITradingAgentV2 — drop-in replacement for AITradingAgent that uses
EnrichedMarketStructureAnalyzer (FVG + OB + Liquidity Sweeps).

Interface is intentionally identical to the V1 agent so the backtest
engine can swap it in without changes:
    agent = AITradingAgentV2()
    agent.analyzer.candles = candles_to_dict_list(df)   # not used by V2, kept for compat
    agent.run_on_df(df)
    signals = agent.current_signals
"""

from __future__ import annotations

from typing import List

import pandas as pd

from tbot.core.smc.structure_v2 import EnrichedMarketStructureAnalyzer


class AITradingAgentV2:
    """
    SMC-enriched agent.  Produces BUY/SELL signals from three triggers:
        1. FVG retracement
        2. Order Block retracement
        3. Liquidity sweep (stop hunt)
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence  = min_confidence
        self.current_signals: List[dict] = []
        # kept for interface compatibility with engine.py
        self.analyzer = _CompatStub()

    def run_on_df(self, df: pd.DataFrame) -> None:
        """Run the enriched analyzer on a full candle DataFrame."""
        self.current_signals = []
        analyzer = EnrichedMarketStructureAnalyzer(df)
        raw = analyzer.get_entry_signals()
        self.current_signals = [
            s for s in raw if s.get("confidence", 0) >= self.min_confidence
        ]


class _CompatStub:
    """Placeholder so engine.py can still set agent.analyzer.candles without crashing."""
    def __init__(self):
        self.candles = []
