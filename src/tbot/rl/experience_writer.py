"""
Convert a closed Trade into an RLExperience row.

Called from:
  - live/runner.py  (periodic sync loop, after sync_closed_trades())
  - scripts/backfill_rl_experiences.py  (one-shot backfill of historical trades)

Action encoding (Discrete(2)):
  0 = BUY / LONG entry
  1 = SELL / SHORT entry

Reward:
  net_pnl normalised by 1% of entry_price — so a trade that earns one
  "standard risk unit" (1% move in your favour) gives reward ≈ 1.0.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from tbot.db.models import RLExperience, Trade
from tbot.rl.observation_builder import build_observation

logger = logging.getLogger(__name__)

_ACTION = {"BUY": 0, "SELL": 1}


def create_experience(trade: Trade, session: Session) -> bool:
    """
    Build and persist an RLExperience from a closed Trade.

    Returns True if written, False if skipped (missing prices / candles).
    """
    if trade.exit_price is None or trade.net_pnl is None:
        logger.debug("Trade %d missing exit_price or net_pnl — skipping", trade.id)
        return False

    if trade.entry_price <= 0:
        logger.debug("Trade %d has zero entry_price — skipping", trade.id)
        return False

    state      = build_observation(trade.entry_time, session=session)
    next_state = build_observation(trade.exit_time,  session=session)

    if state is None or next_state is None:
        logger.debug(
            "Trade %d: could not build observation (not enough history in parquet) — skipping",
            trade.id,
        )
        return False

    reward_scale = trade.entry_price * 0.01   # 1% of entry price
    reward       = trade.net_pnl / reward_scale

    exp = RLExperience(
        trade_id        = trade.id,
        timestamp       = trade.entry_time,
        state_json      = json.dumps(state),
        action          = _ACTION.get(trade.side, 0),
        reward          = reward,
        next_state_json = json.dumps(next_state),
        done            = True,
    )
    session.add(exp)
    logger.info(
        "RLExperience created  trade=%d  side=%s  reward=%.4f",
        trade.id, trade.side, reward,
    )
    return True


def backfill_from_trades(session: Session, instrument: str = "XAU_USD") -> int:
    """
    One-shot: create RLExperience for every closed trade that doesn't have one yet.
    Returns count of experiences written.
    """
    from sqlalchemy import exists, not_, select

    stmt = (
        select(Trade)
        .where(Trade.instrument == instrument)
        .where(Trade.exit_time.is_not(None))
        .where(Trade.exit_price.is_not(None))
        .where(Trade.net_pnl.is_not(None))
        .where(
            not_(exists().where(RLExperience.trade_id == Trade.id))
        )
    )
    trades = session.execute(stmt).scalars().all()
    logger.info("Backfilling RL experiences: %d eligible trades", len(trades))

    written = 0
    for trade in trades:
        if create_experience(trade, session):
            written += 1

    logger.info("Backfill complete: %d experiences written", written)
    return written
