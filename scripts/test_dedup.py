"""
Verification for chunk 1: the runner's _live_signal_exists() dedup guard.

What this proves:
  1. With an empty signals table, the check returns False (allow).
  2. After writing one live signal, the check for the SAME bar returns True (block).
  3. A backtest signal (backtest_run_id != NULL) at the same ts does NOT count
     as a live duplicate — live and backtest never collide.
  4. A signal at a DIFFERENT bar timestamp does not collide.

Run:
    .venv/bin/python scripts/test_dedup.py
Expected: "ALL DEDUP CHECKS PASSED" at the bottom and exit code 0.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

# Point SQLAlchemy at a throwaway DB before any tbot imports run
_TMP_DB = Path(tempfile.mkdtemp()) / "dedup_test.sqlite"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tbot.db.models import BacktestRun, Signal as SignalModel  # noqa: E402
from tbot.db.session import get_session, init_db  # noqa: E402
from tbot.live.runner import LiveRunner  # noqa: E402


def main() -> int:
    init_db()
    ts1 = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 5, 17, 12, 5, 0, tzinfo=timezone.utc)

    fake_client = MagicMock()
    fake_client.account_id = "TEST"
    runner = LiveRunner(client=fake_client)

    sig_at_ts1 = {"timestamp": ts1, "type": "BUY", "price": 1.0}
    sig_at_ts2 = {"timestamp": ts2, "type": "SELL", "price": 1.0}

    with get_session() as s:
        assert runner._live_signal_exists(sig_at_ts1, s) is False, \
            "Case 1 FAIL — empty DB should not have a duplicate"
        print("Case 1 OK — empty DB returns False")

        s.add(SignalModel(
            instrument="XAU_USD", granularity="M5", timestamp=ts1,
            side="BUY", entry_price=1.0, stop_loss=0.9, take_profit=1.1,
            source_strategy="smc_v2_live", backtest_run_id=None,
        ))
        s.flush()

        assert runner._live_signal_exists(sig_at_ts1, s) is True, \
            "Case 2 FAIL — same ts after insert should return True"
        print("Case 2 OK — after live insert, same bar returns True")

        run = BacktestRun(
            created_at=ts1, strategy="test",
            instrument="XAU_USD", granularity="M5",
            start_date=ts1, end_date=ts2,
            config_json="{}",
        )
        s.add(run)
        s.flush()
        s.add(SignalModel(
            instrument="XAU_USD", granularity="M5", timestamp=ts2,
            side="BUY", entry_price=1.0, stop_loss=0.9, take_profit=1.1,
            source_strategy="backtest", backtest_run_id=run.id,
        ))
        s.flush()

        assert runner._live_signal_exists(sig_at_ts2, s) is False, \
            "Case 3 FAIL — backtest row should NOT count as live dup"
        print("Case 3 OK — backtest row at ts2 ignored")

        ts3 = datetime(2026, 5, 17, 13, 0, 0, tzinfo=timezone.utc)
        sig_at_ts3 = {"timestamp": ts3, "type": "BUY", "price": 1.0}
        assert runner._live_signal_exists(sig_at_ts3, s) is False, \
            "Case 4 FAIL — unrelated bar should not collide"
        print("Case 4 OK — different bar returns False")

    print()
    print("ALL DEDUP CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
