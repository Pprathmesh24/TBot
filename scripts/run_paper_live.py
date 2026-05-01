"""
Start the live paper trading loop.

Usage:
    .venv/bin/python scripts/run_paper_live.py

Stops cleanly on Ctrl+C.
"""
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

from tbot.live.runner import LiveRunner

runner = LiveRunner.build()
runner.start()
