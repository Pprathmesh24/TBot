"""Phase 8 Chunk 1 — verify structured JSON logger."""
import json
import tempfile
from pathlib import Path

from tbot.monitoring.logger import configure_logging, get_logger

tmp = Path(tempfile.mktemp(suffix=".log"))
configure_logging(log_file=tmp, level="DEBUG")

log = get_logger("tbot.test")
log.info("trade_signal",  instrument="XAU_USD", confidence=0.72, side="BUY")
log.warning("risk_blocked", reason="daily_loss_cap", equity=99500.0)
log.debug("candle_received", open=2350.1, close=2351.5)

lines = [l for l in tmp.read_text().strip().split("\n") if l.strip()]
assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"

for i, line in enumerate(lines):
    obj = json.loads(line)
    assert "event" in obj,     f"Line {i} missing 'event': {obj}"
    assert "timestamp" in obj, f"Line {i} missing 'timestamp': {obj}"
    assert "level" in obj,     f"Line {i} missing 'level': {obj}"
    assert "logger" in obj,    f"Line {i} missing 'logger': {obj}"

print(f"✓ {len(lines)} JSON lines written to temp log")
print(f"  Sample: {json.loads(lines[0])}")
tmp.unlink()
print("✓ Phase 8 Chunk 1 logger test passed")
