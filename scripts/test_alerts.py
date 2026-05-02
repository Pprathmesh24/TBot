"""Phase 8 Chunk 2 — verify Slack alerts module."""
from tbot.monitoring.logger import configure_logging
from tbot.monitoring.alerts import (
    alert_trade_filled,
    alert_circuit_breaker,
    alert_daily_loss_cap,
    alert_stream_reconnect,
    alert_critical_error,
)

configure_logging(level="DEBUG")

print("Sending test alerts to Slack...")
alert_trade_filled(side="BUY", entry=2352.10, units=20, confidence=0.74, oanda_id="99999")
print("  ✓ trade_filled sent")

alert_circuit_breaker(consecutive_losses=3, cooldown_hours=24)
print("  ✓ circuit_breaker sent")

alert_daily_loss_cap(daily_pnl=-3050.0, cap_pct=3.0)
print("  ✓ daily_loss_cap sent")

alert_stream_reconnect(attempt=2)
print("  ✓ stream_reconnect sent")

alert_critical_error("ConnectionResetError: [Errno 54] Connection reset by peer")
print("  ✓ critical_error sent")

print("\n✓ All 5 alerts sent — check your Slack channel now.")
