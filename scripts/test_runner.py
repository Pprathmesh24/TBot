from tbot.live.runner import LiveRunner, _ts_matches
from datetime import datetime, timezone

print("LiveRunner import OK")

# Test _ts_matches helper
t = datetime(2024, 1, 1, 14, 5, 0, tzinfo=timezone.utc)
assert _ts_matches("2024-01-01T14:05:33Z", t)
assert not _ts_matches("2024-01-01T14:10:00Z", t)
print("_ts_matches OK")
