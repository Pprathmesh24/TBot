from tbot.broker.stream import CandleBuilder, _bar_start, _mid
from datetime import datetime, timezone

t = datetime(2024, 1, 1, 14, 7, 33, tzinfo=timezone.utc)
print("bar start:", _bar_start(t))  # expect 14:05:00

b = CandleBuilder(_bar_start(t), 2350.0)
b.update(2355.0)
b.update(2348.0)
b.update(2352.0)
print("candle:", b.to_dict())
print("import OK")
