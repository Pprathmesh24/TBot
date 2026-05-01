from tbot.broker.executor import Executor
from tbot.risk.manager import RiskManager
print("Executor import OK")

rm = RiskManager(risk_pct=0.01)
units = rm.position_size(equity=10_000, entry=2350.0, stop=2345.0)
print(f"Position size for $10k equity, 5-pip stop: {units:.2f} units")
