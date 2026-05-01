"""
Places a 1-unit BUY on XAU_USD paper account, prints the order ID,
lists open trades to confirm it's there, then closes the position.
"""
from tbot.broker.oanda_client import OandaClient

client = OandaClient()

# 1. Current price
account = client.get_account()
print(f"NAV before order: ${account.nav:,.2f}")

# 2. Place 1-unit BUY (no SL/TP so it stays open for inspection)
print("\nPlacing 1-unit BUY on XAU_USD ...")
order_id = client.place_market_order(
    instrument="XAU_USD",
    units=1.0,
)
print(f"Order placed → OANDA id: {order_id}")

# 3. List open trades to confirm
import time; time.sleep(1)
open_trades = client.get_open_trades()
print(f"\nOpen trades ({len(open_trades)} total):")
for t in open_trades:
    print(f"  id={t['id']}  instrument={t['instrument']}  units={t['currentUnits']}  unrealizedPL={t['unrealizedPL']}")

# 4. Close the position
print("\nClosing XAU_USD position ...")
client.close_position("XAU_USD")
print("Position closed.")

# 5. Confirm no open trades
time.sleep(1)
open_trades = client.get_open_trades()
print(f"Open trades after close: {len(open_trades)}")
