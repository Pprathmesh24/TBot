from tbot.broker.oanda_client import OandaClient
c = OandaClient()
a = c.get_account()
print(f"Account: {a.account_id}")
print(f"Balance: ${a.balance:,.2f}")
print(f"NAV:     ${a.nav:,.2f}")
