"""
OANDA REST client — thin wrapper around oandapyV20.

Exposes three operations the rest of TBot needs:
  get_account()   → dict  (equity, NAV, margin)
  place_order()   → str   (OANDA order id)
  close_position() → None

All config is read from tbot.config.cfg.oanda so callers never
touch credentials directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades_ep

from tbot.config import cfg


@dataclass
class AccountSummary:
    account_id:     str
    balance:        float   # cash in account
    nav:            float   # net asset value (balance + open P&L)
    unrealized_pnl: float
    margin_used:    float
    margin_avail:   float


class OandaClient:
    """
    Authenticated OANDA REST client.

    Uses cfg.oanda.{account_id, api_token, environment} by default.
    Pass overrides in the constructor for testing.

    Args:
        account_id:   OANDA account ID (e.g. "101-001-XXXXXXX-001")
        api_token:    OANDA API token
        environment:  "practice" (paper) or "live"
    """

    def __init__(
        self,
        account_id:  str | None = None,
        api_token:   str | None = None,
        environment: str | None = None,
    ) -> None:
        self.account_id  = account_id  or cfg.oanda.account_id
        self._api_token  = api_token   or cfg.oanda.api_token
        self._env        = environment or cfg.oanda.environment

        self._client = oandapyV20.API(
            access_token=self._api_token,
            environment=self._env,
        )

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> AccountSummary:
        """Fetch live account state from OANDA."""
        req  = accounts.AccountSummary(self.account_id)
        resp = self._client.request(req)
        a    = resp["account"]
        return AccountSummary(
            account_id     = a["id"],
            balance        = float(a["balance"]),
            nav            = float(a["NAV"]),
            unrealized_pnl = float(a["unrealizedPL"]),
            margin_used    = float(a["marginUsed"]),
            margin_avail   = float(a["marginAvailable"]),
        )

    def get_equity(self) -> float:
        """Convenience: current NAV (balance + unrealized P&L)."""
        return self.get_account().nav

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        instrument: str,
        units:      float,       # positive = BUY, negative = SELL
        stop_loss:  float | None = None,
        take_profit: float | None = None,
        comment:    str = "tbot",
    ) -> str:
        """
        Place a market order with optional SL/TP attached.

        OANDA attaches SL/TP natively to the trade (stopLossOnFill /
        takeProfitOnFill), so they are automatically managed server-side.

        Args:
            instrument:  e.g. "XAU_USD"
            units:       positive for BUY, negative for SELL
            stop_loss:   absolute price for stop-loss (None = no SL)
            take_profit: absolute price for take-profit (None = no TP)
            comment:     clientComment stored on the OANDA trade

        Returns:
            OANDA order id string (e.g. "12345")
        """
        order_body: dict[str, Any] = {
            "order": {
                "type":       "MARKET",
                "instrument": instrument,
                "units":      str(units),
                "timeInForce": "FOK",           # Fill-Or-Kill — don't partial fill
                "clientExtensions": {"comment": comment},
            }
        }

        if stop_loss is not None:
            order_body["order"]["stopLossOnFill"] = {
                "price": f"{stop_loss:.5f}",
                "timeInForce": "GTC",
            }

        if take_profit is not None:
            order_body["order"]["takeProfitOnFill"] = {
                "price": f"{take_profit:.5f}",
                "timeInForce": "GTC",
            }

        req  = orders.OrderCreate(self.account_id, data=order_body)
        resp = self._client.request(req)

        # A filled market order always has orderFillTransaction in the response.
        # If it's absent the order was cancelled (e.g. FOK not fillable, market closed).
        fill = resp.get("orderFillTransaction")
        if fill is None:
            cancel = resp.get("orderCancelTransaction", {})
            reason = cancel.get("reason", "unknown")
            raise RuntimeError(
                f"Order not filled (reason: {reason}). "
                "Market may be closed or liquidity unavailable."
            )
        return str(fill.get("id", "unknown"))

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict]:
        """Return all open positions for this account."""
        req  = positions.OpenPositions(self.account_id)
        resp = self._client.request(req)
        return resp.get("positions", [])

    def close_position(self, instrument: str, side: str = "ALL") -> None:
        """
        Close an open position.

        Args:
            instrument: e.g. "XAU_USD"
            side:       "ALL", "LONG", or "SHORT"
        """
        if side == "ALL":
            data = {"longUnits": "ALL", "shortUnits": "ALL"}
        elif side == "LONG":
            data = {"longUnits": "ALL"}
        else:
            data = {"shortUnits": "ALL"}

        req = positions.PositionClose(self.account_id, instrument, data=data)
        self._client.request(req)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    def get_open_trades(self) -> list[dict]:
        """Return all open trades (individual fills, not positions)."""
        req  = trades_ep.OpenTrades(self.account_id)
        resp = self._client.request(req)
        return resp.get("trades", [])
