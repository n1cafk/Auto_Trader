"""Paper broker for simulated execution with fees/slippage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class Position:
    """Simple long spot position."""

    quantity: float = 0.0
    average_entry: float = 0.0


class PaperBroker:
    """Long-only paper execution simulator."""

    def __init__(
        self,
        initial_cash: float,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
    ) -> None:
        self.cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.positions: dict[str, Position] = {}
        self.trade_journal: list[dict[str, float | str]] = []
        self.realized_pnls: list[float] = []

    def _position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position()
        return self.positions[symbol]

    def symbol_exposure(self, symbol: str, mark_price: float) -> float:
        position = self._position(symbol)
        return position.quantity * mark_price

    def total_exposure(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.average_entry)
            total += position.quantity * price
        return total

    def open_positions(self) -> int:
        return sum(1 for pos in self.positions.values() if pos.quantity > 0)

    def place_order(
        self,
        symbol: str,
        side: str,
        price: float,
        notional: float,
        timestamp: datetime,
    ) -> bool:
        """Execute order. Returns True if accepted."""
        if notional <= 0 or price <= 0:
            return False

        side = side.lower()
        position = self._position(symbol)

        if side == "buy":
            fill_price = price * (1 + self.slippage_rate)
            fee = notional * self.fee_rate
            gross_cost = notional + fee
            if gross_cost > self.cash:
                return False
            qty = notional / fill_price
            new_qty = position.quantity + qty
            if new_qty <= 0:
                return False
            position.average_entry = (
                (position.average_entry * position.quantity + fill_price * qty) / new_qty
                if position.quantity > 0
                else fill_price
            )
            position.quantity = new_qty
            self.cash -= gross_cost
            self.trade_journal.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "fill_price": fill_price,
                    "notional": notional,
                    "quantity": qty,
                    "fee": fee,
                    "realized_pnl": 0.0,
                },
            )
            return True

        if side == "sell":
            if position.quantity <= 0:
                return False
            fill_price = price * (1 - self.slippage_rate)
            requested_qty = notional / fill_price
            qty = min(requested_qty, position.quantity)
            realized_notional = qty * fill_price
            fee = realized_notional * self.fee_rate
            gross_entry_notional = qty * position.average_entry
            realized_pnl = realized_notional - gross_entry_notional - fee
            position.quantity -= qty
            if position.quantity <= 1e-12:
                position.quantity = 0.0
                position.average_entry = 0.0
            self.cash += realized_notional - fee
            self.realized_pnls.append(realized_pnl)
            self.trade_journal.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "side": side,
                    "fill_price": fill_price,
                    "notional": realized_notional,
                    "quantity": qty,
                    "fee": fee,
                    "realized_pnl": realized_pnl,
                },
            )
            return True
        return False

    def portfolio_value(self, prices: dict[str, float]) -> float:
        return self.cash + self.total_exposure(prices)

    def export_trade_journal(self, path: Path) -> None:
        """Save executed trades."""
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.trade_journal).to_csv(path, index=False)
