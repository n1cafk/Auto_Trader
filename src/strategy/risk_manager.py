"""Portfolio risk controls and hard guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.config.settings import RiskLimits


@dataclass
class RiskState:
    """Mutable state for drawdown and daily loss checks."""

    peak_equity: float
    day_start_equity: float
    current_day: str
    emergency_stop: bool = False


class RiskManager:
    """Apply risk constraints before orders are accepted."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits
        self.state: RiskState | None = None

    def _day_key(self, timestamp: datetime) -> str:
        return timestamp.date().isoformat()

    def update_equity(self, equity: float, timestamp: datetime) -> None:
        """Refresh rolling risk state with latest account equity."""
        if self.state is None:
            self.state = RiskState(
                peak_equity=equity,
                day_start_equity=equity,
                current_day=self._day_key(timestamp),
            )
            return

        day = self._day_key(timestamp)
        if day != self.state.current_day:
            self.state.current_day = day
            self.state.day_start_equity = equity

        self.state.peak_equity = max(self.state.peak_equity, equity)
        drawdown = (self.state.peak_equity - equity) / max(self.state.peak_equity, 1e-9)
        if drawdown >= self.limits.max_drawdown_pct:
            self.state.emergency_stop = True

    def daily_loss_pct(self, equity: float) -> float:
        """Current day's loss as a percentage of day-start equity."""
        if self.state is None or self.state.day_start_equity <= 0:
            return 0.0
        return (equity - self.state.day_start_equity) / self.state.day_start_equity

    def can_place_order(
        self,
        side: str,
        order_notional: float,
        equity: float,
        symbol_exposure: float,
        total_exposure: float,
        open_positions: int,
        timestamp: datetime,
    ) -> tuple[bool, str]:
        """Evaluate whether an order can be accepted."""
        self.update_equity(equity=equity, timestamp=timestamp)
        if self.state and self.state.emergency_stop and side.lower() == "buy":
            return False, "Emergency stop active due to max drawdown breach."

        if self.daily_loss_pct(equity) <= -self.limits.max_daily_loss_pct and side.lower() == "buy":
            return False, "Daily loss limit breached."

        if side.lower() == "sell":
            return True, "Sell orders are always allowed for de-risking."

        max_symbol_notional = equity * self.limits.max_position_pct
        if symbol_exposure + order_notional > max_symbol_notional:
            return False, "Order breaches per-symbol position cap."

        max_total_exposure = equity * self.limits.max_portfolio_exposure_pct
        if total_exposure + order_notional > max_total_exposure:
            return False, "Order breaches portfolio exposure cap."

        if open_positions >= self.limits.max_open_positions and symbol_exposure <= 0:
            return False, "Order breaches max open positions limit."

        return True, "Order approved."
