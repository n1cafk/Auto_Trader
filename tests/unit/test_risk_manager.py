from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.config.settings import RiskLimits
from src.strategy.risk_manager import RiskManager


def test_daily_loss_limit_blocks_new_buys() -> None:
    limits = RiskLimits(max_daily_loss_pct=0.02)
    manager = RiskManager(limits)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    manager.update_equity(10_000, start)
    manager.update_equity(9_700, start + timedelta(hours=2))

    ok, reason = manager.can_place_order(
        side="buy",
        order_notional=100,
        equity=9_700,
        symbol_exposure=0,
        total_exposure=0,
        open_positions=0,
        timestamp=start + timedelta(hours=3),
    )
    assert ok is False
    assert "Daily loss limit" in reason


def test_sell_allowed_during_emergency_stop() -> None:
    limits = RiskLimits(max_drawdown_pct=0.01)
    manager = RiskManager(limits)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    manager.update_equity(10_000, start)
    manager.update_equity(9_000, start + timedelta(hours=1))

    ok, reason = manager.can_place_order(
        side="sell",
        order_notional=500,
        equity=9_000,
        symbol_exposure=500,
        total_exposure=500,
        open_positions=1,
        timestamp=start + timedelta(hours=1),
    )
    assert ok is True
    assert "always allowed" in reason
