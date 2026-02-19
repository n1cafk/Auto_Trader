"""Position sizing helpers."""

from __future__ import annotations


def size_position_notional(
    equity: float,
    atr_value: float,
    price: float,
    max_position_pct: float,
    risk_fraction_per_trade: float = 0.01,
    stop_loss_atr_multiplier: float = 2.0,
) -> float:
    """Compute position notional constrained by volatility and cap."""
    if equity <= 0 or atr_value <= 0 or price <= 0:
        return 0.0

    max_notional = equity * max_position_pct
    stop_distance = atr_value * stop_loss_atr_multiplier
    if stop_distance <= 0:
        return max_notional

    risk_budget = equity * risk_fraction_per_trade
    qty = risk_budget / stop_distance
    volatility_notional = qty * price
    return max(0.0, min(max_notional, volatility_notional))
