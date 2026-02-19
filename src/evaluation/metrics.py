"""Performance metric calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    expectancy: float
    trade_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _annualization_factor(periods_per_year: int = 24 * 365) -> float:
    return float(np.sqrt(periods_per_year))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve - running_peak) / running_peak.replace(0, np.nan)
    return float(drawdown.min()) if not drawdown.empty else 0.0


def compute_sharpe(returns: pd.Series) -> float:
    if returns.std(ddof=0) == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0) * _annualization_factor())


def compute_sortino(returns: pd.Series) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0) if len(downside) > 0 else 0.0
    if downside_std == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / downside_std * _annualization_factor())


def compute_trade_stats(trade_pnls: pd.Series) -> tuple[float, float, int]:
    if trade_pnls.empty:
        return 0.0, 0.0, 0
    win_rate = float((trade_pnls > 0).sum() / len(trade_pnls))
    expectancy = float(trade_pnls.mean())
    return win_rate, expectancy, int(len(trade_pnls))


def summarize_performance(equity_curve: pd.Series, trade_pnls: pd.Series) -> PerformanceMetrics:
    """Calculate core strategy performance statistics."""
    if equity_curve.empty:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    returns = equity_curve.pct_change().dropna()
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)
    max_drawdown = abs(compute_max_drawdown(equity_curve))
    win_rate, expectancy, trade_count = compute_trade_stats(trade_pnls)
    return PerformanceMetrics(
        total_return=total_return,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        expectancy=expectancy,
        trade_count=trade_count,
    )
