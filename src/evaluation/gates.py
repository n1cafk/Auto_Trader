"""Promotion gates from backtest/paper performance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.evaluation.metrics import PerformanceMetrics


class PromotionDecision(str, Enum):
    RETRAIN_REQUIRED = "retrain_required"
    PAPER_CONTINUE = "paper_continue"
    CANDIDATE_LIVE_TRIAL = "candidate_live_trial"


@dataclass(frozen=True)
class GateThresholds:
    min_sharpe: float = 0.8
    min_sortino: float = 1.0
    max_drawdown: float = 0.12
    min_trade_count: int = 25
    min_win_rate: float = 0.45


@dataclass(frozen=True)
class GateResult:
    decision: PromotionDecision
    passed: bool
    reasons: list[str]


def evaluate_promotion(metrics: PerformanceMetrics, thresholds: GateThresholds) -> GateResult:
    """Evaluate objective quality gates for promotion decisions."""
    reasons: list[str] = []
    if metrics.sharpe < thresholds.min_sharpe:
        reasons.append(f"Sharpe below threshold ({metrics.sharpe:.3f} < {thresholds.min_sharpe:.3f})")
    if metrics.sortino < thresholds.min_sortino:
        reasons.append(
            f"Sortino below threshold ({metrics.sortino:.3f} < {thresholds.min_sortino:.3f})",
        )
    if metrics.max_drawdown > thresholds.max_drawdown:
        reasons.append(
            f"Max drawdown above threshold ({metrics.max_drawdown:.3f} > {thresholds.max_drawdown:.3f})",
        )
    if metrics.trade_count < thresholds.min_trade_count:
        reasons.append(
            f"Trade count below threshold ({metrics.trade_count} < {thresholds.min_trade_count})",
        )
    if metrics.win_rate < thresholds.min_win_rate:
        reasons.append(
            f"Win rate below threshold ({metrics.win_rate:.3f} < {thresholds.min_win_rate:.3f})",
        )

    if not reasons:
        return GateResult(
            decision=PromotionDecision.CANDIDATE_LIVE_TRIAL,
            passed=True,
            reasons=["All quantitative gates passed."],
        )

    high_risk_failures = [
        reason for reason in reasons if "drawdown" in reason.lower() or "sharpe" in reason.lower()
    ]
    if high_risk_failures:
        return GateResult(
            decision=PromotionDecision.RETRAIN_REQUIRED,
            passed=False,
            reasons=reasons,
        )
    return GateResult(
        decision=PromotionDecision.PAPER_CONTINUE,
        passed=False,
        reasons=reasons,
    )
