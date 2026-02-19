from __future__ import annotations

import pandas as pd

from src.evaluation.gates import GateThresholds, PromotionDecision, evaluate_promotion
from src.evaluation.metrics import summarize_performance


def test_metrics_summary_contains_expected_fields() -> None:
    equity = pd.Series([10000, 10100, 10050, 10200, 10300], dtype=float)
    trade_pnls = pd.Series([50, -20, 70], dtype=float)
    metrics = summarize_performance(equity, trade_pnls)

    assert metrics.trade_count == 3
    assert 0 <= metrics.win_rate <= 1
    assert metrics.max_drawdown >= 0


def test_gate_pass_case() -> None:
    equity = pd.Series([10000, 10100, 10300, 10500, 10800, 11000], dtype=float)
    trade_pnls = pd.Series([40] * 30, dtype=float)
    metrics = summarize_performance(equity, trade_pnls)
    thresholds = GateThresholds(
        min_sharpe=-10,  # loose for deterministic test
        min_sortino=-10,
        max_drawdown=0.5,
        min_trade_count=20,
        min_win_rate=0.5,
    )
    gate = evaluate_promotion(metrics, thresholds)
    assert gate.decision == PromotionDecision.CANDIDATE_LIVE_TRIAL
