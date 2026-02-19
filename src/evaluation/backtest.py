"""Backtesting orchestration and report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.settings import AppSettings, RiskLimits
from src.evaluation.gates import GateThresholds, evaluate_promotion
from src.evaluation.metrics import PerformanceMetrics, summarize_performance
from src.execution.runner import run_paper_session


@dataclass(frozen=True)
class BacktestReport:
    metrics: PerformanceMetrics
    gate_decision: str
    gate_reasons: list[str]
    report_path: Path


def run_backtest(
    candles: pd.DataFrame,
    symbol: str,
    settings: AppSettings,
    risk_limits: RiskLimits,
    report_dir: Path,
    model_path: Path | None = None,
    metadata_path: Path | None = None,
    track_name: str = "long_term",
    thresholds: GateThresholds | None = None,
) -> BacktestReport:
    """Run historical simulation and emit report JSON."""
    run_result = run_paper_session(
        candles=candles,
        symbol=symbol,
        settings=settings,
        risk_limits=risk_limits,
        output_dir=report_dir,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    metrics = summarize_performance(
        equity_curve=run_result.equity_curve,
        trade_pnls=run_result.trade_pnls,
    )
    gate_result = evaluate_promotion(metrics, thresholds or GateThresholds())

    report = {
        "track": track_name,
        "metrics": metrics.to_dict(),
        "promotion_gate": {
            "decision": gate_result.decision.value,
            "passed": gate_result.passed,
            "reasons": gate_result.reasons,
        },
        "artifacts": {
            "equity_curve_csv": str(run_result.equity_path),
            "trade_journal_csv": str(run_result.trades_path),
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "backtest_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    return BacktestReport(
        metrics=metrics,
        gate_decision=gate_result.decision.value,
        gate_reasons=gate_result.reasons,
        report_path=report_path,
    )
