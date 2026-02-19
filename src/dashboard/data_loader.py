"""Data loading helpers used by the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.metrics import PerformanceMetrics, summarize_performance


def available_symbols(reports_dir: Path, track: str) -> list[str]:
    """Discover symbols from report artifact directories."""
    symbols: set[str] = set()
    for root in [reports_dir / "backtest" / track, reports_dir / "paper" / track]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                symbols.add(child.name.upper())
    return sorted(symbols)


def load_json(path: Path) -> dict[str, Any] | None:
    """Read JSON file if present."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_equity_curve(path: Path) -> pd.DataFrame:
    """Load equity curve CSV with safe timestamp parsing."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for col in ["timestamp", "timestamp_utc", "timestamp_report_tz"]:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    return frame


def load_trade_journal(path: Path) -> pd.DataFrame:
    """Load trade journal CSV."""
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def summarize_from_artifacts(equity_curve_path: Path, trade_journal_path: Path) -> PerformanceMetrics:
    """Compute performance metrics directly from report artifacts."""
    equity_df = load_equity_curve(equity_curve_path)
    trades_df = load_trade_journal(trade_journal_path)
    equity_series = equity_df["equity"] if "equity" in equity_df.columns else pd.Series(dtype=float)
    trade_pnls = trades_df.get("realized_pnl", pd.Series(dtype=float))
    return summarize_performance(equity_series, trade_pnls)


def read_recent_log_lines(path: Path, max_lines: int = 200) -> list[str]:
    """Return recent log lines with truncation."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-max_lines:]]
