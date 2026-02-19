from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dashboard.data_loader import (
    available_symbols,
    load_json,
    read_recent_log_lines,
    summarize_from_artifacts,
)


def test_available_symbols_discovers_from_report_tree(tmp_path: Path) -> None:
    (tmp_path / "backtest" / "long_term" / "SPY").mkdir(parents=True)
    (tmp_path / "paper" / "long_term" / "QQQ").mkdir(parents=True)
    symbols = available_symbols(tmp_path, "long_term")
    assert symbols == ["QQQ", "SPY"]


def test_load_json_reads_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    payload = load_json(path)
    assert payload == {"ok": True}


def test_summarize_from_artifacts_builds_metrics(tmp_path: Path) -> None:
    equity_path = tmp_path / "paper_equity_curve.csv"
    trades_path = tmp_path / "paper_trades.csv"
    pd.DataFrame({"equity": [10000, 10100, 10200]}).to_csv(equity_path, index=False)
    pd.DataFrame({"realized_pnl": [10, -5, 12]}).to_csv(trades_path, index=False)

    metrics = summarize_from_artifacts(equity_path, trades_path)
    assert metrics.trade_count == 3
    assert metrics.total_return > 0


def test_read_recent_log_lines_returns_tail(tmp_path: Path) -> None:
    path = tmp_path / "paper_runner.log"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    lines = read_recent_log_lines(path, max_lines=2)
    assert lines == ["b", "c"]
