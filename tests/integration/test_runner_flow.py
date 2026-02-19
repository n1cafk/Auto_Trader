from __future__ import annotations

from pathlib import Path

from src.config.settings import AppSettings, RiskLimits, TradingMode
from src.data.fetch_ohlcv import generate_synthetic_ohlcv
from src.execution.runner import run_paper_session
from src.models.train import train_from_candles


def test_runner_executes_end_to_end(tmp_path: Path) -> None:
    candles = generate_synthetic_ohlcv(length=350, seed=10)
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model_meta.json"
    train_from_candles(candles, model_path=model_path, metadata_path=metadata_path)

    settings = AppSettings(
        mode=TradingMode.PAPER,
        symbols="BTC/USDT",
        model_path=model_path,
        metadata_path=metadata_path,
        initial_cash=5000,
    )
    limits = RiskLimits()
    result = run_paper_session(
        candles=candles,
        symbol="BTC/USDT",
        settings=settings,
        risk_limits=limits,
        output_dir=tmp_path / "reports",
    )
    assert result.final_equity > 0
    assert result.equity_curve.shape[0] > 0
    assert result.trades_path.exists()
    assert result.equity_path.exists()
