from __future__ import annotations

from pathlib import Path

from src.data.fetch_ohlcv import generate_synthetic_ohlcv
from src.models.train import train_from_symbol_candles


def test_multisymbol_training_persists_track_metadata(tmp_path: Path) -> None:
    candles_by_symbol = {
        "SPY": generate_synthetic_ohlcv(length=260, seed=1),
        "QQQ": generate_synthetic_ohlcv(length=260, seed=2),
    }
    model_path = tmp_path / "intraday_model.joblib"
    metadata_path = tmp_path / "intraday_model_metadata.json"

    result = train_from_symbol_candles(
        candles_by_symbol=candles_by_symbol,
        model_path=model_path,
        metadata_path=metadata_path,
        track_name="intraday",
    )
    assert model_path.exists()
    assert metadata_path.exists()
    assert result.metadata["training_track"] == "intraday"
    assert sorted(result.metadata["training_symbols"]) == ["QQQ", "SPY"]
