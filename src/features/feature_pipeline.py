"""Feature engineering pipeline with leakage controls."""

from __future__ import annotations

import pandas as pd

from src.features.indicators import atr, macd, rsi


def build_feature_frame(candles: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Generate deterministic model features from OHLCV candles."""
    frame = candles.copy().sort_values("timestamp").reset_index(drop=True)

    frame["return_1"] = frame["close"].pct_change(1)
    frame["return_3"] = frame["close"].pct_change(3)
    frame["return_12"] = frame["close"].pct_change(12)
    frame["volatility_12"] = frame["return_1"].rolling(12).std()

    frame["sma_fast"] = frame["close"].rolling(10).mean()
    frame["sma_slow"] = frame["close"].rolling(30).mean()
    frame["sma_ratio"] = frame["sma_fast"] / frame["sma_slow"]

    frame["rsi_14"] = rsi(frame["close"], period=14)
    macd_line, macd_signal, macd_hist = macd(frame["close"])
    frame["macd_line"] = macd_line
    frame["macd_signal"] = macd_signal
    frame["macd_hist"] = macd_hist
    frame["atr_14"] = atr(frame["high"], frame["low"], frame["close"], period=14)

    frame["volume_mean_20"] = frame["volume"].rolling(20).mean()
    frame["volume_ratio"] = frame["volume"] / frame["volume_mean_20"]

    feature_columns = [
        "return_1",
        "return_3",
        "return_12",
        "volatility_12",
        "sma_ratio",
        "rsi_14",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "volume_ratio",
    ]

    # Shift model features by one full candle to avoid future leakage.
    frame[feature_columns] = frame[feature_columns].shift(1)
    return frame, feature_columns
