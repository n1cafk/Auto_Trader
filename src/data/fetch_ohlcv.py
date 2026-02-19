"""OHLCV retrieval and storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ccxt  # type: ignore
import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class FetchRequest:
    """Input parameters for data fetch."""

    exchange_id: str
    symbol: str
    timeframe: str = "1h"
    limit: int = 1000
    since: int | None = None


def _build_exchange(exchange_id: str) -> Any:
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unsupported exchange id: {exchange_id}")
    return exchange_cls({"enableRateLimit": True})


def fetch_ohlcv(request: FetchRequest) -> pd.DataFrame:
    """Fetch OHLCV candles from exchange public endpoints."""
    exchange = _build_exchange(request.exchange_id)
    rows = exchange.fetch_ohlcv(
        symbol=request.symbol,
        timeframe=request.timeframe,
        since=request.since,
        limit=request.limit,
    )
    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return frame


def generate_synthetic_ohlcv(length: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Generate deterministic pseudo-market candles for local testing."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=length, freq="h", tz="UTC")
    base_returns = rng.normal(loc=0.0002, scale=0.01, size=length)
    close = 20_000 * np.exp(np.cumsum(base_returns))
    open_ = np.insert(close[:-1], 0, close[0]) * (1 + rng.normal(0.0, 0.001, size=length))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.001, 0.0007, size=length)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.001, 0.0007, size=length)))
    volume = rng.uniform(200, 500, size=length)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
    )


def save_ohlcv_csv(frame: pd.DataFrame, path: Path) -> None:
    """Persist candles to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def load_ohlcv_csv(path: Path) -> pd.DataFrame:
    """Load candle CSV into expected typed frame."""
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return frame
