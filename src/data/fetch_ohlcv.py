"""OHLCV retrieval and storage helpers for US equities and synthetic data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class FetchRequest:
    """Input parameters for data fetch."""

    symbol: str
    provider: Literal["yfinance", "alpaca", "synthetic"] = "yfinance"
    timeframe: str = "1d"
    limit: int = 1000
    market_timezone: str = "America/New_York"
    data_feed: str = "iex"


def _period_for_request(timeframe: str, limit: int) -> str:
    if timeframe.endswith("m"):
        if limit <= 300:
            return "30d"
        return "60d"
    if timeframe.endswith("h"):
        if limit <= 600:
            return "2y"
        return "5y"
    return "10y"


def _filter_regular_session(frame: pd.DataFrame, market_timezone: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    local_ts = frame["timestamp"].dt.tz_convert(market_timezone)
    minutes = local_ts.dt.hour * 60 + local_ts.dt.minute
    market_open = 9 * 60 + 30
    market_close = 16 * 60
    in_session = (minutes >= market_open) & (minutes <= market_close) & (local_ts.dt.weekday < 5)
    return frame.loc[in_session].reset_index(drop=True)


def _fetch_with_yfinance(request: FetchRequest) -> pd.DataFrame:
    period = _period_for_request(request.timeframe, request.limit)
    frame = yf.download(
        tickers=request.symbol,
        period=period,
        interval=request.timeframe,
        auto_adjust=False,
        progress=False,
        prepost=False,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for {request.symbol} ({request.timeframe}).")
    frame = frame.reset_index().rename(
        columns={
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame[OHLCV_COLUMNS].sort_values("timestamp").tail(request.limit).reset_index(drop=True)
    if request.timeframe.endswith(("m", "h")):
        frame = _filter_regular_session(frame, request.market_timezone)
    return frame


def _fetch_with_alpaca(request: FetchRequest) -> pd.DataFrame:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    now = datetime.now()
    start = now - timedelta(days=3650)
    timeframe_map: dict[str, TimeFrame] = {
        "1d": TimeFrame(1, TimeFrameUnit.Day),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
    }
    timeframe = timeframe_map.get(request.timeframe)
    if timeframe is None:
        raise ValueError(f"Unsupported timeframe for Alpaca provider: {request.timeframe}")

    client = StockHistoricalDataClient()
    bars_request = StockBarsRequest(
        symbol_or_symbols=[request.symbol],
        timeframe=timeframe,
        start=start,
        end=now,
        feed=request.data_feed,
    )
    bars = client.get_stock_bars(bars_request)
    if request.symbol not in bars.data:
        raise ValueError(f"No Alpaca data returned for {request.symbol}")
    rows = [
        {
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars.data[request.symbol]
    ]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame[OHLCV_COLUMNS].sort_values("timestamp").tail(request.limit).reset_index(drop=True)
    if request.timeframe.endswith(("m", "h")):
        frame = _filter_regular_session(frame, request.market_timezone)
    return frame


def fetch_ohlcv(request: FetchRequest) -> pd.DataFrame:
    """Fetch OHLCV candles from a configured provider."""
    if request.provider == "synthetic":
        return generate_synthetic_ohlcv(length=request.limit)
    if request.provider == "alpaca":
        frame = _fetch_with_alpaca(request)
    elif request.provider == "yfinance":
        frame = _fetch_with_yfinance(request)
    else:
        raise ValueError(f"Unsupported provider: {request.provider}")

    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return frame


def generate_synthetic_ohlcv(length: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Generate deterministic pseudo-market candles for local testing."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=length, freq="h", tz="UTC")
    base_returns = rng.normal(loc=0.0002, scale=0.01, size=length)
    close = 100 * np.exp(np.cumsum(base_returns))
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
