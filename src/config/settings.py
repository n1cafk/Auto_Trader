"""Runtime settings and risk limits loading."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LIVE_ACK_TOKEN = "I_UNDERSTAND_LIVE_TRADING_RISK"


class TradingMode(str, Enum):
    """Supported runtime modes."""

    PAPER = "paper"
    LIVE = "live"


class RiskLimits(BaseModel):
    """Risk guardrails used by portfolio/risk management."""

    max_position_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    max_portfolio_exposure_pct: float = Field(default=0.8, ge=0.0, le=1.0)
    max_daily_loss_pct: float = Field(default=0.03, ge=0.0, le=1.0)
    max_drawdown_pct: float = Field(default=0.1, ge=0.0, le=1.0)
    stop_loss_atr_multiplier: float = Field(default=2.0, gt=0.0)
    max_open_positions: int = Field(default=3, ge=1)


class AppSettings(BaseSettings):
    """Application settings loaded from env variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    mode: TradingMode = TradingMode.PAPER
    live_trading_enabled: bool = False
    live_acknowledgement: str = ""

    exchange_name: str = "binance"
    symbols: str = "BTC/USDT,ETH/USDT"
    timeframe: str = "1h"
    data_limit: int = Field(default=1000, ge=100, le=20000)

    initial_cash: float = Field(default=10_000.0, gt=0.0)
    fee_rate: float = Field(default=0.001, ge=0.0, le=0.01)
    slippage_rate: float = Field(default=0.0005, ge=0.0, le=0.01)

    model_probability_threshold: float = Field(default=0.55, ge=0.5, le=0.95)
    target_return_threshold: float = Field(default=0.001)
    prediction_horizon: int = Field(default=1, ge=1, le=48)
    run_interval_seconds: int = Field(default=60, ge=5, le=3600)

    model_path: Path = Path("models/model.joblib")
    metadata_path: Path = Path("models/model_metadata.json")
    risk_limits_path: Path = Path("src/config/risk_limits.yaml")
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    logs_dir: Path = Path("logs")

    @property
    def symbol_list(self) -> list[str]:
        """Parse comma-separated symbols."""
        return [s.strip() for s in self.symbols.split(",") if s.strip()]

    @model_validator(mode="after")
    def _enforce_live_safety(self) -> "AppSettings":
        """Block live mode unless explicit hard acknowledgements exist."""
        if self.mode == TradingMode.LIVE:
            if not self.live_trading_enabled:
                raise ValueError(
                    "Live mode requested but LIVE_TRADING_ENABLED is not true.",
                )
            if self.live_acknowledgement != LIVE_ACK_TOKEN:
                raise ValueError(
                    "Live mode requested but LIVE_ACKNOWLEDGEMENT token is missing/invalid.",
                )
        return self


def load_risk_limits(path: Path) -> RiskLimits:
    """Load risk limits from YAML."""
    if not path.exists():
        return RiskLimits()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return RiskLimits.model_validate(raw)


def load_settings() -> tuple[AppSettings, RiskLimits]:
    """Load application settings and risk limits together."""
    settings = AppSettings()
    risk_limits = load_risk_limits(settings.risk_limits_path)
    return settings, risk_limits


def mode_banner(mode: Literal["paper", "live"] | TradingMode) -> str:
    """Human readable mode banner."""
    mode_value = mode.value if isinstance(mode, TradingMode) else mode
    if mode_value == TradingMode.PAPER.value:
        return "PAPER TRADING MODE (safe default)"
    return "LIVE TRADING MODE (guardrails acknowledged)"
