from __future__ import annotations

import pytest

from src.config.settings import AppSettings, LIVE_ACK_TOKEN, TradingMode


def test_defaults_to_paper_mode() -> None:
    settings = AppSettings(
        mode=TradingMode.PAPER,
        symbols="BTC/USDT",
    )
    assert settings.mode == TradingMode.PAPER
    assert settings.live_trading_enabled is False


def test_live_mode_requires_acknowledgement() -> None:
    with pytest.raises(ValueError):
        AppSettings(
            mode=TradingMode.LIVE,
            live_trading_enabled=True,
            live_acknowledgement="wrong-token",
            symbols="BTC/USDT",
        )


def test_live_mode_with_valid_token_is_allowed() -> None:
    settings = AppSettings(
        mode=TradingMode.LIVE,
        live_trading_enabled=True,
        live_acknowledgement=LIVE_ACK_TOKEN,
        symbols="BTC/USDT",
    )
    assert settings.mode == TradingMode.LIVE
