from __future__ import annotations

import pytest

from src.config.settings import AppSettings, LIVE_ACK_TOKEN, TradingMode


def test_defaults_to_paper_mode() -> None:
    settings = AppSettings(
        mode=TradingMode.PAPER,
        safe_live_symbols="SPY,VTI,QQQ",
    )
    assert settings.mode == TradingMode.PAPER
    assert settings.live_trading_enabled is False
    assert settings.safe_live_symbol_list == ["SPY", "VTI", "QQQ"]


def test_live_mode_requires_acknowledgement() -> None:
    with pytest.raises(ValueError):
        AppSettings(
            mode=TradingMode.LIVE,
            live_trading_enabled=True,
            live_acknowledgement="wrong-token",
            safe_live_symbols="SPY",
        )


def test_live_mode_with_valid_token_is_allowed() -> None:
    settings = AppSettings(
        mode=TradingMode.LIVE,
        live_trading_enabled=True,
        live_acknowledgement=LIVE_ACK_TOKEN,
        safe_live_symbols="SPY",
    )
    assert settings.mode == TradingMode.LIVE
