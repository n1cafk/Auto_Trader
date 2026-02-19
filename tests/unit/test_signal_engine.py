from __future__ import annotations

from src.strategy.signal_engine import Signal, SignalConfig, probability_to_signal


def test_signal_threshold_edges() -> None:
    cfg = SignalConfig(long_threshold=0.60, short_threshold=0.40, allow_short=True)
    assert probability_to_signal(0.61, cfg) == Signal.LONG
    assert probability_to_signal(0.39, cfg) == Signal.SHORT
    assert probability_to_signal(0.50, cfg) == Signal.FLAT
