"""Signal generation from model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Signal(str, Enum):
    LONG = "long"
    FLAT = "flat"
    SHORT = "short"


@dataclass(frozen=True)
class SignalConfig:
    """Probability thresholds to map model outputs to actions."""

    long_threshold: float = 0.55
    short_threshold: float = 0.45
    allow_short: bool = False


def probability_to_signal(probability_up: float, config: SignalConfig) -> Signal:
    """Convert up-move probability to directional signal."""
    if probability_up >= config.long_threshold:
        return Signal.LONG
    if config.allow_short and probability_up <= config.short_threshold:
        return Signal.SHORT
    return Signal.FLAT
