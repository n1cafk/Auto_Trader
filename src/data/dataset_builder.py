"""Dataset creation and time-aware splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.feature_pipeline import build_feature_frame


@dataclass(frozen=True)
class DatasetBundle:
    """Train/validation split output."""

    train: pd.DataFrame
    validation: pd.DataFrame
    feature_columns: list[str]
    target_column: str


def add_target_label(
    frame: pd.DataFrame,
    horizon: int = 1,
    target_return_threshold: float = 0.001,
) -> pd.DataFrame:
    """Attach classification labels based on forward returns."""
    labeled = frame.copy()
    labeled["future_return"] = labeled["close"].shift(-horizon) / labeled["close"] - 1.0
    labeled["target"] = (labeled["future_return"] > target_return_threshold).astype(int)
    return labeled


def split_time_series(
    frame: pd.DataFrame,
    validation_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation split."""
    split_idx = int(len(frame) * (1 - validation_ratio))
    train = frame.iloc[:split_idx].reset_index(drop=True)
    validation = frame.iloc[split_idx:].reset_index(drop=True)
    return train, validation


def build_datasets(
    candles: pd.DataFrame,
    horizon: int = 1,
    target_return_threshold: float = 0.001,
    validation_ratio: float = 0.2,
) -> DatasetBundle:
    """Build complete model-ready dataset without leakage."""
    features_frame, feature_columns = build_feature_frame(candles)
    labeled = add_target_label(
        frame=features_frame,
        horizon=horizon,
        target_return_threshold=target_return_threshold,
    )
    labeled = labeled.dropna().reset_index(drop=True)
    train, validation = split_time_series(labeled, validation_ratio=validation_ratio)
    return DatasetBundle(
        train=train,
        validation=validation,
        feature_columns=feature_columns,
        target_column="target",
    )
