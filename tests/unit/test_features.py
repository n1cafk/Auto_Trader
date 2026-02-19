from __future__ import annotations

import numpy as np

from src.data.dataset_builder import build_datasets
from src.data.fetch_ohlcv import generate_synthetic_ohlcv
from src.features.feature_pipeline import build_feature_frame


def test_feature_pipeline_shifts_features_to_avoid_leakage() -> None:
    candles = generate_synthetic_ohlcv(length=120, seed=1)
    frame, _ = build_feature_frame(candles)

    raw_return = candles["close"].pct_change(1)
    # feature at i should equal raw_return from i-1 because pipeline shifts by 1.
    idx = 30
    assert np.isclose(frame.loc[idx, "return_1"], raw_return.loc[idx - 1], equal_nan=False)


def test_dataset_builder_outputs_non_nan_training_features() -> None:
    candles = generate_synthetic_ohlcv(length=240, seed=3)
    bundle = build_datasets(candles=candles, horizon=1)

    assert len(bundle.train) > 0
    assert len(bundle.validation) > 0
    assert not bundle.train[bundle.feature_columns].isna().any().any()
    assert not bundle.validation[bundle.feature_columns].isna().any().any()
