"""Baseline model training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.dataset_builder import DatasetBundle, build_datasets
from src.models.registry import save_artifact


@dataclass(frozen=True)
class TrainingResult:
    """Training outputs and key metrics."""

    model: Pipeline
    metadata: dict[str, Any]
    validation_metrics: dict[str, float]


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def train_baseline_model(dataset: DatasetBundle) -> TrainingResult:
    """Fit a logistic regression baseline on engineered features."""
    x_train = dataset.train[dataset.feature_columns]
    y_train = dataset.train[dataset.target_column]
    x_val = dataset.validation[dataset.feature_columns]
    y_val = dataset.validation[dataset.target_column]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
        ],
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "roc_auc": _safe_auc(y_val.to_numpy(), val_prob),
        "validation_samples": float(len(x_val)),
        "train_samples": float(len(x_train)),
    }
    metadata = {
        "model_type": "logistic_regression_baseline",
        "feature_columns": dataset.feature_columns,
        "target_column": dataset.target_column,
        "validation_metrics": metrics,
    }
    return TrainingResult(model=model, metadata=metadata, validation_metrics=metrics)


def train_from_candles(
    candles: pd.DataFrame,
    model_path: Path,
    metadata_path: Path,
    horizon: int = 1,
    target_return_threshold: float = 0.001,
) -> TrainingResult:
    """Build dataset from candles, train model, and persist artifacts."""
    bundle = build_datasets(
        candles=candles,
        horizon=horizon,
        target_return_threshold=target_return_threshold,
    )
    result = train_baseline_model(bundle)
    save_artifact(
        model=result.model,
        metadata=result.metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    return result


def train_from_symbol_candles(
    candles_by_symbol: dict[str, pd.DataFrame],
    model_path: Path,
    metadata_path: Path,
    horizon: int = 1,
    target_return_threshold: float = 0.001,
    track_name: str = "long_term",
) -> TrainingResult:
    """Build datasets for many symbols, then fit one shared baseline model."""
    bundles: list[DatasetBundle] = []
    symbol_counts: dict[str, int] = {}
    for symbol, candles in candles_by_symbol.items():
        bundle = build_datasets(
            candles=candles,
            horizon=horizon,
            target_return_threshold=target_return_threshold,
        )
        if bundle.train.empty or bundle.validation.empty:
            continue
        bundles.append(bundle)
        symbol_counts[symbol] = int(len(bundle.train) + len(bundle.validation))

    if not bundles:
        raise ValueError("No valid training data available for multi-symbol training.")

    feature_columns = bundles[0].feature_columns
    target_column = bundles[0].target_column
    combined_train = pd.concat([b.train for b in bundles], ignore_index=True)
    combined_validation = pd.concat([b.validation for b in bundles], ignore_index=True)
    combined_bundle = DatasetBundle(
        train=combined_train,
        validation=combined_validation,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    result = train_baseline_model(combined_bundle)
    metadata = {
        **result.metadata,
        "training_track": track_name,
        "training_symbols": sorted(symbol_counts.keys()),
        "training_samples_by_symbol": symbol_counts,
    }
    save_artifact(
        model=result.model,
        metadata=metadata,
        model_path=model_path,
        metadata_path=metadata_path,
    )
    return TrainingResult(
        model=result.model,
        metadata=metadata,
        validation_metrics=result.validation_metrics,
    )
