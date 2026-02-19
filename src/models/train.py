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
