"""Inference wrapper around model artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.registry import load_artifact


def predict_probabilities(
    features: pd.DataFrame,
    model_path: Path,
    metadata_path: Path,
) -> pd.Series:
    """Return up-probability for each row of features."""
    artifact = load_artifact(model_path=model_path, metadata_path=metadata_path)
    expected = artifact.metadata["feature_columns"]
    x = features[expected]
    probs = artifact.model.predict_proba(x)[:, 1]
    return pd.Series(probs, index=features.index, name="up_probability")
