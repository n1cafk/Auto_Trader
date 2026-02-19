"""Model artifact persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class ModelArtifact:
    """Model payload with metadata."""

    model: Any
    metadata: dict[str, Any]


def save_artifact(model: Any, metadata: dict[str, Any], model_path: Path, metadata_path: Path) -> None:
    """Persist trained model and metadata."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    stamped_metadata = {
        **metadata,
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    joblib.dump(model, model_path)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(stamped_metadata, handle, indent=2, sort_keys=True)


def load_artifact(model_path: Path, metadata_path: Path) -> ModelArtifact:
    """Load model and metadata."""
    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return ModelArtifact(model=model, metadata=metadata)
