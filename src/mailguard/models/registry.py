"""The MLflow model registry: which version is in production, and how it gets there.

Training registers every run as a new version; nothing is served until a version is given the
`production` alias. Phase 2 starts with a manual promotion (`scripts/promote.py`); the retraining
flow (#39) will only move the alias when a new version is significantly better.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from mailguard.config import Settings

PRODUCTION: Final = "production"
THRESHOLD_TAG: Final = "threshold_"


@dataclass(frozen=True, slots=True)
class Version:
    name: str
    version: str
    tags: Mapping[str, str]

    @property
    def thresholds(self) -> dict[str, float]:
        """Calibrated operating thresholds recorded at training time, by FPR target."""
        return {
            key.removeprefix(THRESHOLD_TAG): float(value)
            for key, value in self.tags.items()
            if key.startswith(THRESHOLD_TAG)
        }


def client(settings: Settings) -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # slow import, only needed here

    return mlflow.MlflowClient(settings.mlflow_tracking_uri)


def latest_version(name: str, settings: Settings) -> Version:
    versions = client(settings).search_model_versions(f"name = '{name}'")
    if not versions:
        raise LookupError(f"no registered model {name}: run `make train` first")
    latest = max(versions, key=lambda v: int(v.version))
    return Version(name, str(latest.version), dict(latest.tags))


def aliased_version(name: str, settings: Settings, alias: str = PRODUCTION) -> Version:
    try:
        found = client(settings).get_model_version_by_alias(name, alias)
    except Exception as exc:  # MLflow raises its own RestException for a missing alias
        raise LookupError(f"{name} has no @{alias} version: run `make promote`") from exc
    return Version(name, str(found.version), dict(found.tags))


def promote(version: Version, settings: Settings, alias: str = PRODUCTION) -> None:
    """Point `alias` at `version`, refusing a version the API could not serve."""
    if not version.thresholds:
        raise ValueError(
            f"{version.name} v{version.version} has no operating thresholds: retrain it"
        )
    client(settings).set_registered_model_alias(version.name, alias, version.version)


def load_version(version: Version, settings: Settings) -> Any:
    """The registered model itself (skops-loaded: only trusted types are rebuilt).

    The artifacts are copied to a temporary folder and loaded from there. Loading a
    `models:/` URI in place writes metadata next to the artifacts, which would force the API to
    mount the model store read-write; read-only, a compromised API cannot alter a model.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow
    import mlflow.artifacts
    import mlflow.sklearn

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with tempfile.TemporaryDirectory(prefix="mailguard-model-") as folder:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{version.name}/{version.version}", dst_path=folder
        )
        return mlflow.sklearn.load_model(local)
