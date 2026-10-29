"""Promotion through the registry alias and loading what the API serves, on a temporary MLflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mailguard.config import Settings
from mailguard.data.dataset import write_dataset
from mailguard.models.registry import (
    Version,
    aliased_version,
    latest_version,
    promote,
    training_record,
)
from mailguard.models.train import track_run
from mailguard.serving.model import ServedModel, load_production
from tests.models import TINY, messages, tiny_model


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    processed = tmp_path / "processed"
    frame = pd.DataFrame({"id": ["m0"], "split": ["train"], "raw": [b"x"]})
    write_dataset(frame, {"assignment_sha256": "a" * 64, "rows": 1}, processed)
    return Settings(
        data_dir=tmp_path,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}",
        mlflow_artifact_dir=tmp_path / "mlruns" / "artifacts",
    )


def register(settings: Settings, with_thresholds: bool = True) -> None:
    model, thresholds = tiny_model()
    track_run(
        "lightgbm", model, TINY, {}, settings.data_dir / "processed", settings,
        thresholds if with_thresholds else None,
    )  # fmt: skip


def test_versions_carry_their_thresholds(settings: Settings) -> None:
    register(settings)

    version = latest_version("mailguard-lightgbm", settings)

    assert version.version == "1"
    assert set(version.thresholds) == {"1pct", "01pct"}


def test_promote_then_serve(settings: Settings) -> None:
    register(settings)
    register(settings)

    promote(latest_version("mailguard-lightgbm", settings), settings)
    served = load_production(settings)

    assert aliased_version("mailguard-lightgbm", settings).version == "2"
    assert served.version == "2" and served.threshold == served.thresholds["1pct"]
    prediction = served.predict(messages(2).emails[1], base_rate=0.01)
    assert prediction.is_phishing


def test_promotion_refuses_a_version_without_thresholds(settings: Settings) -> None:
    register(settings, with_thresholds=False)

    with pytest.raises(ValueError, match="no operating thresholds"):
        promote(latest_version("mailguard-lightgbm", settings), settings)


def test_missing_alias_or_model_is_a_lookup_error(settings: Settings) -> None:
    with pytest.raises(LookupError, match="make train"):
        latest_version("mailguard-lightgbm", settings)
    register(settings)
    with pytest.raises(LookupError, match="make promote"):
        aliased_version("mailguard-lightgbm", settings)


def test_serving_refuses_an_uncalibrated_model(settings: Settings) -> None:
    model, thresholds = tiny_model()
    track_run(
        "lightgbm", model.pipeline, TINY, {}, settings.data_dir / "processed", settings, thresholds
    )
    promote(latest_version("mailguard-lightgbm", settings), settings)

    with pytest.raises(TypeError, match="not a calibrated pipeline"):
        load_production(settings)


def test_served_model_needs_a_threshold_for_its_operating_point() -> None:
    model, _ = tiny_model()

    with pytest.raises(LookupError, match="no threshold for 1pct"):
        ServedModel("mailguard-lightgbm", "1", model, {"01pct": 0.9})


def test_thresholds_ignore_other_tags() -> None:
    version = Version("m", "1", {"threshold_1pct": "0.4", "owner": "x"})

    assert version.thresholds == {"1pct": 0.4}


def test_training_record_of_old_and_new_versions(settings: Settings) -> None:
    register(settings)  # trained before ids were recorded: no seen_ids.json
    model, thresholds = tiny_model()
    track_run(
        "lightgbm", model, TINY, {}, settings.data_dir / "processed", settings, thresholds,
        seen=["m1", "m2"],
    )  # fmt: skip

    old = latest_version("mailguard-lightgbm", settings)
    old = Version(old.name, "1", old.tags, aliased_or_run(settings, "1"))
    new = latest_version("mailguard-lightgbm", settings)

    assert training_record(old, settings) == ("a" * 64, None)
    assert training_record(new, settings) == ("a" * 64, {"m1", "m2"})
    assert training_record(Version("m", "1", {}), settings) == (None, None)


def aliased_or_run(settings: Settings, version: str) -> str:
    from mailguard.models.registry import client

    run_id: str = client(settings).get_model_version("mailguard-lightgbm", version).run_id
    return run_id
