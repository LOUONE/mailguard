"""The retraining flow end to end on a temporary MLflow: every way it can conclude."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mailguard.config import Settings
from mailguard.data.dataset import write_dataset
from mailguard.models.pipelines import lightgbm_pipeline
from mailguard.models.registry import aliased_version, client, latest_version, promote
from mailguard.models.train import SplitData, calibrate, operating_thresholds, track_run
from mailguard.pipeline import retrain
from mailguard.pipeline.retrain import ComparisonError, evaluation_rows, write_outcome
from tests.models import TINY, messages
from tests.unit.test_models_train import tiny_frame


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    frame = tiny_frame()
    manifest = {"assignment_sha256": "a" * 64, "rows": len(frame)}
    write_dataset(frame, manifest, tmp_path / "processed")
    return Settings(
        data_dir=tmp_path,
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}",
        mlflow_artifact_dir=tmp_path / "mlruns" / "artifacts",
    )


def flipped(data: SplitData) -> SplitData:
    target = 1 - data.target
    labels = np.where(target == 1, "phishing", "ham").astype(np.str_)
    return SplitData(data.emails, target, labels, data.clusters)


def register_bad_production(settings: Settings) -> None:
    """A model trained on inverted labels: it misses phishing and flags ham."""
    train, val = flipped(messages(40)), flipped(messages(20, offset=100))
    model, _ = calibrate(lightgbm_pipeline(TINY).fit(train.emails, train.target), val)
    track_run(
        "lightgbm", model, TINY, {}, settings.data_dir / "processed", settings,
        operating_thresholds(model, val), seen=[],
    )  # fmt: skip
    promote(latest_version(settings.model_name, settings), settings)


def test_the_first_candidate_goes_to_production(settings: Settings) -> None:
    outcome = retrain.retrain(settings, TINY, n_boot=50)

    assert outcome.decision.promote and outcome.production is None
    assert aliased_version(settings.model_name, settings).version == "1"


def test_a_candidate_no_better_than_production_is_kept(settings: Settings) -> None:
    retrain.retrain(settings, TINY, n_boot=50)

    outcome = retrain.retrain(settings, TINY, n_boot=50)  # same data, same seed: same model

    assert not outcome.decision.promote and outcome.evidence is not None
    assert outcome.evidence.recall_difference.estimate == 0.0
    assert aliased_version(settings.model_name, settings).version == "1"
    tags = client(settings).get_model_version(settings.model_name, "2").tags
    assert tags["promotion"] == "kept production"


def test_a_significantly_better_candidate_replaces_production(settings: Settings) -> None:
    register_bad_production(settings)

    outcome = retrain.retrain(settings, TINY, n_boot=200)

    assert outcome.decision.promote, outcome.decision.reasons
    assert outcome.evidence is not None and outcome.evidence.recall_difference.low > 0
    assert aliased_version(settings.model_name, settings).version == "2"
    path = write_outcome(outcome, settings.data_dir / "retraining")
    assert '"promoted": true' in path.read_text(encoding="utf-8")


def test_production_on_another_split_without_a_record_is_kept(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_bad_production(settings)
    monkeypatch.setattr(retrain, "training_record", lambda *_: ("b" * 64, None))

    outcome = retrain.retrain(settings, TINY, n_boot=50)

    assert not outcome.decision.promote and outcome.evidence is None
    assert "another split" in outcome.decision.reasons[0]


def test_evaluation_excludes_what_production_saw() -> None:
    frame = tiny_frame()
    test_ids = set(frame.loc[frame["split"] == "test", "id"])
    seen = {sorted(test_ids)[0]}

    rows = evaluation_rows(frame, "a", "b", seen)

    assert set(rows["id"]) == test_ids - seen
    assert len(evaluation_rows(frame, "a", "a", None)) == len(test_ids)
    with pytest.raises(ComparisonError):
        evaluation_rows(frame, "a", "b", None)


def test_main_runs_on_existing_data(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrain, "get_settings", lambda: settings)
    config = settings.data_dir / "tiny.toml"
    config.write_text(
        "[lightgbm]\nn_estimators = 10\nmin_child_samples = 1\nn_jobs = 1\n"
        "[lightgbm.text]\nword_max_features = 60\nchar_max_features = 60\nmin_df = 1\n",
        encoding="utf-8",
    )

    assert retrain.main(["--skip-dataset", "--config", str(config), "--n-boot", "20"]) == 0
    assert (settings.data_dir / "retraining" / "retrain-v1.json").is_file()
