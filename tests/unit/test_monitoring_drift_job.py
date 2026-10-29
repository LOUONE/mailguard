"""Recording traffic, comparing it with validation data, exposing the verdict."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mailguard.config import Settings
from mailguard.data.dataset import write_dataset
from mailguard.monitoring import drift_job
from mailguard.monitoring.recorder import FeatureRecorder, read_records
from mailguard.serving.app import create_app
from mailguard.serving.model import ServedModel
from tests.models import HAM, PHISH, tiny_served_model

CANARY = "zq7canary"


@pytest.fixture(scope="module")
def served() -> ServedModel:
    return tiny_served_model()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A data dir whose validation split is half ham, half phishing."""
    raws = [(PHISH if i % 2 else HAM).format(i=i).encode() for i in range(60)]
    frame = pd.DataFrame({"id": [f"v{i}" for i in range(60)], "split": "val", "raw": raws})
    write_dataset(frame, {"assignment_sha256": "a" * 64, "rows": 60}, tmp_path / "processed")
    return Settings(
        data_dir=tmp_path, feature_log_dir=tmp_path / "traffic", drift_dir=tmp_path / "drift"
    )


def send(client: TestClient, raws: list[str]) -> None:
    for raw in raws:
        assert client.post("/v1/predict", json={"raw": raw}).status_code == 200


def test_the_api_records_numbers_and_never_text(settings: Settings, served: ServedModel) -> None:
    with TestClient(create_app(settings, loader=lambda _: served)) as client:
        send(client, [f"From: {CANARY}@x.example\nSubject: {CANARY}\n\n{CANARY} body"])

    files = list(settings.feature_log_dir.glob("*.jsonl"))  # type: ignore[union-attr]
    assert len(files) == 1 and CANARY not in files[0].read_text(encoding="utf-8")
    (record,) = read_records(settings.feature_log_dir)  # type: ignore[arg-type]
    assert record["model_version"] == "7" and set(record) == {
        "time", "model_version", "score", "label", "features"
    }  # fmt: skip
    assert all(isinstance(v, float) for v in record["features"].values())


def test_traffic_like_validation_does_not_drift(settings: Settings, served: ServedModel) -> None:
    with TestClient(create_app(settings, loader=lambda _: served)) as client:
        send(client, [(PHISH if i % 2 else HAM).format(i=i) for i in range(200, 420)])

    summary = drift_job.run_once(settings, served, window=1000)

    assert summary is not None and summary["drifted"] is False
    assert summary["current_rows"] == 220 and summary["reference_rows"] == 60
    assert (settings.drift_dir / "report.html").read_text(encoding="utf-8").count("<tr") > 30  # type: ignore[operator]


def test_a_phishing_wave_is_flagged_and_exposed(settings: Settings, served: ServedModel) -> None:
    app = create_app(settings, loader=lambda _: served)
    with TestClient(app) as client:
        send(client, [PHISH.format(i=i) for i in range(220)])  # all phishing, no ham
        summary = drift_job.run_once(settings, served, window=1000)
        metrics = client.get("/metrics").text

    assert summary is not None and summary["score_drifted"] is True and summary["drifted"]
    assert "mailguard_drift_detected 1.0" in metrics
    assert 'mailguard_feature_psi{feature="score"}' in metrics


def test_too_little_traffic_waits(settings: Settings, served: ServedModel) -> None:
    with TestClient(create_app(settings, loader=lambda _: served)) as client:
        send(client, [HAM.format(i=1)])

    assert drift_job.run_once(settings, served, window=1000) is None
    assert not (settings.drift_dir / "summary.json").exists()  # type: ignore[operator]


def test_reference_is_cached_per_model_version(settings: Settings, served: ServedModel) -> None:
    first = drift_job.reference(served, settings, settings.drift_dir)  # type: ignore[arg-type]
    cache = settings.drift_dir / "reference-mailguard-lightgbm-v7.json"  # type: ignore[operator]

    assert cache.is_file() and drift_job.reference(served, settings, settings.drift_dir) == first  # type: ignore[arg-type]


def test_only_records_of_the_served_version_are_compared(tmp_path: Path) -> None:
    recorder = FeatureRecorder(tmp_path)
    recorder.record("6", 0.1, "legitimate", {"a": 1.0})
    recorder.record("7", 0.9, "phishing", {"a": 2.0})
    recorder.close()
    (tmp_path / "features-2026-01-01.jsonl").write_text('{"cut', encoding="utf-8")

    columns = drift_job.current(read_records(tmp_path), "7", window=10)

    assert columns == {"a": [2.0], "score": [0.9]}


def test_the_job_needs_both_directories(served: ServedModel) -> None:
    with pytest.raises(ValueError, match="MAILGUARD_FEATURE_LOG_DIR"):
        drift_job.run_once(Settings(), served, window=10)


def test_metrics_ignore_a_missing_or_broken_summary(tmp_path: Path) -> None:
    from mailguard.monitoring.metrics import Metrics

    metrics = Metrics()
    metrics.drift(tmp_path / "absent.json")
    (tmp_path / "summary.json").write_text("{not json", encoding="utf-8")
    metrics.drift(tmp_path / "summary.json")

    assert b"mailguard_drift_detected 0.0" in metrics.render()


def test_main_runs_once_and_fails_cleanly(
    settings: Settings, served: ServedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(drift_job, "load_production", lambda _: served)
    monkeypatch.setattr(drift_job, "get_settings", lambda: settings)

    assert drift_job.main([]) == 0  # no traffic yet: waits, does not fail

    monkeypatch.setattr(drift_job, "get_settings", Settings)
    assert drift_job.main([]) == 1  # no directories configured


def test_main_follows_a_promotion(
    settings: Settings, served: ServedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mailguard.models.registry import Version

    loads: list[str] = []

    def load(_: Settings) -> ServedModel:
        loads.append("load")
        if len(loads) > 1:
            raise LookupError("stop the loop")  # the reload happened
        return served

    monkeypatch.setattr(drift_job, "load_production", load)
    monkeypatch.setattr(drift_job, "get_settings", lambda: settings)
    monkeypatch.setattr("mailguard.monitoring.drift_job.time.sleep", lambda _: None)
    monkeypatch.setattr(
        drift_job, "aliased_version", lambda *_: Version("mailguard-lightgbm", "8", {})
    )

    assert drift_job.main(["--every", "1"]) == 1
    assert loads == ["load", "load"]
