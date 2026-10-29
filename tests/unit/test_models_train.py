"""Training and tracking on a tiny synthetic dataset, with MLflow in a temporary folder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from pydantic import ValidationError

from mailguard.config import Settings
from mailguard.data.dataset import write_dataset
from mailguard.models.config import (
    BaselineConfig,
    LightGBMConfig,
    LogRegConfig,
    TextConfig,
    flatten,
    load_config,
)
from mailguard.models.pipelines import MODEL_NAMES, TRUSTED_TYPES, lightgbm_pipeline
from mailguard.models.train import (
    calibrate,
    feature_summary,
    git_state,
    main,
    operating_thresholds,
    runtime_requirements,
    scores,
    split_data,
    track_run,
    train_baseline,
)

TINY_TEXT = TextConfig(word_max_features=60, char_max_features=60, min_df=1)
TINY = BaselineConfig(
    logreg=LogRegConfig(text=TINY_TEXT),
    lightgbm=LightGBMConfig(n_estimators=5, min_child_samples=1, n_jobs=1, text=TINY_TEXT),
)

HAM = "From: a{i}@example.org\nSubject: lunch\n\nLunch at noon {i}? Bring the slides.\n"
PHISH = (
    "From: PayPal <x{i}@evil.example>\nReply-To: r@other.example\nSubject: verify\n"
    "Content-Type: text/html\n\n"
    '<a href="http://198.51.100.{i}/login">www.paypal.com</a> Verify your account password.\n'
)


def tiny_frame() -> pd.DataFrame:
    rows = []
    for i in range(60):
        split = "train" if i < 36 else ("val" if i < 48 else "test")
        label = "phishing" if i % 2 else "ham"
        raw = (PHISH if i % 2 else HAM).format(i=i)
        if split == "test":
            raw = raw.replace("Lunch", "Zebra").replace("Verify", "Zebra")
        rows.append({"id": f"m{i:02}", "label": label, "target": i % 2, "split": split,
                     "temporal_split": split, "cluster": f"c{i}", "raw": raw.encode()})  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def processed(tmp_path: Path) -> Path:
    """`<tmp>/processed`, laid out like `$MAILGUARD_DATA_DIR/processed`."""
    frame = tiny_frame()
    out = tmp_path / "processed"
    write_dataset(frame, {"assignment_sha256": "a" * 64, "rows": len(frame)}, out)
    return out


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}",
        mlflow_artifact_dir=tmp_path / "mlruns" / "artifacts",
    )


# --- config --------------------------------------------------------------------------------


def test_repository_config_loads() -> None:
    config = load_config()

    assert config.lightgbm.n_jobs >= 1
    assert config.logreg.text.word_max_features > config.lightgbm.text.word_max_features


def test_misspelt_keys_fail_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[logreg]\nc = 1.0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(path)


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


# --- training ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fits_on_training_rows_only(name: str) -> None:
    frame = tiny_frame()

    pipeline = train_baseline(name, frame, TINY)  # type: ignore[arg-type]

    vocabulary = pipeline.named_steps["features"].get_feature_names_out()
    assert not any("zebra" in str(term).casefold() for term in vocabulary)  # test-only word
    assert any("lunch" in str(term).casefold() for term in vocabulary)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_training_is_reproducible(name: str) -> None:
    frame = tiny_frame()
    val = split_data(frame, "val")

    first = scores(train_baseline(name, frame, TINY), val)  # type: ignore[arg-type]
    second = scores(train_baseline(name, frame, TINY), val)  # type: ignore[arg-type]

    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_saved_pipelines_load_with_trusted_types_only(name: str) -> None:
    frame = tiny_frame()
    pipeline = train_baseline(name, frame, TINY)  # type: ignore[arg-type]
    val = split_data(frame, "val")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.skops"
        sio.dump(pipeline, path)
        assert set(sio.get_untrusted_types(file=path)) <= set(TRUSTED_TYPES)
        loaded = sio.load(path, trusted=list(TRUSTED_TYPES))

    assert np.array_equal(scores(pipeline, val), scores(loaded, val))
    assert scores(pipeline, val).shape == (12,)  # saving did not unfit the original


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_calibrate_wraps_the_pipeline_without_changing_its_ranking(name: str) -> None:
    frame = tiny_frame()
    pipeline = train_baseline(name, frame, TINY)  # type: ignore[arg-type]
    val = split_data(frame, "val")

    model, report = calibrate(pipeline, val)

    raw = scores(pipeline, val)
    assert np.array_equal(model.raw_scores(val.emails), raw)
    calibrated = scores(model, val)
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(calibrated[order]) >= -1e-12)  # monotone in the raw score
    assert {"ece_raw", "cv_log_loss_sigmoid", "cv_log_loss_isotonic"} <= set(report)


def test_excluded_features_leave_the_lightgbm_pipeline() -> None:
    lightgbm = TINY.lightgbm.model_copy(update={"exclude_features": ("url_max_length",)})
    config = TINY.model_copy(update={"lightgbm": lightgbm})

    names = feature_summary(train_baseline("lightgbm", tiny_frame(), config))["urls"]["names"]

    assert "url_max_length" not in names and "url_count" in names


def test_excluding_an_unknown_feature_fails_loudly() -> None:
    lightgbm = TINY.lightgbm.model_copy(update={"exclude_features": ("url_length",)})

    with pytest.raises(ValueError, match="url_length"):
        lightgbm_pipeline(TINY.model_copy(update={"lightgbm": lightgbm}))


def test_operating_thresholds_are_on_calibrated_validation_ham() -> None:
    frame = tiny_frame()
    val = split_data(frame, "val")
    model, _ = calibrate(train_baseline("lightgbm", frame, TINY), val)

    thresholds = operating_thresholds(model, val)

    ham = scores(model, val)[val.labels == "ham"]
    assert set(thresholds) == {"1pct", "01pct"}
    assert thresholds["1pct"] == ham.max()  # 6 ham: 1% lets none through
    assert (ham > thresholds["1pct"]).sum() == 0


def test_feature_summary_hides_tfidf_terms() -> None:
    summary = feature_summary(train_baseline("lightgbm", tiny_frame(), TINY))

    assert set(summary) == {"headers", "attachments", "urls", "lexicon", "text"}
    assert "names" not in summary["text"] and summary["text"]["columns"] > 0
    assert "url_ip_hosts" in summary["urls"]["names"]


# --- tracking ------------------------------------------------------------------------------


def test_git_state() -> None:
    state = git_state()
    assert len(state["git_commit"]) == 40 and state["git_dirty"] in {"True", "False"}
    assert git_state(Path(tempfile.gettempdir()) / "not-a-checkout-0") == {
        "git_commit": "unknown",
        "git_dirty": "unknown",
    }


def test_runtime_requirements_are_pinned() -> None:
    assert all("==" in r for r in runtime_requirements())


def test_track_run_records_lineage_and_registers(processed: Path, tmp_path: Path) -> None:
    import mlflow

    settings = settings_for(tmp_path)
    frame = tiny_frame()
    pipeline = train_baseline("logreg", frame, TINY)

    run_id = track_run(
        "logreg", pipeline, TINY, {"pr_auc": 0.9}, processed, settings, {"1pct": 0.25}
    )

    client = mlflow.MlflowClient(settings.mlflow_tracking_uri)
    assert client.get_model_version("mailguard-logreg", "1").tags == {"threshold_1pct": "0.25"}
    run = client.get_run(run_id)
    assert run.data.metrics == {"val_pr_auc": 0.9}
    assert run.data.params["text.word_max_features"] == "60"
    assert run.data.tags["dataset_assignment_sha256"] == "a" * 64
    assert len(run.data.tags["dataset_manifest_sha256"]) == 64
    assert len(run.data.tags["git_commit"]) == 40
    loaded = mlflow.sklearn.load_model("models:/mailguard-logreg/1")
    val = split_data(frame, "val")
    assert np.array_equal(scores(pipeline, val), scores(loaded, val))


def test_main_trains_without_tracking(
    processed: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("MAILGUARD_DATA_DIR", str(processed.parent))
    config = processed.parent / "tiny.toml"
    config.write_text(
        "[logreg.text]\nword_max_features = 60\nchar_max_features = 60\nmin_df = 1\n",
        encoding="utf-8",
    )

    with caplog.at_level("INFO"):
        code = main(["--config", str(config), "--model", "logreg", "--no-tracking"])

    assert code == 0
    assert "logreg: val PR-AUC" in caplog.text


def test_main_fails_cleanly_without_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAILGUARD_DATA_DIR", str(tmp_path))

    assert main(["--no-tracking"]) == 1
