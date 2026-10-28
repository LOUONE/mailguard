"""The evaluation report on a tiny synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mailguard.config import Settings
from mailguard.data.dataset import write_dataset
from mailguard.models.config import BaselineConfig, LightGBMConfig, LogRegConfig, TextConfig
from mailguard.models.evaluate import (
    Rows,
    evaluate,
    load_registered,
    main,
    render_markdown,
    write_report,
)
from mailguard.models.train import calibrate, split_data, track_run, train_baseline

TINY_TEXT = TextConfig(word_max_features=60, char_max_features=60, min_df=1)
TINY = BaselineConfig(
    logreg=LogRegConfig(text=TINY_TEXT),
    lightgbm=LightGBMConfig(n_estimators=5, min_child_samples=1, n_jobs=1, text=TINY_TEXT),
)

HAM = "From: a{i}@example.org\nSubject: lunch\n\nLunch at noon {i}? Bring the slides.\n"
SPAM = "From: deals{i}@shop.example\nSubject: SALE\n\nCheap watches {i}, buy now!\n"
PHISH = (
    "From: Bank <x{i}@evil.example>\nSubject: verify\nContent-Type: text/html\n\n"
    '<a href="http://198.51.100.{i}/login">www.bank.com</a> Verify your account password.\n'
)
SECRET = "Lunch at noon"  # must never appear in a report


def tiny_frame() -> pd.DataFrame:
    rows = []
    for i in range(150):
        split = ("train", "train", "train", "val", "test")[i % 5]
        kind = i % 3
        label = ("ham", "spam", "phishing")[kind]
        period = "2002-2003" if kind < 2 else ("2004-2007" if i % 2 else "2015-2025")
        year = None if period != "2015-2025" else (2018 if i % 4 else 2022)
        temporal = split
        if label == "phishing":
            temporal = "test" if year == 2022 else ("val" if split == "val" else "train")
        raw = (HAM, SPAM, PHISH)[kind].format(i=i % 200)
        rows.append({"id": f"m{i:03}", "label": label, "target": int(kind == 2),
                     "split": split, "temporal_split": temporal, "period": period, "year": year,
                     "cluster": f"c{i // 2}" if label == "phishing" else f"c{i}",
                     "raw": raw.encode()})  # fmt: skip
    frame = pd.DataFrame(rows)
    frame["year"] = frame["year"].astype("Int64")
    # A cluster must not straddle splits; pairs were built inside one split row by row.
    frame["cluster"] = frame["cluster"] + ":" + frame["split"] + ":" + frame["temporal_split"]
    return frame


def calibrated_models(frame: pd.DataFrame) -> dict:  # type: ignore[type-arg]
    val = split_data(frame, "val")
    return {n: calibrate(train_baseline(n, frame, TINY), val)[0] for n in ("logreg", "lightgbm")}


@pytest.fixture(scope="module")
def report() -> dict:  # type: ignore[type-arg]
    frame = tiny_frame()
    pipelines = calibrated_models(frame)
    temporal = {n: train_baseline(n, frame, TINY, "temporal_split") for n in ("logreg", "lightgbm")}
    return evaluate(frame, pipelines, temporal, n_boot=50)  # type: ignore[arg-type]


def test_rows_masks() -> None:
    rows = Rows.of(tiny_frame(), "split", "test")

    assert rows.is_label("ham").sum() + rows.is_label("spam").sum() + rows.target.sum() == 30
    assert not (rows.is_period("2004-2007") & rows.is_period("2015-2025")).any()
    assert (rows.is_recent(2019) <= rows.is_period("2015-2025")).all()


def test_report_has_every_metric_with_an_interval(report: dict) -> None:  # type: ignore[type-arg]
    assert report["test_set"]["messages"] == 30
    for model in report["models"].values():
        assert set(model["pr_auc"]) == {"estimate", "low", "high"}
        for point in model["operating_points"].values():
            for key in ("recall", "recall_2004_2007", "recall_2015_2025", "ham_fpr", "spam_fpr"):
                assert set(point[key]) == {"estimate", "low", "high"}
            assert isinstance(point["ham_false_positives"], int)
    for point in ("1pct", "01pct"):
        mcnemar = report["comparison"][point]["mcnemar"]
        assert 0 <= mcnemar["p_value"] <= 1
    temporal = report["temporal"]["lightgbm"]
    assert temporal["trained_up_to_2019"]["phishing_messages"] > 0
    for name in ("logreg", "lightgbm"):
        calibration = report["calibration"][name]
        assert calibration["method"] in ("sigmoid", "isotonic")
        for key in ("ece_raw", "ece_calibrated", "brier_raw", "brier_calibrated"):
            assert set(calibration[key]) == {"estimate", "low", "high"}
        base_rate = report["base_rate"][name]
        assert set(base_rate["precision_at_1pct_threshold"]) == {"0.001", "0.005", "0.01", "0.05"}
        assert base_rate["probability_at_prior"]["0.01"]["0.9"] < 0.9
        explanation = report["explanations"][name]
        assert sum(explanation["group_share"].values()) == pytest.approx(1.0)
        assert 0 < len(explanation["top_features"]) <= 20
        assert 0 <= explanation["example"]["probability"] <= 1


def test_report_is_deterministic(report: dict) -> None:  # type: ignore[type-arg]
    frame = tiny_frame()
    pipelines = calibrated_models(frame)
    temporal = {n: train_baseline(n, frame, TINY, "temporal_split") for n in ("logreg", "lightgbm")}

    again = evaluate(frame, pipelines, temporal, n_boot=50)  # type: ignore[arg-type]

    assert json.dumps(again, sort_keys=True) == json.dumps(report, sort_keys=True)


def test_markdown_and_json_hold_metrics_only(report: dict, tmp_path: Path) -> None:  # type: ignore[type-arg]
    json_path, md_path, svg_path = write_report(report, tmp_path)
    markdown = md_path.read_text(encoding="utf-8")

    assert svg_path.name in markdown and svg_path.read_text(encoding="utf-8").startswith("<svg")
    sections = ("1% ham FPR", "0.1% ham FPR", "PR-AUC", "McNemar", "Temporal check",
                "Calibration", "real mailbox", "What drives the scores")  # fmt: skip
    for section in sections:
        assert section in markdown
    for text in (markdown, json_path.read_text(encoding="utf-8")):
        assert SECRET not in text and "example.org" not in text and "evil.example" not in text
    assert render_markdown(report) == markdown


def test_single_model_report_skips_the_comparison() -> None:
    frame = tiny_frame()
    pipelines = {"logreg": train_baseline("logreg", frame, TINY)}

    report = evaluate(frame, pipelines, {}, n_boot=20)  # type: ignore[arg-type]

    assert "comparison" not in report and report["temporal"] == {}
    assert report["calibration"] == {} and report["base_rate"] == {}
    assert "Temporal check" not in render_markdown(report)
    assert "Calibration" not in render_markdown(report)


# --- registry ------------------------------------------------------------------------------


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}",
        mlflow_artifact_dir=tmp_path / "mlruns" / "artifacts",
    )


def test_registry_refuses_missing_or_stale_models(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    processed = tmp_path / "processed"
    frame = tiny_frame()
    write_dataset(frame, {"assignment_sha256": "a" * 64, "rows": len(frame)}, processed)

    with pytest.raises(LookupError, match="make train"):
        load_registered("logreg", settings, "a" * 64)

    pipeline = train_baseline("logreg", frame, TINY)
    track_run("logreg", pipeline, TINY, {"pr_auc": 1.0}, processed, settings)

    assert load_registered("logreg", settings, "a" * 64) is not None
    with pytest.raises(LookupError, match="another split"):
        load_registered("logreg", settings, "b" * 64)


def test_main_fails_cleanly_without_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAILGUARD_DATA_DIR", str(tmp_path))

    assert main(["--no-tracking", "--out-dir", str(tmp_path / "reports")]) == 1
    assert not (tmp_path / "reports").exists()
