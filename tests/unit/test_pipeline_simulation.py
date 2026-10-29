"""The drift simulation runs end to end on a tiny synthetic corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from mailguard.monitoring import drift_job
from mailguard.pipeline.simulation import held_out, simulate
from tests.models import TINY
from tests.unit.test_models_evaluate import tiny_frame


def test_held_out_takes_whole_clusters() -> None:
    frame = tiny_frame()

    held = held_out(frame, from_year=2020)

    recent = frame["year"].fillna(0).astype(int) >= 2020
    assert held[recent].all()
    assert set(frame.loc[held, "cluster"]) == set(frame.loc[recent, "cluster"])


def test_simulation_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drift_job, "MIN_ROWS", 5)

    result = simulate(tiny_frame(), TINY, tmp_path / "sim", from_year=2020)

    assert result["held_out"]["phishing"] > 0
    for phase in ("normal_traffic", "new_campaigns"):
        assert result[phase]["messages"] > 5 and len(result[phase]["top_features"]) > 0
    assert 0 <= result["recall_on_new_campaigns"]["production"] <= 1
    assert result["retraining"]["candidate"] != result["production_version"]
    assert {"promoted", "reasons", "evidence"} <= set(result["retraining"])
