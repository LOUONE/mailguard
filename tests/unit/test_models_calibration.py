from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import skops.io as sio
from hypothesis import given
from hypothesis import strategies as st

from mailguard.models.calibration import (
    CalibratedPipeline,
    ScoreCalibrator,
    adjust_prior,
    choose_method,
    expected_calibration_error,
    precision_at_prior,
    reliability_table,
)
from mailguard.models.pipelines import TRUSTED_TYPES
from mailguard.parsing import Email


def overconfident(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores pushed towards 0 and 1 compared with the true probabilities."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    target = (rng.random(n) < true_p).astype(np.int64)
    logit = np.log(true_p / (1 - true_p)) * 3
    return 1 / (1 + np.exp(-logit)), target


# --- measuring -----------------------------------------------------------------------------


def test_ece_of_known_cases() -> None:
    target = np.array([0, 0, 1, 1])

    assert expected_calibration_error(target, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert expected_calibration_error(target, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0
    # Everything in one bin at 0.95 while half is phishing: gap 0.45.
    assert expected_calibration_error(target, np.full(4, 0.95)) == pytest.approx(0.45)


def test_ece_weights_act_like_repeated_rows() -> None:
    scores, target = overconfident(200)
    weights = np.arange(200) % 3

    repeated = expected_calibration_error(np.repeat(target, weights), np.repeat(scores, weights))

    assert expected_calibration_error(target, scores, weights.astype(float)) == pytest.approx(
        repeated
    )
    assert math.isnan(expected_calibration_error(target, scores, np.zeros(200)))


def test_reliability_table_skips_empty_bins() -> None:
    rows = reliability_table(np.array([0, 1, 1]), np.array([0.05, 0.92, 1.0]))

    assert [(r["low"], r["messages"]) for r in rows] == [(0.0, 1), (0.9, 2)]
    assert rows[1]["phishing_rate"] == 1.0


# --- calibrating ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrators_fix_overconfidence_and_stay_monotone(method: str) -> None:
    scores, target = overconfident()
    fit, check = slice(0, 2000), slice(2000, None)

    calibrator = ScoreCalibrator(method).fit(scores[fit], target[fit])  # type: ignore[arg-type]
    calibrated = calibrator.transform(scores[check])

    before = expected_calibration_error(target[check], scores[check])
    after = expected_calibration_error(target[check], calibrated)
    assert after < before / 2
    grid = calibrator.transform(np.linspace(0, 1, 101))
    assert np.all(np.diff(grid) >= -1e-12) and grid.min() >= 0 and grid.max() <= 1
    assert calibrator.prior_ == pytest.approx(target[fit].mean())


def test_choose_method_uses_grouped_folds_and_reports_both_losses() -> None:
    scores, target = overconfident(1000)
    groups = [f"c{i // 2}" for i in range(1000)]

    method, losses = choose_method(scores, target, groups)

    assert method in ("sigmoid", "isotonic")
    assert set(losses) == {"sigmoid", "isotonic"}
    assert losses[method] == min(losses.values())


class FixedScores:
    """Stands in for a fitted pipeline: the score is the email's subject parsed as a float."""

    def predict_proba(self, emails: list[Email]) -> np.ndarray:
        p = np.array([float(e.subject) for e in emails])
        return np.column_stack([1 - p, p])


def test_calibrated_pipeline_wraps_raw_scores() -> None:
    emails = [Email(subject=s) for s in ("0.1", "0.5", "0.99")]
    calibrator = ScoreCalibrator("isotonic").fit([0.1, 0.5, 0.99], [0, 1, 1])
    model = CalibratedPipeline(FixedScores(), calibrator).fit([])

    proba = model.predict_proba(emails)
    assert model.__sklearn_tags__().input_tags.two_d_array is False

    assert np.array_equal(model.raw_scores(emails), [0.1, 0.5, 0.99])
    assert proba.shape == (3, 2) and np.allclose(proba.sum(axis=1), 1)
    assert list(model.predict(emails)) == [0, 1, 1]
    assert list(model.classes_) == [0, 1]


def test_calibrator_saves_with_trusted_types_only() -> None:
    scores, target = overconfident(300)
    calibrator = ScoreCalibrator("sigmoid").fit(scores, target)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "calibrator.skops"
        sio.dump(calibrator, path)
        assert set(sio.get_untrusted_types(file=path)) <= set(TRUSTED_TYPES)
        loaded = sio.load(path, trusted=list(TRUSTED_TYPES))

    assert np.array_equal(loaded.transform(scores), calibrator.transform(scores))


# --- base rate -----------------------------------------------------------------------------


def test_adjust_prior_known_value() -> None:
    # Odds 9 * (0.01/0.99) / (0.571/0.429) = 0.0683, i.e. a probability of about 0.064.
    assert float(adjust_prior(0.9, 0.571, 0.01)) == pytest.approx(0.0639, abs=1e-4)


@given(st.floats(0.001, 0.999), st.floats(0.01, 0.99), st.floats(0.001, 0.999))
def test_adjust_prior_round_trips_and_keeps_order(p: float, a: float, b: float) -> None:
    there = adjust_prior(p, a, b)

    assert float(adjust_prior(there, b, a)) == pytest.approx(p, rel=1e-6, abs=1e-6)
    assert float(adjust_prior(p, a, a)) == pytest.approx(p)
    assert float(adjust_prior(min(p + 0.0005, 0.9995), a, b)) >= float(there)


@pytest.mark.parametrize("prior", [0.0, 1.0, -0.1])
def test_adjust_prior_rejects_degenerate_priors(prior: float) -> None:
    with pytest.raises(ValueError, match="priors"):
        adjust_prior(0.5, prior, 0.5)


def test_precision_at_prior() -> None:
    # 99% recall and 1% FPR in a mailbox with 1% phishing: half of the alerts are false.
    assert precision_at_prior(0.99, 0.01, 0.01) == pytest.approx(0.5)
    assert precision_at_prior(0.99, 0.0, 0.01) == 1.0
    assert math.isnan(precision_at_prior(0.0, 0.0, 0.01))
