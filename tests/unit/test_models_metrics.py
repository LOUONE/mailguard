from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mailguard.models.metrics import (
    FPR_TARGETS,
    operating_point,
    ranking_metrics,
    threshold_at_fpr,
    validation_metrics,
)


def test_threshold_allows_floor_of_fpr_times_n_false_positives() -> None:
    negatives = np.arange(200) / 200  # 0.000, 0.005, ..., 0.995

    assert threshold_at_fpr(negatives, 0.01) == 0.985  # 2 of 200 are above it
    assert threshold_at_fpr(negatives, 0.001) == 0.995  # none above: floor(0.2) = 0
    assert threshold_at_fpr(negatives, 0.0) == 0.995


def test_ties_at_the_threshold_are_not_flagged() -> None:
    negatives = np.array([0.9, 0.9, 0.9, 0.1])

    t = threshold_at_fpr(negatives, 0.5)  # 2 of 4 allowed, but three are tied

    assert (negatives > t).sum() == 0


@pytest.mark.parametrize(("scores", "fpr"), [([], 0.01), ([0.5], 1.0), ([0.5], -0.1)])
def test_threshold_rejects_bad_input(scores: list[float], fpr: float) -> None:
    with pytest.raises(ValueError):
        threshold_at_fpr(scores, fpr)


@given(
    st.lists(st.floats(0, 1), min_size=1, max_size=300),
    st.floats(0, 0.99),
)
def test_flagged_negatives_never_exceed_the_target(negatives: list[float], fpr: float) -> None:
    scores = np.array(negatives)

    flagged = (scores > threshold_at_fpr(scores, fpr)).mean()

    assert flagged <= fpr + 1e-12


def test_operating_point_rates_per_label() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.95])
    labels = np.array(["phishing", "phishing", "phishing", "spam", "ham", "ham"])

    point = operating_point(scores, labels, 0.75)

    assert point.recall == pytest.approx(2 / 3)
    assert (point.ham_fpr, point.spam_fpr) == (0.5, 0.0)


def test_missing_label_gives_nan() -> None:
    point = operating_point(np.array([0.9]), np.array(["phishing"]), 0.5)

    assert math.isnan(point.ham_fpr) and math.isnan(point.spam_fpr)


def test_ranking_metrics_survive_certain_scores() -> None:
    metrics = ranking_metrics(np.array([0, 1]), np.array([0.0, 1.0]))

    assert metrics["pr_auc"] == metrics["roc_auc"] == 1.0
    assert math.isfinite(metrics["log_loss"]) and metrics["brier"] == 0.0


def test_validation_metrics_choose_thresholds_on_ham() -> None:
    rng = np.random.default_rng(0)
    labels = np.array(["ham"] * 300 + ["spam"] * 100 + ["phishing"] * 400)
    target = (labels == "phishing").astype(np.int64)
    scores = np.clip(rng.normal(0.2, 0.1, 800) + 0.6 * target, 0, 1)

    metrics = validation_metrics(target, labels, scores)

    for name, fpr in FPR_TARGETS.items():
        assert metrics[f"ham_fpr_at_{name}_fpr"] <= fpr
        assert 0 < metrics[f"recall_at_{name}_fpr"] <= 1
    assert metrics["threshold_at_01pct_fpr"] >= metrics["threshold_at_1pct_fpr"]
