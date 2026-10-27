from __future__ import annotations

import math

import numpy as np
import pytest

from mailguard.models.stats import (
    ClusterBootstrap,
    mcnemar,
    weighted_pr_auc,
    weighted_rate,
)


def test_replicates_are_reproducible_and_keep_clusters_whole() -> None:
    clusters = ["a", "a", "b", "c", "c", "c"]

    first, second = ClusterBootstrap(clusters, 50, seed=1), ClusterBootstrap(clusters, 50, seed=1)

    assert np.array_equal(first.weights, second.weights)
    assert not np.array_equal(first.weights, ClusterBootstrap(clusters, 50, seed=2).weights)
    # Rows of one cluster always get the same weight, and each replicate draws 3 clusters.
    for w in first.weights:
        assert w[0] == w[1] and w[3] == w[4] == w[5]
    assert np.all(first.weights[:, [0, 2, 3]].sum(axis=1) == 3)


def test_duplicated_evidence_widens_the_interval() -> None:
    # 40 independent results vs the same 40 results each copied 10 times. A message bootstrap
    # would call the copies 400 observations and shrink the interval; clusters must not.
    rng = np.random.default_rng(0)
    flags = rng.random(40) < 0.7
    mask = np.ones(40, dtype=bool)
    single = ClusterBootstrap([f"c{i}" for i in range(40)], 2000, seed=0)
    copied_flags, copied_mask = np.repeat(flags, 10), np.repeat(mask, 10)
    copied = ClusterBootstrap([f"c{i}" for i in range(40) for _ in range(10)], 2000, seed=0)
    as_messages = ClusterBootstrap([f"m{i}" for i in range(400)], 2000, seed=0)

    a = single.interval(lambda w: weighted_rate(flags, mask, w))
    b = copied.interval(lambda w: weighted_rate(copied_flags, copied_mask, w))
    c = as_messages.interval(lambda w: weighted_rate(copied_flags, copied_mask, w))

    assert a.estimate == b.estimate == pytest.approx(flags.mean())
    assert b.high - b.low == pytest.approx(a.high - a.low, rel=0.15)
    assert c.high - c.low < 0.5 * (b.high - b.low)  # what resampling messages would claim
    assert a.low <= a.estimate <= a.high


def test_interval_of_an_undefined_statistic_is_nan() -> None:
    boot = ClusterBootstrap(["a", "b"], 10)
    interval = boot.interval(lambda w: math.nan)

    assert math.isnan(interval.low) and math.isnan(interval.high)


def test_weighted_rate() -> None:
    flags = np.array([True, False, True, True])
    mask = np.array([True, True, True, False])

    assert weighted_rate(flags, mask, np.ones(4)) == pytest.approx(2 / 3)
    assert weighted_rate(flags, mask, np.array([0.0, 2.0, 1.0, 5.0])) == pytest.approx(1 / 3)
    assert math.isnan(weighted_rate(flags, ~np.ones(4, dtype=bool), np.ones(4)))


def test_weighted_pr_auc_needs_both_classes() -> None:
    target = np.array([0, 1, 1])
    scores = np.array([0.1, 0.9, 0.2])

    assert weighted_pr_auc(target, scores, np.ones(3)) == pytest.approx(1.0)
    assert math.isnan(weighted_pr_auc(target, scores, np.array([0.0, 1.0, 1.0])))


def test_mcnemar_counts_discordant_pairs_only() -> None:
    a = np.array([True] * 10 + [False] * 2 + [True] * 50)
    b = np.array([False] * 10 + [True] * 2 + [True] * 50)

    result = mcnemar(a, b)

    assert (result.only_a_correct, result.only_b_correct) == (10, 2)
    assert result.statistic == pytest.approx((8 - 1) ** 2 / 12)
    assert result.p_value == pytest.approx(0.0386, abs=1e-4)  # exact binomial, n=12, k=2


def test_mcnemar_without_disagreement() -> None:
    same = np.array([True, False, True])

    assert mcnemar(same, same).p_value == 1.0
