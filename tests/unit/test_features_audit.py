from __future__ import annotations

import numpy as np

from mailguard.features.audit import era_audit, univariate_auc


def test_univariate_auc_is_direction_free() -> None:
    target = np.array([False, False, True, True])

    assert univariate_auc(np.array([0.0, 1.0, 2.0, 3.0]), target) == 1.0
    assert univariate_auc(np.array([3.0, 2.0, 1.0, 0.0]), target) == 1.0
    assert univariate_auc(np.zeros(4), target) == 0.5


def test_era_audit_flags_a_feature_that_encodes_time() -> None:
    rng = np.random.default_rng(0)
    recent = np.arange(400) % 2 == 0
    noise = rng.normal(size=400)
    leaky = recent + rng.normal(scale=0.01, size=400)
    groups = [f"c{i // 2}" for i in range(400)]  # pairs of copies, kept in the same fold

    clean = era_audit(noise[:, None], recent, groups, ["noise"])
    leaking = era_audit(np.c_[noise, leaky], recent, groups, ["noise", "leaky"])

    assert not clean.alert and clean.group_auc < 0.65
    assert leaking.alert
    assert leaking.feature_auc["leaky"] > 0.99 > clean.feature_auc["noise"]
