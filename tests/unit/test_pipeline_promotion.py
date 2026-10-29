"""The promotion rule: significant recall gain, McNemar agreement, no extra false alarms."""

from __future__ import annotations

import numpy as np

from mailguard.models.stats import Interval, McNemar
from mailguard.pipeline.promotion import Evidence, decide, gather


def evidence(gain: Interval, fpr: Interval, test: McNemar) -> Evidence:
    recall = Interval(0.95, 0.94, 0.96)
    return Evidence(1000, 800, recall, recall, gain, fpr, test)


GAIN = Interval(0.02, 0.01, 0.03)
NO_FPR_CHANGE = Interval(0.0, -0.002, 0.002)
AGREES = McNemar(only_a_correct=3, only_b_correct=25, statistic=15.75, p_value=4e-5)


def test_a_significant_gain_without_false_alarms_is_promoted() -> None:
    decision = decide(evidence(GAIN, NO_FPR_CHANGE, AGREES))

    assert decision.promote
    assert all(r.startswith("pass: ") for r in decision.reasons) and len(decision.reasons) == 3


def test_a_gain_whose_interval_touches_zero_is_kept() -> None:
    decision = decide(evidence(Interval(0.01, -0.001, 0.02), NO_FPR_CHANGE, AGREES))

    assert not decision.promote
    assert decision.reasons[0].startswith("fail: recall gain")


def test_mcnemar_must_agree() -> None:
    weak = McNemar(only_a_correct=8, only_b_correct=14, statistic=1.1, p_value=0.29)
    reversed_ = McNemar(only_a_correct=25, only_b_correct=3, statistic=15.75, p_value=4e-5)

    assert not decide(evidence(GAIN, NO_FPR_CHANGE, weak)).promote
    assert not decide(evidence(GAIN, NO_FPR_CHANGE, reversed_)).promote


def test_more_false_alarms_block_the_promotion() -> None:
    decision = decide(evidence(GAIN, Interval(0.004, 0.001, 0.008), AGREES))

    assert not decision.promote
    assert decision.reasons[2].startswith("fail: ham FPR")


def test_gather_pairs_the_two_models_on_the_same_replicates() -> None:
    target = np.array([1] * 40 + [0] * 40)
    production = np.array([True] * 20 + [False] * 20 + [False] * 40)  # catches half
    candidate = np.array([True] * 40 + [False] * 40)  # catches all, no false alarm
    clusters = [f"c{i}" for i in range(80)]

    found = gather(target, target == 0, clusters, (production, candidate), n_boot=200)

    assert found.recall_difference.estimate == 0.5 and found.recall_difference.low > 0.3
    assert found.ham_fpr_difference.estimate == 0.0
    assert (found.mcnemar.only_a_correct, found.mcnemar.only_b_correct) == (0, 20)
    assert decide(found).promote
