"""When does a retrained model replace the one in production? (ADR 0008)

Retraining always produces a model that looks a little different; most of the time the
difference is noise. Promoting on any improvement would churn production on luck, and on this
corpus luck is large (one test ham is 0.16% of the FPR). So a candidate is promoted only when,
on messages neither model has seen and at the operating thresholds each one actually serves:

1. **its recall gain is significant**: the 95% cluster-bootstrap interval of
   `recall(candidate) - recall(production)` lies above zero;
2. **McNemar agrees**: more messages that only the candidate gets right than the reverse, with
   p below `ALPHA` (optimistic with duplicates, which is why it is not the only test);
3. **it does not raise false alarms**: the upper bound of the ham FPR difference stays below
   `FPR_MARGIN`, half a point (a non-inferiority margin: "not worse by more than").

Anything else keeps production. That includes a candidate that is better on average but not
significantly: the cost of a missed improvement is one retraining cycle, the cost of promoting
noise is a model nobody can explain the change of.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from mailguard.models.stats import (
    ClusterBootstrap,
    Interval,
    McNemar,
    mcnemar,
    weighted_rate,
)

ALPHA: Final = 0.05
FPR_MARGIN: Final = 0.005


@dataclass(frozen=True, slots=True)
class Evidence:
    """Candidate against production on the same messages, at their own thresholds."""

    messages: int
    clusters: int
    production_recall: Interval
    candidate_recall: Interval
    recall_difference: Interval  # candidate minus production
    ham_fpr_difference: Interval  # candidate minus production
    mcnemar: McNemar  # a = production, b = candidate

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Decision:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def gather(
    target: NDArray[np.int64],
    is_ham: NDArray[np.bool_],
    clusters: list[str],
    flags: tuple[NDArray[np.bool_], NDArray[np.bool_]],
    n_boot: int = 1000,
    seed: int = 0,
) -> Evidence:
    """Paired evidence from both models' decisions (`flags`: production, candidate)."""
    production, candidate = flags
    phishing = target == 1
    truth = phishing
    boot = ClusterBootstrap(clusters, n_boot, seed)

    def recall(f: NDArray[np.bool_]) -> Interval:
        return boot.interval(lambda w: weighted_rate(f, phishing, w))

    def difference(mask: NDArray[np.bool_]) -> Interval:
        return boot.interval(
            lambda w: weighted_rate(candidate, mask, w) - weighted_rate(production, mask, w)
        )

    return Evidence(
        messages=len(target),
        clusters=len(set(clusters)),
        production_recall=recall(production),
        candidate_recall=recall(candidate),
        recall_difference=difference(phishing),
        ham_fpr_difference=difference(is_ham),
        mcnemar=mcnemar(production == truth, candidate == truth),
    )


def decide(evidence: Evidence) -> Decision:
    gain, fpr, test = evidence.recall_difference, evidence.ham_fpr_difference, evidence.mcnemar
    checks = {
        f"recall gain {100 * gain.estimate:+.2f} points, interval "
        f"[{100 * gain.low:+.2f}, {100 * gain.high:+.2f}] above zero": gain.low > 0,
        f"McNemar {test.only_b_correct} only-candidate vs {test.only_a_correct} only-production, "
        f"p = {test.p_value:.3g} below {ALPHA}": (
            test.only_b_correct > test.only_a_correct and test.p_value < ALPHA
        ),
        f"ham FPR difference upper bound {100 * fpr.high:+.2f} points within "
        f"{100 * FPR_MARGIN:.1f}": fpr.high <= FPR_MARGIN,
    }
    reasons = [("pass: " if ok else "fail: ") + check for check, ok in checks.items()]
    return Decision(all(checks.values()), reasons)
