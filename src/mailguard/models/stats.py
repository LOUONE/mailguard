"""Uncertainty for metrics on clustered data: cluster bootstrap and McNemar's test.

21% of messages repeat another message's body, and copies of a campaign are not independent
observations. Resampling messages would treat 70 copies as 70 pieces of evidence and give
intervals that are too narrow, so the bootstrap resamples duplicate clusters instead. Each
replicate is a vector of row weights (how many times the row's cluster was drawn), which every
metric below accepts, and the same replicates are reused for every model so that differences
between models are paired.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score

Weights = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class McNemar:
    only_a_correct: int
    only_b_correct: int
    statistic: float  # chi-squared with continuity correction, for reference
    p_value: float  # exact two-sided binomial test, valid for small counts

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class ClusterBootstrap:
    """A fixed set of cluster-bootstrap replicates for one set of rows."""

    def __init__(self, clusters: Sequence[str], n_boot: int = 1000, seed: int = 0) -> None:
        codes, uniques = pd.factorize(pd.Series(list(clusters)), sort=True)
        rng = np.random.default_rng(seed)
        n = len(uniques)
        draws = rng.integers(0, n, size=(n_boot, n))
        counts = np.stack([np.bincount(row, minlength=n) for row in draws])
        self.weights: NDArray[np.float64] = counts[:, codes].astype(np.float64)
        self.n_rows = len(codes)

    def interval(self, statistic: Callable[[Weights], float], level: float = 0.95) -> Interval:
        """Point estimate on the data as is, and a percentile interval over the replicates."""
        estimate = statistic(np.ones(self.n_rows))
        values = np.array([statistic(w) for w in self.weights])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return Interval(estimate, math.nan, math.nan)
        tail = (1 - level) / 2
        low, high = np.quantile(values, [tail, 1 - tail])
        return Interval(estimate, float(low), float(high))


def weighted_rate(flags: NDArray[np.bool_], mask: NDArray[np.bool_], weights: Weights) -> float:
    """Weighted share of `flags` among the rows in `mask`; NaN when the mask is empty."""
    total = float(weights[mask].sum())
    return float(weights[mask & flags].sum()) / total if total else math.nan


def weighted_pr_auc(
    target: NDArray[np.int64], scores: NDArray[np.float64], weights: Weights
) -> float:
    if not (weights[target == 1].sum() and weights[target == 0].sum()):
        return math.nan
    return float(average_precision_score(target, scores, sample_weight=weights))


def mcnemar(correct_a: NDArray[np.bool_], correct_b: NDArray[np.bool_]) -> McNemar:
    """Do two classifiers make significantly different errors on the same rows?

    Only discordant rows count. The exact binomial test is used because the discordant count is
    small at a 1% false-positive operating point. It assumes independent rows: with duplicate
    clusters the p-value is optimistic, which is why the paired cluster-bootstrap interval of the
    difference is reported next to it.
    """
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return McNemar(b, c, 0.0, 1.0)
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return McNemar(b, c, float(statistic), float(binomtest(min(b, c), b + c, 0.5).pvalue))
