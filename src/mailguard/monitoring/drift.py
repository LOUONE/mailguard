"""Does live traffic still look like the data the model was validated on?

Two standard statistics per feature, computed on numbers only (the structured features and the
score: never text):

- **PSI**, the Population Stability Index of credit scoring. Bin the reference, compare the share
  of each population in each bin: `sum((cur - ref) * ln(cur / ref))`. Conventional reading:
  below 0.1 stable, 0.1 to 0.25 moderate shift, above 0.25 significant. It measures *how much*
  a distribution moved, whatever the sample sizes.
- **Kolmogorov-Smirnov**, the largest gap between the two cumulative distributions, with its
  p-value. With thousands of emails the p-value calls any trivial shift significant, so it is
  reported but does not decide; PSI does.

A feature drifts when its PSI exceeds `PSI_DRIFT`. The traffic drifts when at least
`SHARE_DRIFT` of the features do (the default of common tools), or when the score itself does:
a shift in what the model outputs matters even if each input moved a little.

Many features here are small counts (0, 1, 2 links...). Quantile bins collapse on such data, so
a feature with few distinct values is binned by value instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import ks_2samp

BINS: Final = 10
PSI_DRIFT: Final = 0.25
SHARE_DRIFT: Final = 0.5
# Floor on bin shares, so that an empty bin does not make the logarithm infinite.
_EPS: Final = 1e-4
SCORE: Final = "score"


def bin_edges(reference: NDArray[np.float64], bins: int = BINS) -> NDArray[np.float64]:
    """Inner bin edges from the reference: by value for counts, by quantile otherwise."""
    values = np.unique(reference)
    if values.size <= bins:
        return (values[:-1] + values[1:]) / 2  # one bin per observed value
    return np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))


def shares(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.float64]:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=edges.size + 1)
    return np.maximum(counts / max(values.size, 1), _EPS)


def psi(reference: ArrayLike, current: ArrayLike, bins: int = BINS) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    edges = bin_edges(ref, bins)
    p, q = shares(ref, edges), shares(cur, edges)
    return float(np.sum((q - p) * np.log(q / p)))


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float

    @property
    def drifted(self) -> bool:
        return self.psi > PSI_DRIFT


@dataclass(frozen=True, slots=True)
class DriftReport:
    reference_rows: int
    current_rows: int
    features: list[FeatureDrift]

    @property
    def inputs(self) -> list[FeatureDrift]:
        return [f for f in self.features if f.feature != SCORE]

    @property
    def share_drifted(self) -> float:
        inputs = self.inputs
        return sum(f.drifted for f in inputs) / len(inputs) if inputs else 0.0

    @property
    def score_drifted(self) -> bool:
        return any(f.drifted for f in self.features if f.feature == SCORE)

    @property
    def drifted(self) -> bool:
        return self.share_drifted >= SHARE_DRIFT or self.score_drifted

    def summary(self) -> dict[str, Any]:
        """Numbers only, safe to log, expose as metrics and commit."""
        return {
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "share_drifted": self.share_drifted,
            "score_drifted": self.score_drifted,
            "drifted": self.drifted,
            "thresholds": {"psi": PSI_DRIFT, "share": SHARE_DRIFT},
            "features": [
                {**asdict(f), "drifted": f.drifted}
                for f in sorted(self.features, key=lambda f: -f.psi)
            ],
        }


def compare(
    reference: Mapping[str, Sequence[float] | NDArray[np.float64]],
    current: Mapping[str, Sequence[float] | NDArray[np.float64]],
) -> DriftReport:
    """Drift of every column present in both, `reference` and `current` being column-wise."""
    columns = [c for c in reference if c in current]
    if not columns:
        raise ValueError("no column in common between reference and current data")
    out = []
    for column in columns:
        ref = np.asarray(reference[column], dtype=np.float64)
        cur = np.asarray(current[column], dtype=np.float64)
        if ref.size == 0 or cur.size == 0:
            raise ValueError(f"no data for {column}")
        test = ks_2samp(ref, cur)
        out.append(
            FeatureDrift(
                feature=column,
                psi=psi(ref, cur),
                ks_statistic=float(test.statistic),
                ks_pvalue=float(test.pvalue),
                reference_mean=float(ref.mean()),
                current_mean=float(cur.mean()) if not math.isnan(cur.mean()) else 0.0,
            )
        )
    rows = len(next(iter(reference.values())))
    return DriftReport(rows, len(next(iter(current.values()))), out)
