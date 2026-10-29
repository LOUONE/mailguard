"""Per-feature contributions to a score: for the report, the API and the Phase 3 agent.

Contributions are additive in log-odds: bias + sum of contributions = the model's raw margin.

- LightGBM: exact TreeSHAP values from LightGBM itself (`pred_contrib=True`), the algorithm the
  `shap` package implements for trees, without its numba and llvmlite dependencies.
- Logistic regression: `coefficient * value` for each feature, the exact decomposition of a linear
  margin (relative to an all-zero input rather than to the average message).

They explain the raw pipeline. Calibration is a monotone map applied afterwards, so it changes the
probability but not which features pushed the score up or down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from mailguard.parsing import Email

# Feature names are "<group>__<feature>"; TF-IDF terms are "text__word:..." or "text__char:...".
TEXT_GROUP: Final = "text"


@dataclass(frozen=True, slots=True)
class Contributions:
    values: sp.csr_matrix  # one row per email, one column per feature
    bias: NDArray[np.float64]  # one value per email
    names: NDArray[np.str_]

    @property
    def margin(self) -> NDArray[np.float64]:
        """The model's raw log-odds, one per email: bias plus every contribution."""
        return self.bias + np.asarray(self.values.sum(axis=1)).ravel()

    @property
    def probability(self) -> NDArray[np.float64]:
        """The pipeline's own probability (both models are logistic in their margin)."""
        return 1 / (1 + np.exp(-self.margin))


def feature_names(pipeline: Pipeline) -> NDArray[np.str_]:
    """Names of every column the classifier sees. Tens of thousands: compute them once."""
    return np.asarray(pipeline[:-1].get_feature_names_out(), dtype=np.str_)


def contributions_of(classifier: Any, features: Any, names: NDArray[np.str_]) -> Contributions:
    """Contributions from an already computed feature matrix."""
    if hasattr(classifier, "booster_"):  # LightGBM
        raw = sp.csr_matrix(classifier.booster_.predict(features, pred_contrib=True))
        values, bias = raw[:, :-1], raw[:, -1].toarray().ravel()
    else:  # linear model
        # float64 first: the TF-IDF matrix is float32, and summing contributions in float32
        # drifts from the classifier's own margin in the sixth decimal.
        values = sp.csr_matrix(features, dtype=np.float64).multiply(classifier.coef_[0]).tocsr()
        bias = np.full(features.shape[0], float(classifier.intercept_[0]))
    return Contributions(sp.csr_matrix(values), bias.astype(np.float64), names)


def contributions(pipeline: Pipeline, emails: Sequence[Email]) -> Contributions:
    """Additive log-odds contributions of every feature, for every email."""
    features = pipeline[:-1].transform(emails)
    return contributions_of(pipeline[-1], features, feature_names(pipeline))


def group_of(name: str) -> str:
    return name.split("__", 1)[0]


def global_importance(c: Contributions) -> list[tuple[str, float]]:
    """Mean absolute contribution of each feature, largest first."""
    mean_abs = np.asarray(abs(c.values).mean(axis=0)).ravel()
    order = np.argsort(-mean_abs, kind="stable")
    return [(str(c.names[i]), float(mean_abs[i])) for i in order if mean_abs[i] > 0]


def group_importance(c: Contributions) -> dict[str, float]:
    """Share of the total mean absolute contribution carried by each feature group."""
    totals: dict[str, float] = {}
    for name, value in global_importance(c):
        totals[group_of(name)] = totals.get(group_of(name), 0.0) + value
    grand = sum(totals.values())
    return {g: v / grand for g, v in sorted(totals.items())} if grand else {}


def top_reasons(c: Contributions, top: int = 5, row: int = 0) -> list[dict[str, Any]]:
    """The `top` features that moved one email's score most, with their direction.

    Works on the sparse row directly: an email has a few hundred non-zero contributions out of
    tens of thousands of columns. Ties go to the lower column, as a stable dense sort would.
    """
    sparse = c.values.getrow(row)
    keep = sparse.data != 0
    columns, values = sparse.indices[keep], sparse.data[keep]
    order = np.lexsort((columns, -np.abs(values)))[:top]
    return [
        {
            "feature": str(c.names[columns[i]]),
            "group": group_of(str(c.names[columns[i]])),
            "contribution": float(values[i]),
            "direction": "phishing" if values[i] > 0 else "legitimate",
        }
        for i in order
    ]


def explain(pipeline: Pipeline, email: Email, top: int = 5) -> list[dict[str, Any]]:
    """The `top` features that moved one email's score most, with their direction."""
    return top_reasons(contributions(pipeline, [email]), top)
