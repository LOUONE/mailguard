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


def contributions(pipeline: Pipeline, emails: Sequence[Email]) -> Contributions:
    """Additive log-odds contributions of every feature, for every email."""
    features = pipeline[:-1].transform(emails)
    names = np.asarray(pipeline[:-1].get_feature_names_out(), dtype=np.str_)
    classifier = pipeline[-1]
    if hasattr(classifier, "booster_"):  # LightGBM
        raw = sp.csr_matrix(classifier.booster_.predict(features, pred_contrib=True))
        values, bias = raw[:, :-1], raw[:, -1].toarray().ravel()
    else:  # linear model
        values = sp.csr_matrix(features).multiply(classifier.coef_[0]).tocsr()
        bias = np.full(features.shape[0], float(classifier.intercept_[0]))
    return Contributions(sp.csr_matrix(values), bias.astype(np.float64), names)


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


def explain(pipeline: Pipeline, email: Email, top: int = 5) -> list[dict[str, Any]]:
    """The `top` features that moved one email's score most, with their direction."""
    c = contributions(pipeline, [email])
    row = c.values.toarray().ravel()
    order = np.argsort(-np.abs(row), kind="stable")[:top]
    return [
        {
            "feature": str(c.names[i]),
            "group": group_of(str(c.names[i])),
            "contribution": float(row[i]),
            "direction": "phishing" if row[i] > 0 else "legitimate",
        }
        for i in order
        if row[i] != 0
    ]
