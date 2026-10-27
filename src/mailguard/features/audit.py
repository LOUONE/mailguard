"""Leakage audit: how well does a feature group guess the era of a phishing message?

Every row is phishing, so the label is constant and any skill at telling 2004-2007 from
2015-2025 comes from time alone. All ham dates from 2002: a group that encodes the era can separate
ham from phishing for the wrong reason. A group-level AUC above `ALERT_AUC` is investigated
feature by feature (docs/features.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

ALERT_AUC: Final = 0.95


@dataclass(frozen=True, slots=True)
class AuditResult:
    group_auc: float
    feature_auc: dict[str, float]  # univariate, folded so that 0.5 is useless and 1.0 perfect

    @property
    def alert(self) -> bool:
        return self.group_auc > ALERT_AUC


def univariate_auc(values: NDArray[np.float64], target: NDArray[np.bool_]) -> float:
    """AUC of one feature used as a score, in whichever direction separates better."""
    if np.ptp(values) == 0:
        return 0.5
    auc = float(roc_auc_score(target, values))
    return max(auc, 1.0 - auc)


def era_audit(
    X: NDArray[np.float64],
    recent: NDArray[np.bool_],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    folds: int = 5,
    seed: int = 0,
) -> AuditResult:
    """Cross-validated AUC of a gradient-boosted model predicting `recent` from `X`.

    Folds are grouped by duplicate cluster, so a campaign is never on both sides of a fold.
    """
    model = HistGradientBoostingClassifier(random_state=seed)
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_predict(model, X, recent, groups=groups, cv=cv, method="predict_proba")
    return AuditResult(
        group_auc=float(roc_auc_score(recent, scores[:, 1])),
        feature_auc={n: univariate_auc(X[:, j], recent) for j, n in enumerate(names)},
    )
