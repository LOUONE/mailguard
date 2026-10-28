"""Turn scores into probabilities, and probabilities into the prior of a real mailbox.

Phase 3 sends a band of uncertain scores to the agent, and the API returns a probability: both only
make sense if a score of 0.8 means that 80% of messages scored 0.8 are phishing. This module
measures that (reliability bins, expected calibration error), learns a monotone score -> probability
map on the validation split, and corrects for the base rate.

Calibrated probabilities hold at the prior of the data they were fitted on: 57% phishing here,
against well under 1% in a real mailbox. `adjust_prior` moves them to another base rate with Bayes'
rule on the odds, which assumes the class-conditional score distributions stay the same.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, Literal, Self, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mailguard.parsing import Email

Method = Literal["sigmoid", "isotonic"]
METHODS: Final[tuple[Method, ...]] = get_args(Method)
BINS: Final = 10
_EPS: Final = 1e-6


# --- measuring -----------------------------------------------------------------------------


def _bin_index(scores: NDArray[np.float64], bins: int) -> NDArray[np.int64]:
    return np.minimum((scores * bins).astype(np.int64), bins - 1)


def expected_calibration_error(
    target: NDArray[np.int64],
    scores: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    bins: int = BINS,
) -> float:
    """Weighted mean gap between predicted and observed phishing rate, over equal-width bins."""
    w = np.ones(len(scores)) if weights is None else weights
    total = float(w.sum())
    if not total:
        return math.nan
    index = _bin_index(scores, bins)
    error = 0.0
    for b in range(bins):
        mask = index == b
        mass = float(w[mask].sum())
        if mass:
            gap = np.average(scores[mask], weights=w[mask]) - np.average(
                target[mask], weights=w[mask]
            )
            error += mass / total * abs(float(gap))
    return error


def reliability_table(
    target: NDArray[np.int64], scores: NDArray[np.float64], bins: int = BINS
) -> list[dict[str, float]]:
    """One row per non-empty bin: its range, message count, mean score and phishing rate."""
    index = _bin_index(scores, bins)
    rows = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append(
                {
                    "low": b / bins,
                    "high": (b + 1) / bins,
                    "messages": int(mask.sum()),
                    "mean_score": float(scores[mask].mean()),
                    "phishing_rate": float(target[mask].mean()),
                }
            )
    return rows


# --- calibrating ---------------------------------------------------------------------------


def _logit(scores: ArrayLike) -> NDArray[np.float64]:
    p = np.clip(np.asarray(scores, dtype=np.float64), _EPS, 1 - _EPS)
    out: NDArray[np.float64] = np.log(p / (1 - p))
    return out


class ScoreCalibrator(BaseEstimator):  # type: ignore[misc]  # sklearn is untyped
    """A monotone map from a model's score to a probability, learnt on held-out data.

    `sigmoid` is Platt scaling on the logit of the score (two parameters, smooth, hard to overfit);
    `isotonic` is a free non-decreasing step function (more flexible, needs more data).
    """

    def __init__(self, method: Method = "sigmoid") -> None:
        self.method = method

    def fit(self, scores: ArrayLike, target: ArrayLike) -> Self:
        y = np.asarray(target)
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e6).fit(_logit(scores)[:, None], y)
        else:
            self.model_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(
                np.asarray(scores, dtype=np.float64), y
            )
        self.prior_ = float(y.mean())
        return self

    def transform(self, scores: ArrayLike) -> NDArray[np.float64]:
        check_is_fitted(self, "model_")
        if self.method == "sigmoid":
            out: NDArray[np.float64] = self.model_.predict_proba(_logit(scores)[:, None])[:, 1]
        else:
            out = self.model_.predict(np.asarray(scores, dtype=np.float64))
        return np.clip(out, 0.0, 1.0)


def choose_method(
    scores: NDArray[np.float64],
    target: NDArray[np.int64],
    groups: Sequence[str],
    folds: int = 5,
) -> tuple[Method, dict[str, float]]:
    """Pick the calibration method with the lowest cross-validated log loss.

    Folds are grouped by duplicate cluster and all of it happens on validation scores, so the
    choice never sees the test split. Log loss is a proper scoring rule and less noisy than a
    binned ECE on a few thousand messages.
    """
    losses: dict[str, float] = {}
    for method in METHODS:
        predicted = np.empty(len(scores))
        for fit_rows, held_out in GroupKFold(n_splits=folds).split(scores, target, groups):
            calibrator = ScoreCalibrator(method).fit(scores[fit_rows], target[fit_rows])
            predicted[held_out] = calibrator.transform(scores[held_out])
        losses[method] = float(log_loss(target, np.clip(predicted, _EPS, 1 - _EPS), labels=[0, 1]))
    best = min(METHODS, key=lambda m: (losses[m], METHODS.index(m)))
    return best, losses


class CalibratedPipeline(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """A fitted pipeline followed by a fitted `ScoreCalibrator`. This is what gets registered.

    `predict_proba` returns calibrated probabilities at `calibrator.prior_`, the phishing share of
    the calibration data; `raw_scores` gives the pipeline's own scores for comparison.
    """

    def __init__(self, pipeline: Pipeline, calibrator: ScoreCalibrator) -> None:
        self.pipeline = pipeline
        self.calibrator = calibrator

    def fit(self, X: Sequence[Email], y: object = None) -> Self:
        # Both parts arrive fitted (pipeline on train, calibrator on validation): nothing to learn.
        self.classes_ = np.array([0, 1])
        return self

    def raw_scores(self, X: Sequence[Email]) -> NDArray[np.float64]:
        scores: NDArray[np.float64] = self.pipeline.predict_proba(X)[:, 1]
        return scores

    def predict_proba(self, X: Sequence[Email]) -> NDArray[np.float64]:
        calibrated = self.calibrator.transform(self.raw_scores(X))
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X: Sequence[Email]) -> NDArray[np.int64]:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int64)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        return tags


# --- base rate -----------------------------------------------------------------------------


def adjust_prior(probability: ArrayLike, from_prior: float, to_prior: float) -> NDArray[np.float64]:
    """Move a probability from one phishing base rate to another (Bayes' rule on the odds).

    odds' = odds * [to / (1 - to)] / [from / (1 - from)]. With from = 0.571 and to = 0.01, a
    calibrated 0.9 becomes about 0.064: most "90% sure" messages in the corpus would be ham in an
    inbox where only 1% of mail is phishing.
    """
    for prior in (from_prior, to_prior):
        if not 0 < prior < 1:
            raise ValueError(f"priors must be in (0, 1), got {prior}")
    # Only exact 0 and 1 are moved (their odds are 0 and infinite); a wider clip would lose the
    # information a round trip needs near the extremes.
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    odds = p / (1 - p) * (to_prior / (1 - to_prior)) / (from_prior / (1 - from_prior))
    out: NDArray[np.float64] = odds / (1 + odds)
    return out


def precision_at_prior(recall: float, false_positive_rate: float, prior: float) -> float:
    """Share of flagged messages that are phishing, at a given base rate.

    With recall r, false-positive rate f and base rate p: r p / (r p + f (1 - p)). Recall and FPR
    do not depend on the base rate; precision does, which is why it is never reported at 57%.
    """
    flagged = recall * prior + false_positive_rate * (1 - prior)
    return recall * prior / flagged if flagged else math.nan
