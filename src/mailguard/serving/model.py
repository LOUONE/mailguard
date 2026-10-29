"""The model behind the API: loaded once at startup, then plain function calls per request."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from mailguard.config import Settings
from mailguard.models.calibration import CalibratedPipeline, adjust_prior
from mailguard.models.explain import contributions_of, feature_names, top_reasons
from mailguard.models.registry import aliased_version, load_version
from mailguard.parsing import Email
from mailguard.serving.schemas import ModelRef

TOP_REASONS: Final = 5


@dataclass(frozen=True, slots=True)
class Prediction:
    score: float  # calibrated, at the training base rate
    probability: float  # the same, at the requested base rate
    is_phishing: bool
    reasons: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ServedModel:
    name: str
    version: str
    model: CalibratedPipeline
    thresholds: Mapping[str, float]
    operating_point: str = "1pct"
    # Computed once at load: rebuilding tens of thousands of n-gram names was 38% of a request.
    names: NDArray[np.str_] = field(init=False, repr=False)
    features: Pipeline = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.operating_point not in self.thresholds:
            raise LookupError(
                f"{self.name} v{self.version} has no threshold for {self.operating_point}"
            )
        object.__setattr__(self, "names", feature_names(self.model.pipeline))
        object.__setattr__(self, "features", self.model.pipeline[:-1])

    @property
    def threshold(self) -> float:
        return self.thresholds[self.operating_point]

    @property
    def ref(self) -> ModelRef:
        return ModelRef(
            name=self.name,
            version=self.version,
            operating_point=self.operating_point,
            threshold=self.threshold,
        )

    def predict(self, email: Email, base_rate: float) -> Prediction:
        """Score and explain one email in a single pass over the features.

        `predict_proba` then `explain` computed the features twice and called the model twice.
        Here the features are computed once. For LightGBM the TreeSHAP contributions add up to
        the model's own margin, so one call gives the score and its reasons; the linear model
        computes its margin in float32, which the float64 contributions do not reproduce to the
        last digit, so its score comes from its own (cheap) `predict_proba`. Either way the score
        is exactly the evaluated model's. The label uses the FPR threshold, which `base_rate`
        does not move.
        """
        features = self.features.transform([email])
        classifier = self.model.pipeline[-1]
        c = contributions_of(classifier, features, self.names)
        raw = (
            c.probability
            if hasattr(classifier, "booster_")
            else classifier.predict_proba(features)[:, 1]
        )
        score = float(self.model.calibrator.transform(raw)[0])
        probability = float(adjust_prior(score, self.model.calibrator.prior_, base_rate))
        return Prediction(score, probability, score > self.threshold, top_reasons(c, TOP_REASONS))


def load_production(settings: Settings) -> ServedModel:
    """The version behind the registry alias, with the thresholds recorded at training time."""
    version = aliased_version(settings.model_name, settings, settings.model_alias)
    model = load_version(version, settings)
    if not isinstance(model, CalibratedPipeline):
        raise TypeError(f"{version.name} v{version.version} is not a calibrated pipeline")
    return ServedModel(
        version.name, version.version, model, version.thresholds, settings.operating_point
    )
