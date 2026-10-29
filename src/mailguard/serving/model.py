"""The model behind the API: loaded once at startup, then plain function calls per request."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from mailguard.config import Settings
from mailguard.models.calibration import CalibratedPipeline, adjust_prior
from mailguard.models.explain import explain
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

    def __post_init__(self) -> None:
        if self.operating_point not in self.thresholds:
            raise LookupError(
                f"{self.name} v{self.version} has no threshold for {self.operating_point}"
            )

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
        """Score one email. The label uses the FPR threshold, which `base_rate` does not move."""
        score = float(self.model.predict_proba([email])[0, 1])
        probability = float(adjust_prior(score, self.model.calibrator.prior_, base_rate))
        reasons = explain(self.model.pipeline, email, top=TOP_REASONS)
        return Prediction(score, probability, score > self.threshold, reasons)


def load_production(settings: Settings) -> ServedModel:
    """The version behind the registry alias, with the thresholds recorded at training time."""
    version = aliased_version(settings.model_name, settings, settings.model_alias)
    model = load_version(version, settings)
    if not isinstance(model, CalibratedPipeline):
        raise TypeError(f"{version.name} v{version.version} is not a calibrated pipeline")
    return ServedModel(
        version.name, version.version, model, version.thresholds, settings.operating_point
    )
