"""A tiny calibrated model trained on made-up messages, for the serving and registry tests."""

from __future__ import annotations

import numpy as np

from mailguard.models.calibration import CalibratedPipeline
from mailguard.models.config import BaselineConfig, LightGBMConfig, LogRegConfig, TextConfig
from mailguard.models.pipelines import lightgbm_pipeline
from mailguard.models.train import SplitData, calibrate, operating_thresholds
from mailguard.parsing import parse_email
from mailguard.serving.model import ServedModel

TINY_TEXT = TextConfig(word_max_features=60, char_max_features=60, min_df=1)
TINY = BaselineConfig(
    logreg=LogRegConfig(text=TINY_TEXT),
    lightgbm=LightGBMConfig(n_estimators=10, min_child_samples=1, n_jobs=1, text=TINY_TEXT),
)

HAM = "From: a{i}@example.org\nSubject: lunch\n\nLunch at noon {i}? Bring the slides.\n"
PHISH = (
    "From: PayPal <x{i}@evil.example>\nReply-To: r@other.example\nSubject: verify your account\n"
    "Content-Type: text/html\n\n"
    '<a href="http://198.51.100.{i}/login">www.paypal.com</a> Verify your account password.\n'
)


def messages(n: int, offset: int = 0) -> SplitData:
    raws = [(PHISH if i % 2 else HAM).format(i=i + offset) for i in range(n)]
    target = np.array([i % 2 for i in range(n)], dtype=np.int64)
    return SplitData(
        emails=[parse_email(r.encode()) for r in raws],
        target=target,
        labels=np.where(target == 1, "phishing", "ham").astype(np.str_),
        clusters=[f"c{i + offset}" for i in range(n)],
    )


def tiny_model() -> tuple[CalibratedPipeline, dict[str, float]]:
    """A calibrated LightGBM and its operating thresholds, as `make train` would produce."""
    train, val = messages(40), messages(20, offset=100)
    pipeline = lightgbm_pipeline(TINY).fit(train.emails, train.target)
    model, _ = calibrate(pipeline, val)
    return model, operating_thresholds(model, val)


def tiny_served_model(version: str = "7") -> ServedModel:
    model, thresholds = tiny_model()
    return ServedModel("mailguard-lightgbm", version, model, thresholds)
