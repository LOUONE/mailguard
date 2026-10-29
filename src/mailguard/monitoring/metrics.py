"""Prometheus metrics of the scoring API: traffic, latency, and what the model is deciding.

Each app gets its own registry, so tests (and several apps in one process) never share counters.
Labels are bounded sets (route templates, status codes, two labels), never user data: a label
value per email would also be a memory leak in Prometheus.

The model metrics are the ones an ML operator watches: the score histogram and the share of
mail labelled phishing move when traffic drifts or when a model misbehaves, long before anyone
reports a missed phishing email. Names are documented in docs/monitoring.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

PREFIX: Final = "mailguard"
LATENCY_BUCKETS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
SCORE_BUCKETS: Final = tuple(round(0.05 * i, 2) for i in range(1, 21))
SIZE_BUCKETS: Final = (1e3, 4e3, 16e3, 64e3, 256e3, 1e6, 4e6)
CONTENT_TYPE: Final = CONTENT_TYPE_LATEST


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter(
            f"{PREFIX}_requests", "HTTP requests.", ["route", "method", "status"], registry=r
        )
        self.request_seconds = Histogram(
            f"{PREFIX}_request_duration_seconds",
            "Time from request to response, by route.",
            ["route"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.scoring_seconds = Histogram(
            f"{PREFIX}_scoring_duration_seconds",
            "Parsing, features and model for one email.",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.email_bytes = Histogram(
            f"{PREFIX}_email_bytes", "Size of scored emails.", buckets=SIZE_BUCKETS, registry=r
        )
        self.predictions = Counter(
            f"{PREFIX}_predictions", "Scored emails, by label.", ["label"], registry=r
        )
        self.scores = Histogram(
            f"{PREFIX}_score",
            "Calibrated score at the training base rate.",
            buckets=SCORE_BUCKETS,
            registry=r,
        )
        self.model_info = Gauge(
            f"{PREFIX}_model_info",
            "The served model: always 1, the version is in the labels.",
            ["name", "version", "operating_point"],
            registry=r,
        )
        self.model_threshold = Gauge(
            f"{PREFIX}_model_threshold", "Score above which mail is labelled phishing.", registry=r
        )
        self.model_ready = Gauge(f"{PREFIX}_model_ready", "1 when a model is loaded.", registry=r)
        self.drift_share = Gauge(
            f"{PREFIX}_drift_share", "Share of input features drifting (PSI).", registry=r
        )
        self.drift_detected = Gauge(
            f"{PREFIX}_drift_detected", "1 when the last drift report flagged traffic.", registry=r
        )
        self.feature_psi = Gauge(
            f"{PREFIX}_feature_psi",
            "PSI of each recorded feature against the validation data.",
            ["feature"],
            registry=r,
        )
        self._drift_mtime = 0.0

    def request(self, route: str, method: str, status: int, seconds: float) -> None:
        self.requests.labels(route, method, str(status)).inc()
        self.request_seconds.labels(route).observe(seconds)

    def prediction(self, label: str, score: float, seconds: float, email_bytes: int) -> None:
        self.predictions.labels(label).inc()
        self.scores.observe(score)
        self.scoring_seconds.observe(seconds)
        self.email_bytes.observe(email_bytes)

    def model(self, name: str, version: str, operating_point: str, threshold: float) -> None:
        self.model_info.labels(name, version, operating_point).set(1)
        self.model_threshold.set(threshold)
        self.model_ready.set(1)

    def drift(self, summary_path: Path | None) -> None:
        """Load the drift job's latest summary, when it changed since the last scrape."""
        if summary_path is None or not summary_path.is_file():
            return
        mtime = summary_path.stat().st_mtime
        if mtime == self._drift_mtime:
            return
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:  # being rewritten: next scrape
            return
        self._drift_mtime = mtime
        self.drift_share.set(summary["share_drifted"])
        self.drift_detected.set(float(summary["drifted"]))
        for feature in summary["features"]:
            self.feature_psi.labels(feature["feature"]).set(feature["psi"])

    def render(self) -> bytes:
        return generate_latest(self.registry)
