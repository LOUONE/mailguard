"""Contract tests of the scoring API, with a tiny model and no MLflow."""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mailguard.config import Settings
from mailguard.serving.app import create_app
from mailguard.serving.model import ServedModel
from tests.models import HAM, PHISH, tiny_served_model

OPENAPI = Path("docs/api/openapi.json")
PHISHING = PHISH.format(i=7)
LEGITIMATE = HAM.format(i=7)


@pytest.fixture(scope="module")
def served() -> ServedModel:
    return tiny_served_model()


@pytest.fixture
def client(served: ServedModel) -> Iterator[TestClient]:
    app = create_app(Settings(base_rate=0.02), loader=lambda _: served)
    with TestClient(app) as test_client:  # `with` runs the startup that loads the model
        yield test_client


def broken_loader(settings: Settings) -> ServedModel:
    raise LookupError("mailguard-lightgbm has no @production version")


# --- health --------------------------------------------------------------------------------


def test_health_reports_the_served_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["version"] == "7" and body["model"]["operating_point"] == "1pct"


def test_health_is_not_ready_when_the_model_cannot_load() -> None:
    with TestClient(create_app(Settings(), loader=broken_loader)) as client:
        health = client.get("/health")
        predict = client.post("/v1/predict", json={"raw": PHISHING})

    assert health.status_code == 503
    assert health.json()["status"] == "unavailable"
    assert "no @production version" in health.json()["detail"]
    assert predict.status_code == 503


# --- predict -------------------------------------------------------------------------------


def test_predict_returns_the_v1_contract(client: TestClient) -> None:
    response = client.post("/v1/predict", json={"raw": PHISHING})

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "v1"
    assert body["label"] == "phishing"
    assert 0 <= body["probability"] <= body["score"] <= 1  # 2% phishing is below the corpus rate
    assert body["base_rate"] == 0.02  # from the settings
    assert body["model"]["threshold"] < body["score"]
    assert 0 < len(body["reasons"]) <= 5
    assert {"feature", "group", "contribution", "direction"} == set(body["reasons"][0])


def test_legitimate_mail_is_labelled_legitimate(client: TestClient) -> None:
    body = client.post("/v1/predict", json={"raw": LEGITIMATE}).json()

    assert body["label"] == "legitimate"
    assert body["score"] <= body["model"]["threshold"]


def test_base_rate_moves_the_probability_not_the_decision(client: TestClient) -> None:
    low = client.post("/v1/predict", json={"raw": PHISHING, "base_rate": 0.001}).json()
    high = client.post("/v1/predict", json={"raw": PHISHING, "base_rate": 0.5}).json()

    assert low["score"] == high["score"] and low["label"] == high["label"]
    assert low["probability"] < high["probability"]


def test_base64_and_text_encodings_agree(client: TestClient) -> None:
    encoded = base64.b64encode(PHISHING.encode()).decode()

    text = client.post("/v1/predict", json={"raw": PHISHING}).json()
    b64 = client.post("/v1/predict", json={"raw": encoded, "encoding": "base64"}).json()

    assert text == b64


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"raw": ""},
        {"raw": PHISHING, "unknown": 1},
        {"raw": PHISHING, "base_rate": 0},
        {"raw": PHISHING, "base_rate": 1.5},
        {"raw": "not base64!", "encoding": "base64"},
        {"raw": PHISHING, "encoding": "utf-16"},
    ],
)
def test_invalid_requests_are_refused(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/v1/predict", json=body).status_code == 422


# --- OpenAPI -------------------------------------------------------------------------------


def test_committed_openapi_document_is_up_to_date() -> None:
    """`uv run python scripts/export_openapi.py` regenerates it after a schema change."""
    current = create_app(Settings(), loader=broken_loader).openapi()

    assert json.loads(OPENAPI.read_text(encoding="utf-8")) == current
