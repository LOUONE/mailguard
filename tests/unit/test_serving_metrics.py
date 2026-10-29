"""The Prometheus endpoint: traffic, latency, model decisions, readiness."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from mailguard.config import Settings
from mailguard.serving.app import create_app
from mailguard.serving.model import ServedModel
from tests.models import HAM, PHISH, tiny_served_model


@pytest.fixture(scope="module")
def served() -> ServedModel:
    return tiny_served_model()


@pytest.fixture
def client(served: ServedModel) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(), loader=lambda _: served)) as test_client:
        yield test_client


def samples(client: TestClient) -> dict[tuple[str, frozenset[tuple[str, str]]], float]:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return {
        (s.name, frozenset(s.labels.items())): s.value
        for family in text_string_to_metric_families(response.text)
        for s in family.samples
    }


def value(
    found: dict[tuple[str, frozenset[tuple[str, str]]], float], metric: str, /, **labels: str
) -> float:
    # Positional-only `metric`, so that a label called `name` cannot collide with it.
    return found.get((metric, frozenset(labels.items())), 0.0)


def test_predictions_are_counted_by_label(client: TestClient) -> None:
    for raw in (PHISH.format(i=1), PHISH.format(i=2), HAM.format(i=3)):
        client.post("/v1/predict", json={"raw": raw})

    found = samples(client)

    assert value(found, "mailguard_predictions_total", label="phishing") == 2
    assert value(found, "mailguard_predictions_total", label="legitimate") == 1
    assert value(found, "mailguard_score_count") == 3
    assert value(found, "mailguard_scoring_duration_seconds_count") == 3
    assert value(found, "mailguard_email_bytes_count") == 3


def test_requests_are_counted_by_route_template_and_status(client: TestClient) -> None:
    client.post("/v1/predict", json={"raw": PHISH})
    client.post("/v1/predict", json={})
    client.get("/anything/goes/here")

    found = samples(client)

    ok = value(found, "mailguard_requests_total", route="/v1/predict", method="POST", status="200")
    bad = value(found, "mailguard_requests_total", route="/v1/predict", method="POST", status="422")
    unmatched = value(
        found, "mailguard_requests_total", route="unmatched", method="GET", status="404"
    )
    assert (ok, bad, unmatched) == (1, 1, 1)
    assert value(found, "mailguard_request_duration_seconds_count", route="/v1/predict") == 2


def test_the_served_model_is_described(client: TestClient, served: ServedModel) -> None:
    found = samples(client)

    labels = {"name": "mailguard-lightgbm", "version": "7", "operating_point": "1pct"}
    assert value(found, "mailguard_model_info", **labels) == 1
    assert value(found, "mailguard_model_threshold") == pytest.approx(served.threshold)
    assert value(found, "mailguard_model_ready") == 1


def test_not_ready_without_a_model() -> None:
    def broken(settings: Settings) -> ServedModel:
        raise LookupError("no model")

    with TestClient(create_app(Settings(), loader=broken)) as client:
        assert value(samples(client), "mailguard_model_ready") == 0


def test_two_apps_do_not_share_counters(served: ServedModel) -> None:
    with TestClient(create_app(Settings(), loader=lambda _: served)) as first:
        first.post("/v1/predict", json={"raw": PHISH})
    with TestClient(create_app(Settings(), loader=lambda _: served)) as second:
        assert value(samples(second), "mailguard_score_count") == 0


def test_metrics_stay_out_of_the_api_contract(client: TestClient) -> None:
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]
