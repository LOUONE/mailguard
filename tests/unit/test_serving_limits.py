"""Hostile input: size limits, timeouts, one error shape, no echoed content, and fuzzing."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from mailguard.config import Settings
from mailguard.serving import app as app_module
from mailguard.serving.app import create_app
from mailguard.serving.limits import content_length
from mailguard.serving.model import Prediction, ServedModel
from tests.models import PHISH, tiny_served_model
from tests.strategies import email_like

CANARY = "CANARY-7f3a9c2e"  # must never come back in a response
SMALL = Settings(max_email_bytes=2000, scoring_timeout_s=5.0)


@pytest.fixture(scope="module")
def served() -> ServedModel:
    return tiny_served_model()


@pytest.fixture(scope="module")
def client(served: ServedModel) -> Iterator[TestClient]:
    app = create_app(SMALL, loader=lambda _: served)
    # Server errors become responses, as a real client would see them, instead of raising.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def assert_error_shape(response_json: dict[str, object], code: str) -> None:
    error = response_json["error"]
    assert isinstance(error, dict)
    assert error["code"] == code and error["message"] and error["request_id"]


# --- size ----------------------------------------------------------------------------------


def test_oversized_email_is_413(client: TestClient) -> None:
    response = client.post("/v1/predict", json={"raw": "x" * 2001})

    assert response.status_code == 413
    assert_error_shape(response.json(), "payload_too_large")


def test_announced_oversized_body_is_refused_before_reading(client: TestClient) -> None:
    body = json.dumps({"raw": "x" * 10_000_000}).encode()

    response = client.post(
        "/v1/predict", content=body, headers={"content-type": "application/json"}
    )

    assert response.status_code == 413
    assert "request body over" in response.json()["error"]["message"]


def test_streamed_oversized_body_is_cut_off(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:  # no Content-Length: the server only sees chunks
        yield b'{"raw": "'
        for _ in range(200):
            yield b"x" * 10_000
        yield b'"}'

    response = client.post(
        "/v1/predict", content=chunks(), headers={"content-type": "application/json"}
    )

    assert response.status_code == 413


def test_email_at_the_limit_is_scored(client: TestClient) -> None:
    raw = PHISH.format(i=1)
    raw += " " * (2000 - len(raw.encode()))

    assert client.post("/v1/predict", json={"raw": raw}).status_code == 200


# --- time ----------------------------------------------------------------------------------


def test_slow_scoring_times_out_with_503(
    served: ServedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    def slow(model: ServedModel, raw: bytes, base_rate: float) -> Prediction:
        time.sleep(0.5)
        return Prediction(0.0, 0.0, False)

    monkeypatch.setattr(app_module, "score", slow)
    settings = Settings(scoring_timeout_s=0.05)
    with TestClient(create_app(settings, loader=lambda _: served)) as slow_client:
        response = slow_client.post("/v1/predict", json={"raw": PHISH})

    assert response.status_code == 503
    assert "scoring took over" in response.json()["error"]["message"]


# --- errors --------------------------------------------------------------------------------


def test_validation_errors_do_not_echo_the_email(client: TestClient) -> None:
    for body in (
        {"raw": CANARY, "unknown": CANARY},
        {"raw": CANARY + "!", "encoding": "base64"},
        {"raw": CANARY, "base_rate": 7},
    ):
        response = client.post("/v1/predict", json=body)

        assert response.status_code == 422
        assert CANARY not in response.text
        assert_error_shape(response.json(), "invalid_request")
        assert response.json()["error"]["details"]


def test_unexpected_errors_are_generic_500s(
    served: ServedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(model: ServedModel, raw: bytes, base_rate: float) -> Prediction:
        raise RuntimeError(f"failed on {raw!r}")  # an exception that quotes the email

    monkeypatch.setattr(app_module, "score", broken)
    app = create_app(Settings(), loader=lambda _: served)
    with TestClient(app, raise_server_exceptions=False) as broken_client:
        response = broken_client.post("/v1/predict", json={"raw": CANARY})

    assert response.status_code == 500
    assert CANARY not in response.text and "Traceback" not in response.text
    assert response.json()["error"]["message"] == "internal error"


@pytest.mark.parametrize(
    ("method", "path", "code", "status"),
    [("get", "/nowhere", "not_found", 404), ("get", "/v1/predict", "method_not_allowed", 405)],
)
def test_routing_errors_share_the_shape(
    client: TestClient, method: str, path: str, code: str, status: int
) -> None:
    response = client.request(method, path)

    assert response.status_code == status
    assert_error_shape(response.json(), code)


# --- request ids ---------------------------------------------------------------------------


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    ok = client.get("/health")
    error = client.post("/v1/predict", json={})

    assert len(ok.headers["x-request-id"]) == 32
    assert error.json()["error"]["request_id"] == error.headers["x-request-id"]


def test_a_well_formed_client_id_is_kept_and_a_bad_one_replaced(client: TestClient) -> None:
    kept = client.get("/health", headers={"x-request-id": "mta-42.queue_7"})
    replaced = client.get("/health", headers={"x-request-id": "<script>"})

    assert kept.headers["x-request-id"] == "mta-42.queue_7"
    assert replaced.headers["x-request-id"] != "<script>"


# --- fuzzing -------------------------------------------------------------------------------


@given(raw=st.one_of(email_like, st.binary(max_size=1500)))
def test_fuzzed_emails_never_cause_a_server_error(client: TestClient, raw: bytes) -> None:
    body = {"raw": base64.b64encode(raw).decode(), "encoding": "base64"}

    response = client.post("/v1/predict", json=body)

    expected = (422,) if not raw else (200, 413)  # an empty message is not an email
    assert response.status_code in expected, response.text
    if response.status_code == 200:
        assert response.json()["label"] in ("phishing", "legitimate")


@given(
    body=st.dictionaries(
        st.sampled_from(["raw", "encoding", "base_rate", "extra"]),
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=50),
        ),
    )
)
def test_fuzzed_request_bodies_get_200_or_422(client: TestClient, body: dict[str, object]) -> None:
    response = client.post("/v1/predict", json=body)

    assert response.status_code in (200, 413, 422), response.text


def test_a_malformed_content_length_is_ignored_not_trusted() -> None:
    assert content_length({"headers": [(b"content-length", b"12ab")]}) is None
    assert content_length({"headers": [(b"content-length", b"12")]}) == 12
