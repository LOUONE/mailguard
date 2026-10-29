"""Logs are JSON, useful, and never contain what the sender wrote: the canary test."""

from __future__ import annotations

import base64
import json
import logging

import pytest
import structlog
from fastapi.testclient import TestClient

from mailguard.config import Settings
from mailguard.monitoring.logs import REDACTED, configure_logging
from mailguard.serving import app as app_module
from mailguard.serving.app import create_app
from mailguard.serving.model import Prediction, ServedModel
from tests.models import tiny_served_model

CANARY = "zq7canary"
# The canary in every place an email can carry text: names, addresses, subject, body, URL,
# link text, attachment name.
CANARY_EMAIL = f"""\
From: "{CANARY} Bank" <{CANARY}@{CANARY}.example>
To: victim-{CANARY}@example.org
Reply-To: {CANARY}@other.example
Subject: verify your account {CANARY}
Content-Type: multipart/mixed; boundary=B

--B
Content-Type: text/html

<p>Dear {CANARY}, <a href="http://{CANARY}.example/login?u={CANARY}">www.{CANARY}.com</a></p>
--B
Content-Type: application/pdf; name="{CANARY}.pdf"
Content-Disposition: attachment; filename="{CANARY}.pdf"

JVBERi0=
--B--
"""


@pytest.fixture(scope="module")
def served() -> ServedModel:
    return tiny_served_model()


def every_path(client: TestClient) -> None:
    encoded = base64.b64encode(CANARY_EMAIL.encode()).decode()
    client.post("/v1/predict", json={"raw": CANARY_EMAIL})
    client.post("/v1/predict", json={"raw": encoded, "encoding": "base64"})
    client.post("/v1/predict", json={"raw": CANARY_EMAIL, CANARY: CANARY})  # 422
    client.post("/v1/predict", json={"raw": CANARY_EMAIL, "encoding": "base64"})  # 422
    client.post("/v1/predict", json={"raw": CANARY_EMAIL * 2000})  # 413
    client.get(f"/{CANARY}")  # 404: a client can put anything in a URL


def server_lines(out: str) -> list[str]:
    # httpx logs the test client's own requests: client side, not something the API writes.
    return [line for line in out.splitlines() if json.loads(line)["logger"] != "httpx"]


def test_no_email_content_reaches_the_logs(
    served: ServedModel, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(create_app(Settings(max_email_bytes=50_000), loader=lambda _: served)) as c:
        every_path(c)

    lines = server_lines(capsys.readouterr().out)
    assert len([line for line in lines if '"request"' in line]) == 6  # one access line each
    assert not [line for line in lines if CANARY in line]
    assert any('"path": "unmatched"' in line for line in lines)


def test_exceptions_quoting_the_email_are_logged_by_type_only(
    served: ServedModel,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(model: ServedModel, raw: bytes, base_rate: float) -> Prediction:
        raise RuntimeError(f"cannot score {raw[:200]!r}")

    monkeypatch.setattr(app_module, "score", broken)
    app = create_app(Settings(), loader=lambda _: served)
    with TestClient(app, raise_server_exceptions=True) as client:  # nothing may escape the app
        response = client.post("/v1/predict", json={"raw": CANARY_EMAIL})

    assert response.status_code == 500
    out = capsys.readouterr().out
    assert CANARY not in out and "Traceback" not in out
    error = next(json.loads(line) for line in out.splitlines() if "unhandled_error" in line)
    assert error["error_type"] == "RuntimeError" and error["level"] == "error"


def test_one_access_line_per_request_with_numbers_only(
    served: ServedModel, capsys: pytest.CaptureFixture[str]
) -> None:
    with TestClient(create_app(Settings(), loader=lambda _: served)) as client:
        response = client.post(
            "/v1/predict", json={"raw": CANARY_EMAIL}, headers={"x-request-id": "mta-1"}
        )

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    access = [line for line in lines if line["event"] == "request"]
    assert len(access) == 1
    line = access[0]
    assert line["request_id"] == "mta-1" and line["status"] == 200
    assert line["path"] == "/v1/predict" and line["method"] == "POST"
    assert line["label"] == response.json()["label"] and line["model_version"] == "7"
    assert line["email_bytes"] == len(CANARY_EMAIL.encode())
    assert isinstance(line["duration_ms"], float) and isinstance(line["scoring_ms"], float)
    assert {"timestamp", "level", "logger"} <= set(line)
    assert any(entry["event"] == "model_loaded" for entry in lines)


def test_fields_outside_the_allowlist_are_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()

    structlog.get_logger("test").info("scored", subject=CANARY, score=0.9)
    logging.getLogger("third.party").warning("plain stdlib record")

    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["subject"] == REDACTED and lines[0]["score"] == 0.9
    assert lines[1]["event"] == "plain stdlib record" and lines[1]["level"] == "warning"


def test_configuring_twice_keeps_one_handler_and_foreign_handlers() -> None:
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    try:
        configure_logging()
        configure_logging(json=False)

        handlers = logging.getLogger().handlers
        assert foreign in handlers
        assert sum(getattr(h, "_mailguard_handler", False) for h in handlers) == 1
    finally:
        logging.getLogger().removeHandler(foreign)
