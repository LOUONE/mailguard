from __future__ import annotations

import json
from typing import Any

import pytest

from mailguard.agent.llm import ToolCall
from mailguard.agent.tools import (
    MAX_RESULT_CHARS,
    Arguments,
    Case,
    NoArguments,
    Opinion,
    Output,
    Reason,
    Tool,
    Toolbox,
    clip,
)
from mailguard.agent.tools.offline import (
    ANALYZE_URL,
    CHECK_AUTH_HEADERS,
    GET_CLASSIFIER_EXPLANATION,
    INSPECT_ATTACHMENTS,
    OFFLINE_TOOLS,
    brand_signals,
    deconfuse,
    osa_distance,
)
from mailguard.parsing import Attachment, Email, parse_email

pytestmark = pytest.mark.anyio

OPINION = Opinion(
    score=0.62,
    probability=0.0123,
    is_phishing=True,
    threshold=0.29,
    reasons=tuple(Reason(f"f{i}", 0.5 - i / 10) for i in range(10)),
)
RAW = b"""\
Authentication-Results: mx.mailguard.example; spf=fail smtp.mailfrom=evil.example;
 dkim=none (no signature); dmarc=fail header.from=paypal.com
Authentication-Results: forged.example; spf=pass; dkim=pass; dmarc=pass
From: "PayPal Service" <service@paypa1-secure.example>
Reply-To: help@collector.example
Return-Path: <bounce@evil.example>
Subject: Account limited
Content-Type: text/html

<p>Your account is limited.
<a href="http://paypa1-secure.example/login?id=1">www.paypal.com</a>
<a href="https://www.paypal.com/help">help</a></p>
"""


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def case(email: Email | None = None, authserv: str | None = None) -> Case:
    return Case(email or parse_email(RAW), OPINION, trusted_authserv_id=authserv)


async def call(tool: Tool[Any, Any], c: Case, **arguments: Any) -> dict[str, Any]:
    result = await Toolbox.of([tool]).call(c, ToolCall(id="1", name=tool.name, arguments=arguments))
    assert result.ok, result.content
    parsed: dict[str, Any] = json.loads(result.content)
    return parsed


# --- the toolbox -----------------------------------------------------------------------------


class Echo(Arguments):
    text: str


class Echoed(Output):
    text: str


async def echo(c: Case, a: Echo) -> Echoed:
    return Echoed(text=a.text)


async def broken(c: Case, a: NoArguments) -> Echoed:
    raise RuntimeError("secret detail")


ECHO = Tool("echo", "Echo text.", Echo, echo)


def test_specs_carry_the_argument_schema() -> None:
    (spec,) = Toolbox.of([ECHO]).specs
    assert spec.name == "echo"
    assert spec.parameters["properties"]["text"]["type"] == "string"
    assert spec.parameters["required"] == ["text"]
    assert spec.parameters["additionalProperties"] is False
    assert "title" not in spec.parameters


def test_every_offline_tool_has_a_valid_spec() -> None:
    box = Toolbox.of(OFFLINE_TOOLS)
    assert [s.name for s in box.specs] == [
        "check_auth_headers",
        "analyze_url",
        "inspect_attachments",
        "get_classifier_explanation",
    ]
    assert all(s.description and s.parameters["type"] == "object" for s in box.specs)


def test_duplicate_names_are_refused() -> None:
    with pytest.raises(ValueError, match="two tools named echo"):
        Toolbox.of([ECHO, ECHO])


@pytest.mark.parametrize(
    ("name", "arguments", "error"),
    [
        ("nope", {}, "unknown tool 'nope'; tools: echo, broken"),
        ("echo", {}, "invalid arguments: text: Field required"),
        ("echo", {"text": "a", "extra": 1}, "extra: Extra inputs are not permitted"),
        ("broken", {}, "broken failed (RuntimeError)"),
    ],
)
async def test_failures_come_back_as_errors_for_the_model(
    name: str, arguments: dict[str, Any], error: str
) -> None:
    box = Toolbox.of([ECHO, Tool("broken", "Fails.", NoArguments, broken)])
    result = await box.call(case(), ToolCall(id="7", name=name, arguments=arguments))
    assert not result.ok
    assert result.call.id == "7"
    assert error in json.loads(result.content)["error"]
    assert "secret detail" not in result.content


async def test_long_results_are_truncated() -> None:
    result = await Toolbox.of([ECHO]).call(
        case(), ToolCall(id="1", name="echo", arguments={"text": "x" * 10_000})
    )
    body = json.loads(result.content)
    assert result.ok and body["truncated"] is True
    assert len(body["partial"]) == MAX_RESULT_CHARS


def test_clip() -> None:
    assert clip(None) is None
    assert clip("short") == "short"
    assert clip("x" * 300, 10) == "x" * 10 + "…"


# --- check_auth_headers ----------------------------------------------------------------------


async def test_auth_headers_from_the_trusted_server() -> None:
    out = await call(CHECK_AUTH_HEADERS, case(authserv="mx.mailguard.example"))
    assert out["trusted"] is True
    assert (out["spf"], out["dkim"], out["dmarc"]) == ("fail", "none", "fail")
    assert out["from_domain"] == "paypa1-secure.example"
    assert out["reply_to_differs"] and out["return_path_differs"]
    assert out["display_name_names_other_brand"] is True


async def test_a_forged_header_is_ignored_when_a_server_is_trusted() -> None:
    out = await call(CHECK_AUTH_HEADERS, case(authserv="forged.example"))
    assert out["dmarc"] == "pass"  # that server's header, since it is the one trusted
    other = await call(CHECK_AUTH_HEADERS, case(authserv="mx.other.example"))
    assert other["header_found"] is False and other["trusted"] is False
    assert other["spf"] is None


async def test_without_a_trusted_server_the_topmost_header_is_untrusted() -> None:
    out = await call(CHECK_AUTH_HEADERS, case())
    assert out["header_found"] is True
    assert out["trusted"] is False
    assert out["authserv_id"] == "mx.mailguard.example"


async def test_auth_headers_on_a_bare_message() -> None:
    out = await call(CHECK_AUTH_HEADERS, case(Email()))
    assert out["header_found"] is False
    assert out["from_domain"] is None
    assert out["reply_to_differs"] is False


# --- analyze_url -----------------------------------------------------------------------------


async def test_lookalike_link_behind_a_brand_text() -> None:
    out = await call(ANALYZE_URL, case(), url="http://paypa1-secure.example/login?id=1")
    assert out["in_email"] and out["parsed"]
    assert out["registrable_domain"] == "paypa1-secure.example"
    assert out["keywords"] == ["login", "secure"]
    assert {"brand": "paypal", "kind": "combosquat"} in out["brands"]
    assert out["shown_as"] == ["www.paypal.com"]
    assert out["shown_domain_differs"] is True


async def test_the_brand_own_link_is_recognised_as_such() -> None:
    out = await call(ANALYZE_URL, case(), url="https://www.paypal.com/help")
    assert out["brands"] == [{"brand": "paypal", "kind": "brand_domain"}]
    assert out["shown_domain_differs"] is False
    assert out["subdomain_depth"] == 1


async def test_url_not_in_the_email_is_still_analysed_but_flagged() -> None:
    out = await call(ANALYZE_URL, case(), url="http://198.51.100.7:8080/x")
    assert out["in_email"] is False
    assert out["is_ip"] and out["non_standard_port"]
    assert out["brands"] == []


async def test_userinfo_shortener_and_suspicious_tld() -> None:
    trick = await call(ANALYZE_URL, case(), url="http://www.paypal.com@evil.top/verify")
    assert trick["has_userinfo"] and trick["registrable_domain"] == "evil.top"
    assert trick["suspicious_tld"] is True
    short = await call(ANALYZE_URL, case(), url="https://bit.ly/3xYz")
    assert short["shortener"] is True


async def test_punycode_homoglyph_domain() -> None:
    # "paypal" with Cyrillic a (U+0430) and er (U+0440), written as punycode.
    host = (chr(0x0440) + chr(0x0430) + "yp" + chr(0x0430) + "l").encode("idna").decode()
    out = await call(ANALYZE_URL, case(), url=f"https://{host}.com/")
    assert out["punycode"] is True
    assert out["unicode_host"] != out["host"]
    assert out["brands"] == [{"brand": "paypal", "kind": "homoglyph"}]


async def test_unparseable_url() -> None:
    out = await call(ANALYZE_URL, case(), url="mailto:someone@example.org")
    assert (out["parsed"], out["in_email"], out["host"]) == (False, False, None)
    assert out["brands"] == []


async def test_url_argument_is_bounded() -> None:
    box = Toolbox.of([ANALYZE_URL])
    result = await box.call(case(), ToolCall(id="1", name="analyze_url", arguments={"url": ""}))
    assert not result.ok


@pytest.mark.parametrize(
    ("domain", "rest", "expected"),
    [
        ("paypal.com", "", [("paypal", "brand_domain")]),
        ("paypa1.com", "", [("paypal", "typosquat")]),
        ("paypall.com", "", [("paypal", "typosquat")]),
        ("rnicrosoft.com", "", [("microsoft", "typosquat")]),
        ("micorsoft.com", "", [("microsoft", "typosquat")]),
        ("secure-paypal-login.example", "", [("paypal", "combosquat")]),
        ("securepaypal.example", "", [("paypal", "combosquat")]),
        ("evil.example", "paypal.com.login. /webscr", [("paypal", "brand_outside_domain")]),
        ("evil.example", " /office365/login", [("office365", "brand_outside_domain")]),
        ("dhs.gov", "", []),  # one edit from "dhl", but short brands need an exact match
        ("purchase.example", "", []),  # "chase" inside a word
        ("bankofamerica.co.uk", "", [("bankofamerica", "brand_domain")]),
    ],
)
def test_brand_signals(domain: str, rest: str, expected: list[tuple[str, str]]) -> None:
    assert [(s.brand, s.kind) for s in brand_signals(domain, rest)] == expected


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("paypal", "paypal", 0), ("paypal", "paypa", 1), ("paypal", "papyal", 1), ("", "abc", 3)],
)
def test_osa_distance(a: str, b: str, distance: int) -> None:
    assert osa_distance(a, b) == distance == osa_distance(b, a)


def test_deconfuse() -> None:
    assert deconfuse("PAYPA1") == "paypal"
    assert deconfuse("rnicrosoft") == "microsoft"
    assert deconfuse(chr(0x03BF) + "utl" + chr(0x03BF) + "ok") == "outlook"


# --- inspect_attachments ---------------------------------------------------------------------


def attachment(name: str | None, content_type: str, detected: str | None) -> Attachment:
    return Attachment(
        filename=name, content_type=content_type, detected_type=detected, size=42, sha256="0" * 64
    )


async def test_attachments_metadata_and_red_flags() -> None:
    email = Email(
        attachments=(
            attachment("Invoice.PDF.exe", "application/octet-stream", "exe"),
            attachment("report.pdf", "application/pdf", "zip"),
            attachment("photos.zip", "application/zip", "zip"),
            attachment(None, "image/png", "image"),
        )
    )
    out = await call(INSPECT_ATTACHMENTS, case(email))
    exe, fake_pdf, archive, unnamed = out["attachments"]
    assert out["count"] == 4
    assert exe["extensions"] == ["pdf", "exe"]
    assert exe["risky_extension"] and exe["double_extension"]
    assert fake_pdf["type_mismatch"] and not fake_pdf["risky_extension"]
    assert archive["archive"] and not archive["type_mismatch"]
    assert unnamed["filename"] is None and unnamed["extensions"] == []


async def test_attachment_list_is_capped() -> None:
    email = Email(attachments=tuple(attachment(f"a{i}.txt", "text/plain", None) for i in range(30)))
    out = await call(INSPECT_ATTACHMENTS, case(email))
    assert out["count"] == 30 and len(out["attachments"]) == 20


# --- get_classifier_explanation --------------------------------------------------------------


async def test_classifier_explanation() -> None:
    out = await call(GET_CLASSIFIER_EXPLANATION, case())
    assert out["label"] == "phishing"
    assert (out["score"], out["probability"], out["threshold"]) == (0.62, 0.0123, 0.29)
    assert len(out["reasons"]) == 8
    assert out["reasons"][0] == {"feature": "f0", "contribution": 0.5}


async def test_classifier_explanation_of_a_legitimate_verdict() -> None:
    c = Case(Email(), Opinion(score=0.1, probability=0.001, is_phishing=False, threshold=0.29))
    out = await call(GET_CLASSIFIER_EXPLANATION, c)
    assert out["label"] == "legitimate" and out["reasons"] == []


async def test_broken_punycode_is_shown_as_written() -> None:
    out = await call(ANALYZE_URL, case(), url="http://xn--a.example/")
    assert out["punycode"] is True
    assert out["unicode_host"] is None
