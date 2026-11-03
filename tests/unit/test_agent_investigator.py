from __future__ import annotations

import json
from typing import Any

import pytest
from structlog.testing import capture_logs

from mailguard.agent.investigator import (
    SYSTEM_PROMPT,
    Investigator,
    classifier_verdict,
    render_email,
)
from mailguard.agent.llm import LLMResponseError, LLMUnavailable, Message, Reply, ToolCall, Usage
from mailguard.agent.tools import Case, Opinion, Reason, Toolbox
from mailguard.agent.tools.offline import OFFLINE_TOOLS
from mailguard.parsing import Attachment, Email, parse_email
from tests.fakes import ScriptedLLM

pytestmark = pytest.mark.anyio

SUBJECT = "Your mailbox quota is full"
EMAIL = parse_email(
    f"""\
From: "IT Desk" <it@mail-quota-help.example>
Subject: {SUBJECT}

Click http://mail-quota-help.example/login to keep your mailbox.
""".encode()
)
OPINION = Opinion(
    score=0.55,
    probability=0.012,
    is_phishing=True,
    threshold=0.29,
    reasons=(Reason("text__word:mailbox", 0.8), Reason("url__url_login_keyword", 0.4)),
)
CASE = Case(EMAIL, OPINION)
TOOLBOX = Toolbox.of(OFFLINE_TOOLS)
URL = "http://mail-quota-help.example/login"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def calls(*names: str, args: dict[str, Any] | None = None) -> Reply:
    tool_calls = tuple(
        ToolCall(id=f"c{i}", name=name, arguments=args or {}) for i, name in enumerate(names)
    )
    return Reply(
        message=Message(role="assistant", tool_calls=tool_calls),
        usage=Usage(prompt_tokens=100, completion_tokens=10),
    )


def says(text: str) -> Reply:
    return Reply(
        message=Message(role="assistant", content=text),
        usage=Usage(prompt_tokens=50, completion_tokens=5),
    )


def verdict(
    label: str = "phishing", sources: tuple[str, ...] = ("analyze_url", "email"), **extra: Any
) -> Reply:
    body = {
        "label": label,
        "confidence": 0.9,
        "evidence": [{"source": s, "finding": f"finding from {s}"} for s in sources],
        "explanation": "It asks for your password on a site that is not your provider's.",
    } | extra
    return says(json.dumps(body))


async def investigate(llm: ScriptedLLM, **kwargs: Any) -> Any:
    return await Investigator(llm, TOOLBOX, **kwargs).investigate(CASE)


# --- the happy path --------------------------------------------------------------------------


async def test_tools_then_verdict() -> None:
    llm = ScriptedLLM([calls("analyze_url", args={"url": URL}), says("Done."), verdict()])
    result = await investigate(llm)
    assert result.source == "agent" and result.fallback_reason is None
    assert result.verdict.label == "phishing"
    assert [e.source for e in result.verdict.evidence] == ["analyze_url", "email"]
    assert result.tool_calls == ("analyze_url",)
    assert result.steps == 2
    assert (result.prompt_tokens, result.completion_tokens) == (200, 20)
    assert result.model == "scripted"
    assert result.seconds >= 0


async def test_what_the_model_is_sent() -> None:
    llm = ScriptedLLM([calls("analyze_url", args={"url": URL}), says("Done."), verdict()])
    await investigate(llm)
    first, second, last = llm.calls
    system, user = first.messages
    assert system.content == SYSTEM_PROMPT
    assert "<<<EMAIL" in user.content and "EMAIL>>>" in user.content
    assert SUBJECT in user.content
    assert str(OPINION.score) not in user.content  # no anchoring on the classifier
    assert {t.name for t in first.tools} == set(TOOLBOX.tools)
    assert first.schema is None
    tool_answer = second.messages[-1]
    assert (tool_answer.role, tool_answer.tool_call_id, tool_answer.name) == (
        "tool",
        "c0",
        "analyze_url",
    )
    assert json.loads(tool_answer.content)["registrable_domain"] == "mail-quota-help.example"
    assert last.tools == () and last.schema is not None
    assert last.schema["properties"]["label"]["enum"] == ["phishing", "legitimate"]


async def test_verdict_in_code_fences_is_read() -> None:
    fenced = says("```json\n" + verdict().message.content + "\n```")
    result = await investigate(
        ScriptedLLM([calls("analyze_url", args={"url": URL}), says("ok"), fenced])
    )
    assert result.source == "agent"


async def test_no_tool_needed() -> None:
    result = await investigate(
        ScriptedLLM([says("Nothing to check."), verdict("legitimate", ("email",))])
    )
    assert (result.source, result.steps, result.tool_calls) == ("agent", 1, ())
    assert result.verdict.label == "legitimate"


# --- bounds ----------------------------------------------------------------------------------


async def test_step_limit_forces_the_verdict() -> None:
    loop = [calls("check_auth_headers") for _ in range(3)]
    llm = ScriptedLLM([*loop, verdict(sources=("check_auth_headers",))])
    result = await investigate(llm, max_steps=3)
    assert result.steps == 3
    assert len(llm.calls) == 4
    assert result.tool_calls == ("check_auth_headers",) * 3


async def test_calls_per_step_are_capped_and_unanswered_calls_dropped() -> None:
    llm = ScriptedLLM(
        [calls(*["inspect_attachments"] * 6), says("ok"), verdict(sources=("email",))]
    )
    result = await investigate(llm, max_calls_per_step=2)
    assert result.tool_calls == ("inspect_attachments",) * 2
    assistant = llm.calls[1].messages[2]
    assert len(assistant.tool_calls) == 2
    assert [m.tool_call_id for m in llm.calls[1].messages[3:]] == ["c0", "c1"]


async def test_an_unknown_tool_is_answered_with_an_error() -> None:
    llm = ScriptedLLM([calls("open_attachment"), says("ok"), verdict(sources=("email",))])
    result = await investigate(llm)
    answer = llm.calls[1].messages[-1]
    assert "unknown tool 'open_attachment'" in json.loads(answer.content)["error"]
    assert result.source == "agent"


# --- invalid verdicts ------------------------------------------------------------------------


async def test_one_retry_after_an_invalid_verdict() -> None:
    llm = ScriptedLLM([says("ok"), says('{"label": "maybe"}'), verdict(sources=("email",))])
    result = await investigate(llm)
    assert result.source == "agent"
    retry = llm.calls[2].messages[-1].content
    assert retry.startswith("That verdict was not valid:")
    assert "label" in retry


async def test_two_invalid_verdicts_fall_back_to_the_classifier() -> None:
    llm = ScriptedLLM([says("ok"), says("I think it is phishing."), verdict(confidence=7)])
    result = await investigate(llm)
    assert result.source == "classifier"
    assert result.fallback_reason is not None and result.fallback_reason.startswith(
        "invalid verdict"
    )
    assert result.verdict.label == "phishing"


async def test_evidence_from_tools_never_called_is_dropped() -> None:
    llm = ScriptedLLM([says("ok"), verdict(sources=("domain_age", "email"))])
    result = await investigate(llm)
    assert [e.source for e in result.verdict.evidence] == ["email"]
    assert result.discarded_evidence == 1


async def test_entirely_invented_evidence_is_retried_then_refused() -> None:
    invented = verdict(sources=("domain_age", "lookup_reputation"))
    llm = ScriptedLLM([says("ok"), invented, invented])
    result = await investigate(llm)
    assert result.source == "classifier"
    assert "did not call (domain_age, lookup_reputation)" in (result.fallback_reason or "")


# --- failures --------------------------------------------------------------------------------


@pytest.mark.parametrize("failing_call", [0, 2])
async def test_llm_failure_falls_back(failing_call: int) -> None:
    script: list[Reply | Exception] = [calls("check_auth_headers"), says("ok"), verdict()]
    script[failing_call] = LLMUnavailable("ConnectError from http://127.0.0.1:11434/")
    llm = ScriptedLLM(script[: failing_call + 1])  # type: ignore[arg-type]
    result = await investigate(llm)
    assert result.source == "classifier"
    assert result.fallback_reason == "LLMUnavailable: ConnectError from http://127.0.0.1:11434/"
    assert result.steps == (0 if failing_call == 0 else 2)


async def test_unusable_llm_answer_falls_back() -> None:
    result = await investigate(ScriptedLLM([LLMResponseError("response is not JSON")]))
    assert result.fallback_reason == "LLMResponseError: response is not JSON"


def test_classifier_verdict_keeps_the_classifier_view() -> None:
    v = classifier_verdict(CASE)
    assert (v.label, v.confidence) == ("phishing", 0.55)
    assert [e.finding for e in v.evidence] == [
        "text__word:mailbox (+0.80)",
        "url__url_login_keyword (+0.40)",
    ]
    assert "classifier" in v.explanation


def test_classifier_verdict_without_reasons() -> None:
    v = classifier_verdict(
        Case(Email(), Opinion(score=0.2, probability=0.002, is_phishing=False, threshold=0.29))
    )
    assert (v.label, v.confidence) == ("legitimate", 0.8)
    assert v.evidence[0].finding == "score 0.200"


# --- what is logged and rendered -------------------------------------------------------------


async def test_the_log_line_holds_no_email_content() -> None:
    llm = ScriptedLLM([calls("analyze_url", args={"url": URL}), says("ok"), verdict()])
    with capture_logs() as logs:
        await investigate(llm)
    (line,) = [entry for entry in logs if entry["event"] == "investigation"]
    assert line["tool_calls"] == ["analyze_url"] and line["source"] == "agent"
    text = json.dumps(line)
    assert SUBJECT not in text and "mail-quota-help" not in text and "password" not in text


def test_render_email_is_bounded() -> None:
    email = Email(
        subject="s",
        text_body="x" * 7000,
        urls=tuple(f"https://u{i}.example/" for i in range(25)),
        attachments=(Attachment(filename=None, content_type="image/png", size=1, sha256="0" * 64),),
    )
    shown = render_email(email)
    assert "[... 1000 characters cut]" in shown
    assert "5 more" in shown
    assert "(no name) (image/png)" in shown
    assert "Date: -" in shown


def test_render_bare_email() -> None:
    shown = render_email(Email())
    assert "Links:\n(none)" in shown and "Attachments:\n(none)" in shown
