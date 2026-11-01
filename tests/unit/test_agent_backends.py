from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from mailguard.agent.backends import Ollama, OpenAICompatible, make_client
from mailguard.agent.llm import (
    LLMClient,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailable,
    Message,
    Reply,
    ToolCall,
    ToolSpec,
)
from mailguard.config import Settings
from tests.fakes import ScriptedLLM

pytestmark = pytest.mark.anyio

FIXTURES = Path(__file__).parents[1] / "fixtures" / "llm"
BACKENDS = ("ollama", "openai")
SCENARIOS = ("plain", "tool_call", "tool_result", "schema")
Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def no_local_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A developer's `.env` or `MAILGUARD_LLM_*` variables must not change these tests."""
    monkeypatch.chdir(tmp_path)
    for var in (
        "BACKEND",
        "BASE_URL",
        "MODEL",
        "API_KEY",
        "TIMEOUT_S",
        "MAX_RETRIES",
        "CONTEXT_TOKENS",
        "TEMPERATURE",
        "THINK",
    ):
        monkeypatch.delenv(f"MAILGUARD_LLM_{var}", raising=False)


def build(backend: str, handler: Handler, model: str = "m", **kwargs: Any) -> LLMClient:
    transport = httpx.MockTransport(handler)
    if backend == "ollama":
        return Ollama(model=model, transport=transport, **kwargs)
    return OpenAICompatible(
        model=model, base_url="http://127.0.0.1:11434/v1", transport=transport, **kwargs
    )


def load(backend: str, scenario: str) -> dict[str, Any]:
    fixture: dict[str, Any] = json.loads(
        (FIXTURES / backend / f"{scenario}.json").read_text(encoding="utf-8")
    )
    return fixture


async def replay(backend: str, scenario: str) -> Reply:
    """Run the recorded input through the client against the recorded server answer."""
    fixture = load(backend, scenario)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(fixture["status"], json=fixture["response"])

    client = build(backend, handler, model=fixture["request"]["model"])
    given = fixture["input"]
    reply = await client.chat(
        [Message.model_validate(m) for m in given["messages"]],
        tools=[ToolSpec.model_validate(t) for t in given["tools"]],
        schema=given["schema"],
    )
    await client.aclose()
    # The request a real server accepted, byte for byte in meaning.
    assert sent[0].url.path == fixture["path"]
    assert json.loads(sent[0].content) == fixture["request"]
    return reply


# --- the contract, the same for both backends ------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
async def test_plain_answer(backend: str) -> None:
    reply = await replay(backend, "plain")
    assert reply.message.role == "assistant"
    assert reply.message.content.strip().lower().startswith("ready")
    assert not reply.message.tool_calls
    assert reply.usage.prompt_tokens > 0 and reply.usage.completion_tokens > 0
    assert reply.finish_reason == "stop"


@pytest.mark.parametrize("backend", BACKENDS)
async def test_tool_call(backend: str) -> None:
    reply = await replay(backend, "tool_call")
    (call,) = reply.message.tool_calls
    assert call.id
    assert call.name == "check_domain"
    assert call.arguments == {"domain": "paypa1-login.example"}


@pytest.mark.parametrize("backend", BACKENDS)
async def test_answer_after_a_tool_result(backend: str) -> None:
    reply = await replay(backend, "tool_result")
    assert not reply.message.tool_calls
    assert "3" in reply.message.content


@pytest.mark.parametrize("backend", BACKENDS)
async def test_schema_constrained_answer(backend: str) -> None:
    reply = await replay(backend, "schema")
    verdict = json.loads(reply.message.content)
    assert verdict["label"] in {"phishing", "legitimate"}
    assert 0 <= verdict["confidence"] <= 1


def test_every_recorded_fixture_is_tested() -> None:
    recorded = {(p.parent.name, p.stem) for p in FIXTURES.glob("*/*.json")}
    assert recorded == {(b, s) for b in BACKENDS for s in SCENARIOS}


async def test_ollama_asks_for_its_context_length_and_no_thinking() -> None:
    sent = load("ollama", "plain")["request"]
    assert sent["options"]["num_ctx"] == 16_384
    assert sent["think"] is False
    assert sent["stream"] is False


# --- failures --------------------------------------------------------------------------------


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def ok_body(backend: str) -> dict[str, Any]:
    response: dict[str, Any] = load(backend, "plain")["response"]
    return response


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("status", [429, 500, 503])
async def test_transient_status_is_retried_with_backoff(backend: str, status: int) -> None:
    statuses = iter([status, status, 200])
    sleeps = Sleeps()

    def handler(request: httpx.Request) -> httpx.Response:
        code = next(statuses)
        return httpx.Response(code, json=ok_body(backend) if code == 200 else None)

    client = build(backend, handler, max_retries=2, sleep=sleeps)
    reply = await client.chat([Message.user("hi")])
    assert reply.message.content
    assert len(sleeps.delays) == 2
    assert 0.25 <= sleeps.delays[0] <= 0.75
    assert 0.5 <= sleeps.delays[1] <= 1.5


@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.ConnectError("refused")],
)
async def test_network_failure_gives_up_after_the_retries(error: httpx.HTTPError) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise error

    client = build("ollama", handler, max_retries=2, sleep=Sleeps())
    with pytest.raises(LLMUnavailable, match=type(error).__name__):
        await client.chat([Message.user("hi")])
    assert attempts == 3


async def test_no_retry_when_retries_are_off() -> None:
    sleeps = Sleeps()
    client = build("ollama", lambda r: httpx.Response(503), max_retries=0, sleep=sleeps)
    with pytest.raises(LLMUnavailable, match="HTTP 503"):
        await client.chat([Message.user("hi")])
    assert sleeps.delays == []


@pytest.mark.parametrize(
    ("backend", "status", "body", "detail"),
    [
        ("ollama", 404, {"error": "model 'nope' not found"}, "model 'nope' not found"),
        ("openai", 401, {"error": {"message": "invalid api key"}}, "invalid api key"),
        ("openai", 400, {"unexpected": True}, "Bad Request"),
    ],
)
async def test_client_error_is_raised_at_once_with_the_backend_message(
    backend: str, status: int, body: dict[str, Any], detail: str
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status, json=body)

    client = build(backend, handler, sleep=Sleeps())
    with pytest.raises(LLMRequestError, match=detail):
        await client.chat([Message.user("hi")])
    assert attempts == 1


async def test_errors_never_quote_the_email() -> None:
    secret = "Dear customer, your account 4242 is locked"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="<html>upstream died</html>")

    client = build("openai", handler, max_retries=0)
    with pytest.raises(LLMUnavailable) as raised:
        await client.chat([Message.user(secret)])
    assert secret not in str(raised.value)
    assert "Internal Server Error" in str(raised.value)


async def test_long_backend_messages_are_cut() -> None:
    client = build("ollama", lambda r: httpx.Response(400, json={"error": "x" * 5000}))
    with pytest.raises(LLMRequestError) as raised:
        await client.chat([Message.user("hi")])
    assert len(str(raised.value)) < 300


@pytest.mark.parametrize(
    ("backend", "response"),
    [
        ("ollama", httpx.Response(200, text="not json")),
        ("ollama", httpx.Response(200, json=["a", "list"])),
        ("ollama", httpx.Response(200, json={"done": True})),
        ("ollama", httpx.Response(200, json={"message": {"tool_calls": [{"function": {}}]}})),
        ("openai", httpx.Response(200, json={"choices": []})),
        ("openai", httpx.Response(200, json={"choices": [{"message": None}]})),
    ],
)
async def test_unusable_answer(backend: str, response: httpx.Response) -> None:
    client = build(backend, lambda r: response)
    with pytest.raises(LLMResponseError):
        await client.chat([Message.user("hi")])


def openai_call(arguments: object) -> dict[str, Any]:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": arguments}}
    return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [call]}}]}


@pytest.mark.parametrize("arguments", ['{"domain": ', "[1, 2]", 42])
async def test_bad_tool_arguments(arguments: object) -> None:
    client = build("openai", lambda r: httpx.Response(200, json=openai_call(arguments)))
    with pytest.raises(LLMResponseError, match="arguments"):
        await client.chat([Message.user("hi")])


async def test_empty_tool_arguments_mean_no_arguments() -> None:
    client = build("openai", lambda r: httpx.Response(200, json=openai_call("")))
    reply = await client.chat([Message.user("hi")])
    assert reply.message.tool_calls[0].arguments == {}
    assert reply.message.content == ""
    assert reply.usage.prompt_tokens == 0


async def test_ollama_call_without_id_gets_one() -> None:
    body = {"message": {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}}
    client = build("ollama", lambda r: httpx.Response(200, json=body))
    reply = await client.chat([Message.user("hi")])
    assert reply.message.tool_calls == (ToolCall(id="call_0", name="f", arguments={}),)


async def test_api_key_goes_in_the_authorization_header_only() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body("openai"))

    with_key = build("openai", handler, api_key="sk-test")
    await with_key.chat([Message.user("hi")])
    without = build("openai", handler)
    await without.chat([Message.user("hi")])
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    assert b"sk-test" not in seen[0].content
    assert "authorization" not in seen[1].headers


# --- configuration ---------------------------------------------------------------------------


def test_make_client_defaults_to_local_ollama() -> None:
    client = make_client(Settings())
    assert isinstance(client, Ollama)
    assert client.model == "qwen3:8b"


async def test_make_client_honours_the_settings() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body("openai"))

    settings = Settings(
        llm_backend="openai",
        llm_base_url="https://api.example.org/v1/",
        llm_model="small",
        llm_api_key=SecretStr("sk-x"),
    )
    client = make_client(settings, transport=httpx.MockTransport(handler))
    assert isinstance(client, OpenAICompatible)
    await client.chat([Message.user("hi")])
    await client.aclose()
    assert str(seen[0].url) == "https://api.example.org/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-x"
    assert json.loads(seen[0].content)["model"] == "small"


async def test_openai_backend_without_url_uses_the_local_compatibility_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body("openai"))

    client = make_client(Settings(llm_backend="openai"), transport=httpx.MockTransport(handler))
    await client.chat([Message.user("hi")])
    assert str(seen[0].url) == "http://127.0.0.1:11434/v1/chat/completions"


# --- the fake --------------------------------------------------------------------------------


async def test_scripted_llm_plays_its_script_and_records_calls() -> None:
    tool = ToolSpec(name="f", description="d", parameters={"type": "object"})
    client: LLMClient = (fake := ScriptedLLM.saying("one", "two"))
    first = await client.chat([Message.user("a")], tools=[tool])
    second = await client.chat([Message.user("b")], schema={"type": "object"})
    assert (first.message.content, second.message.content) == ("one", "two")
    assert fake.calls[0].tools == (tool,)
    assert fake.calls[1].schema == {"type": "object"}
    with pytest.raises(AssertionError, match="exhausted"):
        await client.chat([Message.user("c")])
    await client.aclose()
    assert fake.closed


async def test_scripted_llm_raises_scripted_errors() -> None:
    client = ScriptedLLM([LLMUnavailable("down")])
    with pytest.raises(LLMUnavailable):
        await client.chat([Message.user("a")])


def test_tool_result_points_back_to_its_call() -> None:
    call = ToolCall(id="c7", name="check_domain", arguments={"domain": "x.example"})
    result = Message.tool_result(call, "{}")
    assert (result.role, result.tool_call_id, result.name) == ("tool", "c7", "check_domain")
    assert Message.system("s").role == "system"
