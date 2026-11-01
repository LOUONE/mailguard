"""The two ways the agent reaches a model: Ollama's native API, or any OpenAI-compatible one.

- `Ollama` is the default: an open model on the same machine, so no email leaves it. It speaks
  `/api/chat`, the only Ollama endpoint that takes a context length per request. Ollama's own
  default is a few thousand tokens and it truncates the start of a longer prompt without an
  error: the system prompt goes first, so a long email would silently erase the instructions.
- `OpenAICompatible` speaks `/chat/completions`: a hosted API (Mistral, Infomaniak...) when the
  key is in `.env`, or Ollama's own compatibility endpoint for local tests.

Both are plain `httpx` calls (ADR 0009). Failures are retried when retrying can help (timeouts,
refused connections, 429 and 5xx) with exponential backoff and jitter; a 4xx is the caller's
fault and raised at once. Errors never quote the request: it holds the email.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final

import httpx

from mailguard.agent.llm import (
    JsonSchema,
    LLMClient,
    LLMError,
    LLMRequestError,
    LLMResponseError,
    LLMUnavailable,
    Message,
    Reply,
    ToolCall,
    ToolSpec,
    Usage,
)
from mailguard.config import Settings

Sleep = Callable[[float], Awaitable[None]]
# Long enough for a slow local model to answer, short enough to notice a dead connection.
CONNECT_TIMEOUT_S: Final = 5.0
BACKOFF_S: Final = 0.5
# Backend error messages are logged: keep them short.
MAX_DETAIL: Final = 200


class _HttpBackend:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_s: float,
        max_retries: int,
        headers: dict[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=min(CONNECT_TIMEOUT_S, timeout_s)),
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with retries; the parsed JSON object, or an `LLMError`."""
        error: LLMError = LLMUnavailable("no attempt made")
        for attempt in range(self._max_retries + 1):
            if attempt:
                await self._sleep(BACKOFF_S * 2 ** (attempt - 1) * random.uniform(0.5, 1.5))
            try:
                response = await self._http.post(path, json=payload)
            except httpx.TimeoutException as exc:
                error = LLMUnavailable(f"{type(exc).__name__} from {self._http.base_url}")
                continue
            except httpx.TransportError as exc:
                error = LLMUnavailable(f"{type(exc).__name__} from {self._http.base_url}")
                continue
            if response.status_code == 429 or response.status_code >= 500:
                error = LLMUnavailable(f"HTTP {response.status_code}: {_detail(response)}")
                continue
            if response.is_error:
                raise LLMRequestError(f"HTTP {response.status_code}: {_detail(response)}")
            try:
                body = response.json()
            except ValueError:
                raise LLMResponseError("response is not JSON") from None
            if not isinstance(body, dict):
                raise LLMResponseError("response is not a JSON object")
            return body
        raise error


def _detail(response: httpx.Response) -> str:
    """The backend's own error message: `{"error": "..."}` or `{"error": {"message": "..."}}`."""
    try:
        error = response.json().get("error")
    except ValueError, AttributeError:
        return response.reason_phrase
    if isinstance(error, dict):
        error = error.get("message")
    return str(error)[:MAX_DETAIL] if error else response.reason_phrase


def _arguments(raw: object) -> dict[str, Any]:
    """Tool arguments arrive as an object (Ollama) or as a JSON string of one (OpenAI)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raise LLMResponseError("tool call arguments are not valid JSON") from None
    if not isinstance(raw, dict):
        raise LLMResponseError("tool call arguments are not a JSON object")
    return raw


def _tool(spec: ToolSpec) -> dict[str, Any]:
    return {"type": "function", "function": spec.model_dump()}


class Ollama(_HttpBackend):
    """Ollama's native chat API, `POST /api/chat`, without streaming."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 60.0,
        max_retries: int = 2,
        context_tokens: int = 16_384,
        temperature: float = 0.0,
        think: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            transport=transport,
            sleep=sleep,
        )
        self._options = {"num_ctx": context_tokens, "temperature": temperature}
        self._think = think

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        schema: JsonSchema | None = None,
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._message(m) for m in messages],
            "stream": False,
            "think": self._think,
            "options": self._options,
        }
        if tools:
            payload["tools"] = [_tool(t) for t in tools]
        if schema is not None:
            payload["format"] = schema
        return self._reply(await self._post("api/chat", payload))

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "function": {"name": c.name, "arguments": c.arguments}}
                for c in message.tool_calls
            ]
        if message.role == "tool":
            out["tool_name"] = message.name
        return out

    @staticmethod
    def _reply(body: dict[str, Any]) -> Reply:
        try:
            raw = body["message"]
            calls = tuple(
                ToolCall(
                    # Ollama gives ids since 0.12; older servers did not, so make one up.
                    id=str(call.get("id") or f"call_{i}"),
                    name=call["function"]["name"],
                    arguments=_arguments(call["function"].get("arguments", {})),
                )
                for i, call in enumerate(raw.get("tool_calls") or ())
            )
            message = Message(role="assistant", content=raw.get("content") or "", tool_calls=calls)
        except (KeyError, TypeError, AttributeError) as exc:
            raise LLMResponseError(f"unexpected Ollama response: {exc!r}") from None
        usage = Usage(
            prompt_tokens=body.get("prompt_eval_count") or 0,
            completion_tokens=body.get("eval_count") or 0,
        )
        return Reply(message=message, usage=usage, finish_reason=body.get("done_reason"))


class OpenAICompatible(_HttpBackend):
    """`POST /chat/completions` as OpenAI defined it; Mistral, Infomaniak and Ollama follow it."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            transport=transport,
            sleep=sleep,
        )
        self._temperature = temperature

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        schema: JsonSchema | None = None,
    ) -> Reply:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._message(m) for m in messages],
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = [_tool(t) for t in tools]
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }
        return self._reply(await self._post("chat/completions", payload))

    @staticmethod
    def _message(message: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            out["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in message.tool_calls
            ]
        if message.role == "tool":
            out["tool_call_id"] = message.tool_call_id
        return out

    @staticmethod
    def _reply(body: dict[str, Any]) -> Reply:
        try:
            choice = body["choices"][0]
            raw = choice["message"]
            calls = tuple(
                ToolCall(
                    id=str(call["id"]),
                    name=call["function"]["name"],
                    arguments=_arguments(call["function"].get("arguments", "")),
                )
                for call in raw.get("tool_calls") or ()
            )
            message = Message(role="assistant", content=raw.get("content") or "", tool_calls=calls)
            usage = body.get("usage") or {}
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMResponseError(f"unexpected chat completion: {exc!r}") from None
        return Reply(
            message=message,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens") or 0,
                completion_tokens=usage.get("completion_tokens") or 0,
            ),
            finish_reason=choice.get("finish_reason"),
        )


def make_client(settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> LLMClient:
    """The client the settings ask for. `transport` replaces the network in tests."""
    if settings.llm_backend == "ollama":
        return Ollama(
            model=settings.llm_model,
            base_url=settings.llm_base_url or "http://127.0.0.1:11434",
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
            context_tokens=settings.llm_context_tokens,
            temperature=settings.llm_temperature,
            think=settings.llm_think,
            transport=transport,
        )
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return OpenAICompatible(
        model=settings.llm_model,
        # Without a URL, Ollama's compatibility endpoint: the same local model, the OpenAI way.
        base_url=settings.llm_base_url or "http://127.0.0.1:11434/v1",
        api_key=key,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        temperature=settings.llm_temperature,
        transport=transport,
    )
