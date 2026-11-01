"""What the agent needs from a language model, whatever serves it.

The agent only ever talks to an `LLMClient`: one chat turn in, one assistant message out, with
tool calling and, optionally, an output held to a JSON schema. The backends (`backends.py`) turn
that into the wire format of Ollama or of an OpenAI-compatible API; tests use a scripted fake.

The client is async. A model call is seconds of waiting on a socket, not of computing, and unlike
the classifier's worker thread an awaited request can really be cancelled when the API's time
budget runs out.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
JsonSchema = dict[str, Any]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolSpec(_Frozen):
    """A tool the model may call: its name, what it is for, and a JSON schema of its arguments."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    description: str
    parameters: JsonSchema


class ToolCall(_Frozen):
    """One call the model asked for. `id` ties the tool's answer back to it."""

    id: str
    name: str
    arguments: dict[str, Any]


class Message(_Frozen):
    role: Role
    content: str = ""
    # Assistant messages only: the calls the model asked for.
    tool_calls: tuple[ToolCall, ...] = ()
    # Tool messages only: which call this is the result of, and of which tool.
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def tool_result(cls, call: ToolCall, content: str) -> Message:
        return cls(role="tool", content=content, tool_call_id=call.id, name=call.name)


class Usage(_Frozen):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Reply(_Frozen):
    """The assistant's turn, with what it cost."""

    message: Message
    usage: Usage = Usage()
    # Why generation stopped, as the backend says it ("stop", "length", "tool_calls"...).
    finish_reason: str | None = None


class LLMError(Exception):
    """Base of every failure a client raises: callers catch this one and fall back."""


class LLMUnavailable(LLMError):
    """The backend could not be reached or kept failing: timeout, refused connection, 5xx, 429."""


class LLMRequestError(LLMError):
    """The backend refused the request (4xx): a wrong model name or key, an unsupported field."""


class LLMResponseError(LLMError):
    """The backend answered, but not with something we can use: bad JSON, missing fields."""


class LLMClient(Protocol):
    """One chat turn. Implementations raise only `LLMError` subclasses."""

    @property
    def model(self) -> str:
        """The model's name as the backend knows it, for logs and reports."""
        ...

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[ToolSpec] = (),
        schema: JsonSchema | None = None,
    ) -> Reply:
        """Send the conversation, get the next assistant message.

        `tools` are offered to the model; `schema`, when given, asks for a final answer that is a
        JSON document following it. Whether the content really follows the schema is for the
        caller to validate: a backend's constrained decoding is a help, not a guarantee.
        """
        ...

    async def aclose(self) -> None:
        """Release the connection pool."""
        ...
