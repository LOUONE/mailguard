"""What the agent can do: small, typed, read-only functions over the email under investigation.

A `Tool` is a name, a description for the model, a pydantic model of its arguments (whose JSON
schema is what the model sees) and an async function. Tools read the `Case`, never change it,
and never raise into the loop: bad arguments, an unknown name or a failure come back as an error
result the model can read and correct.

Everything a tool returns about the email (filenames, URLs, header values) was written by the
sender. It is data, sized and truncated here; #46 decides how the loop marks it as untrusted.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from mailguard.agent.llm import ToolCall, ToolSpec
from mailguard.parsing import Email

log = structlog.get_logger("mailguard.agent")

# A tool result is read by the model: long ones cost tokens and bury the evidence.
MAX_RESULT_CHARS: Final = 4000


class Arguments(BaseModel):
    """Base of every tool's arguments. Unknown fields are refused, not ignored."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NoArguments(Arguments):
    pass


class Output(BaseModel):
    model_config = ConfigDict(frozen=True)


@dataclass(frozen=True, slots=True)
class Reason:
    feature: str
    contribution: float  # log-odds, positive towards phishing


@dataclass(frozen=True, slots=True)
class Opinion:
    """What the classifier said about the email, before the agent was called."""

    score: float  # calibrated, at the training base rate
    probability: float  # at the deployment base rate
    is_phishing: bool
    threshold: float
    reasons: Sequence[Reason] = ()


@dataclass(frozen=True, slots=True)
class Case:
    """The email under investigation and what the classifier made of it."""

    email: Email
    opinion: Opinion
    # The receiving server's id in Authentication-Results. Only its header can be trusted.
    trusted_authserv_id: str | None = None


@dataclass(frozen=True, slots=True)
class Tool[A: Arguments, O: Output]:
    name: str
    description: str
    arguments: type[A]
    run: Callable[[Case, A], Awaitable[O]]

    @property
    def spec(self) -> ToolSpec:
        schema = self.arguments.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(name=self.name, description=self.description, parameters=schema)


@dataclass(frozen=True, slots=True)
class Result:
    """One tool call's answer, as the loop sends it back to the model."""

    call: ToolCall
    content: str  # JSON
    ok: bool


@dataclass(frozen=True)
class Toolbox:
    tools: Mapping[str, Tool[Any, Any]] = field(default_factory=dict)

    @classmethod
    def of(cls, tools: Iterable[Tool[Any, Any]]) -> Toolbox:
        by_name: dict[str, Tool[Any, Any]] = {}
        for tool in tools:
            if tool.name in by_name:
                raise ValueError(f"two tools named {tool.name}")
            by_name[tool.name] = tool
        return cls(by_name)

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools.values()]

    async def call(self, case: Case, call: ToolCall) -> Result:
        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(self.tools)
            return _error(call, f"unknown tool {call.name!r}; tools: {known}")
        try:
            arguments = tool.arguments.model_validate(call.arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '-'}: {e['msg']}" for e in exc.errors()
            )
            return _error(call, f"invalid arguments: {problems}")
        try:
            output = await tool.run(case, arguments)
        except Exception as exc:  # a tool bug must not end the investigation
            log.warning("tool_failed", tool=call.name, error_type=type(exc).__name__)
            return _error(call, f"{call.name} failed ({type(exc).__name__})")
        return Result(call, _bounded(output.model_dump_json()), ok=True)


def _error(call: ToolCall, message: str) -> Result:
    return Result(call, json.dumps({"error": message[:500]}), ok=False)


def _bounded(content: str) -> str:
    if len(content) <= MAX_RESULT_CHARS:
        return content
    return json.dumps({"truncated": True, "partial": content[:MAX_RESULT_CHARS]})


def clip(text: str | None, limit: int = 200) -> str | None:
    """Sender-written strings are cut before they reach the model."""
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "…"
