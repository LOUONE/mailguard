"""Record what a real model server answers, for the LLM client's contract tests.

    uv run python scripts/record_llm_fixtures.py                 # local Ollama, qwen3:8b
    uv run python scripts/record_llm_fixtures.py --model llama3.1:8b

Each scenario is sent once through each backend (Ollama's native API and its OpenAI-compatible
endpoint), and the exchange is written to `tests/fixtures/llm/<backend>/<scenario>.json`: the
client's input, the exact request it sent, and the server's answer. The tests replay the answer
and check that the client still sends that same request, the one a real server accepted.

The prompts are made up and short: no email, real or not, goes into a fixture.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from mailguard.agent.backends import Ollama, OpenAICompatible
from mailguard.agent.llm import JsonSchema, LLMClient, LLMError, Message, ToolSpec

OUT = Path("tests/fixtures/llm")
SYSTEM = Message.system("You are a terse assistant for a mail security team.")
CHECK_DOMAIN = ToolSpec(
    name="check_domain",
    description="Look up how long ago a domain was registered.",
    parameters={
        "type": "object",
        "properties": {"domain": {"type": "string", "description": "Domain name, no scheme."}},
        "required": ["domain"],
    },
)
ASK_TOOL = Message.user("Use the tool to check the domain paypa1-login.example, then stop.")
VERDICT: JsonSchema = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["phishing", "legitimate"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["label", "confidence"],
}


class Recorder(httpx.AsyncBaseTransport):
    """A real transport that keeps a copy of every exchange."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.exchanges: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        content = await response.aread()
        self.exchanges.append(
            {
                "path": request.url.path,
                "request": json.loads(request.content),
                "status": response.status_code,
                "response": json.loads(content),
            }
        )
        return httpx.Response(response.status_code, headers=response.headers, content=content)

    async def aclose(self) -> None:
        await self._inner.aclose()


async def record(
    name: str,
    client: LLMClient,
    recorder: Recorder,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] = (),
    schema: JsonSchema | None = None,
) -> Message:
    reply = await client.chat(messages, tools=tools, schema=schema)
    exchange = recorder.exchanges[-1]
    fixture = {
        "input": {
            "messages": [m.model_dump(mode="json") for m in messages],
            "tools": [t.model_dump(mode="json") for t in tools],
            "schema": schema,
        },
        **exchange,
    }
    path = OUT / backend_name(client) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"{path}: {reply.usage.prompt_tokens}+{reply.usage.completion_tokens} tokens")
    return reply.message


def backend_name(client: LLMClient) -> str:
    return "ollama" if isinstance(client, Ollama) else "openai"


async def scenarios(client: LLMClient, recorder: Recorder) -> None:
    await record("plain", client, recorder, [SYSTEM, Message.user("Reply with one word: ready.")])
    asked = await record("tool_call", client, recorder, [SYSTEM, ASK_TOOL], tools=[CHECK_DOMAIN])
    if not asked.tool_calls:
        raise LLMError("the model did not call the tool: try another model")
    result = Message.tool_result(asked.tool_calls[0], json.dumps({"age_days": 3}))
    await record(
        "tool_result",
        client,
        recorder,
        [SYSTEM, ASK_TOOL, asked, result],
        tools=[CHECK_DOMAIN],
    )
    await record(
        "schema",
        client,
        recorder,
        [SYSTEM, Message.user("A 3-day-old domain asks for a password. Give your verdict.")],
        schema=VERDICT,
    )


async def run(model: str, base_url: str) -> None:
    for make in (
        lambda t: Ollama(model=model, base_url=base_url, transport=t, timeout_s=300),
        lambda t: OpenAICompatible(
            model=model, base_url=f"{base_url}/v1", transport=t, timeout_s=300
        ),
    ):
        recorder = Recorder()
        client = make(recorder)
        try:
            await scenarios(client, recorder)
        finally:
            await client.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="record_llm_fixtures")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args.model, args.base_url))
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
