# 0009. Reach LLMs through our own small protocol over plain `httpx`, with a native Ollama backend

- Status: Accepted
- Date: 2026-11-01
- Issue: #42

## Context

The Phase 3 agent needs chat with tool calling and, for its final verdict, an answer held to a
JSON schema. It must run on a local open model by default (no email leaves the machine, which is
the point for a privacy-focused mail provider), and on a hosted OpenAI-compatible API (Mistral,
Infomaniak) when one is configured. Tests must run in CI without a model.

One measurement shaped the decision. With Ollama 0.40.13 and `qwen3:8b` on an RTX 5070, a
system prompt holding a secret word, followed by about 9,000 tokens of user text:

| Request | Prompt tokens read | Answer |
|---|---|---|
| `/api/chat`, default options | 2,050 | "lazy" (wrong) |
| `/api/chat`, `num_ctx: 16384` | 9,048 | "PINEAPPLE" (right) |
| `/v1/chat/completions` (OpenAI-compatible) | 2,050 | "FOX" (wrong), after 1,913 tokens of reasoning |

Ollama's default context here is about 2,000 tokens, and it truncates the **start** of a longer
prompt without an error. The start is the system prompt. For an agent that reads
attacker-controlled email, a long email would erase the instructions that tell the model to
treat the email as data (#46). The OpenAI-compatible endpoint has no per-request context length.

## Options

### A. The `openai` SDK for every backend

Ollama, Mistral and Infomaniak all speak its protocol. Typed responses, retries built in. But
through Ollama it hits the truncation above, unless every deployment sets
`OLLAMA_CONTEXT_LENGTH` on the server, a setting the code cannot check. It adds a dependency
whose types would leak into the agent code.

### B. `ollama` SDK plus `openai` SDK

Each backend gets its native options. Two SDKs, two sets of types and retry behaviours to hide
behind an interface anyway.

### C. A framework (LangChain, LiteLLM)

Many providers for free, and agent helpers. A large dependency tree for two HTTP endpoints, its
own abstractions to learn and debug, and the tool-calling loop (#45) would follow the framework's
design rather than ours.

### D. Our own `LLMClient` protocol, two backends over `httpx`

About 450 lines with docstrings. `httpx` is already locked (FastAPI's test client). Every byte
sent is visible and tested against recorded responses. Retries, timeouts and errors are ours to
write and test.

## Decision

D. `mailguard.agent.llm` defines the protocol and the message types; `mailguard.agent.backends`
has `Ollama` (native `/api/chat`, sends `num_ctx`, 16,384 by default, and `think: false`) and
`OpenAICompatible` (`/chat/completions`, key from `.env` only). Ollama is the default backend.
Retries cover timeouts, refused connections, 429 and 5xx, with exponential backoff and jitter; a
4xx is raised at once. No error message quotes the request.

Contract tests replay responses recorded from a real server (`scripts/record_llm_fixtures.py`)
and check the client still sends the request that server accepted.

## Consequences

- Swapping the local model is a setting; adding a provider that follows the OpenAI protocol is a
  base URL and a key.
- A provider with another protocol (Anthropic's, Google's) needs a third backend.
- The recorded fixtures pin today's wire formats. When Ollama or a provider changes theirs, the
  fixtures are recorded again and the diff shows what changed.
- The OpenAI-compatible path through Ollama still truncates at the server's default: it is for
  tests and for hosted APIs, not for production against Ollama.
- We own the retry logic. Revisit if streaming, or more than two or three providers, become
  necessary.
