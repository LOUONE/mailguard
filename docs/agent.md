# Investigation agent

LightGBM decides the obvious cases in milliseconds. When its calibrated score falls in the grey
zone, an LLM agent investigates the email with tools and returns a verdict a user can read. This
page describes the pieces as they land in Phase 3.

| Piece | Code | Issue |
|---|---|---|
| LLM client: Ollama (default, local) or any OpenAI-compatible API | `agent/llm.py`, `agent/backends.py` | #42, [ADR 0009](adr/0009-llm-client-over-plain-http.md) |
| Offline tools | `agent/tools/offline.py` | #43 |
| Sandboxed network tools | `agent/tools/network.py` | #44, [ADR 0010](adr/0010-network-evidence.md) |
| Tool-calling loop and verdict | `agent/investigator.py`, `scripts/investigate.py` | #45 |
| Prompt-injection defence | `agent/injection.py`, [report](../reports/injection.md) | #46 |
| `/v1/investigate` and grey-zone routing | | #47 |
| Hard evaluation set | `agent/hard_set.py`, [hard_set.md](hard_set.md) | #48 |

## The investigation loop

```
uv run python scripts/investigate.py docs/demo/phishing.eml
```

`Investigator.investigate(case)` runs four stages and always returns an `Investigation`:

1. **Investigate.** The model gets a system prompt (what phishing is, that the email is
   untrusted data between `<<<EMAIL` and `EMAIL>>>`), the email and the tools. It calls tools
   for up to 6 turns, 4 calls a turn, and stops when it answers without asking for one.
2. **Conclude.** One more turn, without tools, asks for a `Verdict` (label, confidence,
   evidence, plain explanation) held to its JSON schema. Tools and a forced format are kept in
   separate turns: with both in one request, constrained decoding tends to stop small models
   from calling tools.
3. **Check.** The verdict is validated against the schema. Evidence citing a tool the model
   never called is dropped and counted (`discarded_evidence`). An invalid verdict, or one whose
   evidence is all invented, gets one retry with the reason.
4. **Fall back.** An LLM failure or a second invalid verdict returns the classifier's verdict in
   the same format, with `source: classifier` and the reason.

The classifier's score is not in the prompt: the model can ask for it with
`get_classifier_explanation`, but is not anchored on it. Each investigation logs one line with
its source, label, steps, tool names, tokens and duration, and nothing the sender or the model
wrote.

A run with `qwen3:8b` on an RTX 5070, on the two demo emails:

| Email | Verdict | Turns | Tool calls | Time |
|---|---|---|---|---|
| `phishing.eml` (PayPal lure, link to an IP) | phishing, 0.95 | 2 | auth headers, 2 URLs, domain age | 14 s |
| `colleague.eml` (work follow-up) | legitimate, 0.95 | 1 | none | 3 s |

On the first run the model read `domain_age`'s `no_rdap_service` for `.example` as "fake
domain". The tool description now says that status means unknown, not suspicious, and the next
run said "reserved for testing". Tool descriptions are part of the prompt, and #49 evaluates
them like the rest.

## Prompt-injection defence

Every email is attacker-written text read by a language model. The tools only read, so the
worst an injection can buy is a wrong verdict, usually "legitimate" on a phishing email. Three
layers (`agent/injection.py`):

1. **An unguessable fence.** The email sits between `<<<EMAIL <id>` and `EMAIL <id>>>`, with a
   random id drawn for each investigation, and any `<<<` or `>>>` in the content is broken up.
2. **A deterministic scan** for text written for an AI: override phrasings (English and
   French), "you are an AI", chat-template tokens and named system prompts, requests to label
   the message safe, fake tool calls, fake verdict JSON, marker spoofing. It reads the subject,
   display name, body, raw HTML (comments and CSS-hidden text included), attachment names and
   URLs. A hit is shown to the model as a note from the system, outside the fence: legitimate
   mail does not address an AI. Bare `System:` lines are not flagged; IT notices use them.
3. **A floor.** If the scan found something and the agent still says "legitimate", the
   classifier's verdict is returned instead. An injection cannot buy a better verdict than the
   classifier alone gives.

### Measured resistance

`scripts/injection_eval.py` runs 13 trapped emails and 3 legitimate controls
(`tests/fixtures/injection/`, all made up) through the served classifier and the agent with
every tool, with the fence only and with the full defence. With `qwen3:8b`
([full report](../reports/injection.md)):

| | Fence only | Full defence |
|---|---|---|
| Traps where the agent itself said phishing | 11/13 | 13/13 |
| Traps finally labelled phishing | 11/13 | 13/13 |
| Controls finally labelled legitimate | 2/3 | 2/3 |

- **Without the scan, the model fell for exactly the two attacks that imitate the
  conversation's structure**: a fake end-of-email marker followed by a fake conclusion, and
  chat-template tokens opening a fake system turn. With the note, it resisted both on its own;
  the floor was never needed.
- **The scan found 11 of 13 traps.** The two it missed carry no trigger words (a claimed prior
  approval, a request to check 30 mirror links first); the model resisted both anyway, and the
  step and call limits bound the second.
- **One control stays wrong in both conditions**: a newsletter that quotes "ignore all
  previous instructions" while explaining prompt injection. The classifier also flags it
  (0.37 against a 0.29 threshold). That is the known cost of the scan: an email about
  injections looks like one.
- **One prompt sentence was added after the first run**, which also labelled a planned-
  maintenance notice phishing because it had no `Authentication-Results` header and a reserved
  domain: "Missing information is not evidence". Controls went from 1/3 to 2/3, traps did not
  move. The agent now clears that notice although the classifier flags it (0.85). Three
  controls are too few to tune on any further; the agent's false-positive rate is measured on
  the hand-labelled hard set (#49), which no prompt was tuned on.

## Tools

A tool is a name, a description, a pydantic model of its arguments (its JSON schema is what the
model sees) and an async function over the `Case`: the parsed email and the classifier's opinion.
Tools are read-only and never raise into the loop. An unknown tool, bad arguments or a failure
come back to the model as an error it can correct, without the exception's text. Results are
capped at 4,000 characters, and every sender-written string in them is clipped.

| Tool | Arguments | What it returns |
|---|---|---|
| `check_auth_headers` | none | SPF, DKIM and DMARC from `Authentication-Results`; From, Reply-To and Return-Path domains and whether they differ; a display name naming another brand |
| `analyze_url` | `url` | host, registrable domain, IP or punycode host and its Unicode form, `user@host` trick, port, shortener, suspicious TLD, login keywords, brand signals, link texts naming another domain, and whether the URL is in the email at all |
| `inspect_attachments` | none | per attachment: name, declared and sniffed type, size, risky, archive or double extension, name/content mismatch |
| `get_classifier_explanation` | none | score, probability, label, threshold and the top TreeSHAP contributions |

### Why `Authentication-Results` is trusted only from one server

Any sender can write an `Authentication-Results` header claiming `dmarc=pass`. Only the one added
by the receiving server is evidence, and RFC 8601 identifies it by its first token, the
`authserv-id`. With `MAILGUARD_TRUSTED_AUTHSERV_ID` set, the tool reads only that server's
header. Without it, the tool reads the topmost header and returns `trusted: false`, so the
agent can weigh it accordingly. The model never used this signal: on the corpora the header
exists only in recent phishing, so it would have learnt the era (docs/features.md).

### Brand signals

The registrable domain (via the Public Suffix List: `paypal.co.uk`, not `co.uk`) is compared
with 29 brands most often impersonated (`features/lists.py`).

| Signal | Example | Rule |
|---|---|---|
| `brand_domain` | `www.paypal.com` | the domain label is the brand |
| `typosquat` | `paypa1.com`, `rnicrosoft.com`, `micorsoft.com` | equal after mapping lookalike characters (`1`→`l`, `rn`→`m`), or within 1 edit (brands of 5 to 7 letters) or 2 edits (8 or more), adjacent swaps counting as one |
| `homoglyph` | `xn--ypl-5cdc6g.com` ("paypal" with Cyrillic er and a) | non-ASCII label equal to the brand after mapping Cyrillic and Greek lookalikes |
| `combosquat` | `secure-paypal-login.example` | the brand is a word of the label, or inside it for brands of 6 letters or more |
| `brand_outside_domain` | `paypal.com.evil.example/webscr` | the brand appears in subdomains, path or query only, where it proves nothing |

Brands of 4 letters or fewer (`irs`, `dhl`, `usps`) match only exactly: one edit from `dhl` is
`dhs.gov`. A signal is evidence for the agent to weigh, not a verdict: `brand_domain` does not
check the brand's official domain list, and a `typosquat` can be an unrelated word.
The character map is a small part of Unicode's confusables; Phase 4 (#53) brings the full
UTS #39 skeletons.

### Network tools

| Tool | Arguments | What it returns |
|---|---|---|
| `domain_age` | `domain` | registration date, age in days, expiry, registrar and registry statuses, from the registry's RDAP server; `not_in_email`, `not_found`, `no_rdap_service` or `unavailable` otherwise |
| `lookup_reputation` | `url` | whether the exact URL is in the PhishTank or OpenPhish snapshot (with the brand it targets), how many listed URLs share its host, whether that host is a shared platform, and how old the snapshot is |

Both are built from their dependencies (`network_tools(rdap, reputation)`), so tests pass mocked
transports and an index built from a temporary folder. The fence, from ADR 0010:

- RDAP requests go through `AllowlistTransport`: HTTPS to the registry servers of the bundled
  IANA bootstrap only, refused before any connection otherwise, redirects not followed.
- `domain_age` looks up only domains that appear in the email, so a message cannot point it
  anywhere else.
- `lookup_reputation` never goes online. `make reputation` downloads the feeds into
  `data/reputation/` with their SHA-256 and download time; a file changed since then is
  refused at load, and snapshots older than 7 days are reported stale.
- The tests run with outbound sockets refused.

A real run, on a Netflix lure taken from the PhishTank feed: listed, and its `.com` registered 5
days earlier at a low-cost registrar. Two tool calls, and the lure itself was never contacted.

Refreshing the RDAP bootstrap is a reviewed change, like the Public Suffix List:
`curl -o src/mailguard/agent/tools/data/rdap_dns.json https://data.iana.org/rdap/dns.json`.

### What the tools never do

- Fetch, resolve or click a URL, or open an attachment. Attachments are metadata from the
  parser, which kept only a hash and the sniffed type (ADR 0002).
- Trust what they return. Filenames, URLs and display names are attacker-written text that ends
  up in the model's context; #46 measures how the agent resists instructions hidden there.
