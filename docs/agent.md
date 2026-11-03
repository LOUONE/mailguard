# Investigation agent

LightGBM decides the obvious cases in milliseconds. When its calibrated score falls in the grey
zone, an LLM agent investigates the email with tools and returns a verdict a user can read. This
page describes the pieces as they land in Phase 3.

| Piece | Code | Issue |
|---|---|---|
| LLM client: Ollama (default, local) or any OpenAI-compatible API | `agent/llm.py`, `agent/backends.py` | #42, [ADR 0009](adr/0009-llm-client-over-plain-http.md) |
| Offline tools | `agent/tools/offline.py` | #43 |
| Sandboxed network tools | `agent/tools/network.py` | #44, [ADR 0010](adr/0010-network-evidence.md) |
| Tool-calling loop and verdict | | #45 |
| Prompt-injection defence | | #46 |
| `/v1/investigate` and grey-zone routing | | #47 |
| Hard evaluation set | `agent/hard_set.py`, [hard_set.md](hard_set.md) | #48 |

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
