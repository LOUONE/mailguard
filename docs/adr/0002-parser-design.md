# 0002. Parser design: lenient, never raises, metadata-only attachments

- Status: Accepted (retroactive: the code shipped in PR #18 and #19)
- Date: 2026-10-27
- Issues: #11, #12

## Context

Every downstream step, from features to the API, starts from `parse_email(raw) -> Email`. The input
is hostile by nature: phishing is written to fool parsers, and the corpora contain broken MIME,
unknown charsets, raw 8-bit headers and dates in year 102 or 2038. One message in
`phishing3.mbox` makes the standard library raise when it is re-serialised. In production, a
parser exception is either a 500 for the caller or a message that skips the filter.

The parser also holds email content in memory, which matters for privacy: Phase 2 logs must never
contain any, and attachments are the largest and riskiest part of a message.

## Options

### A. `email.policy.default` and let exceptions propagate

The modern API, with typed header objects. But it parses headers eagerly and raises on hostile
input, and every caller would need its own error handling.

### B. A third-party parser (`mail-parser`, `flanker`)

More features, but another dependency to audit on the attack surface, and less control over what
happens on malformed input.

### C. `email.policy.compat32`, never raise, record defects

The most lenient standard-library policy: malformed headers come back as raw strings and problems
are recorded as defects instead of being raised. The model is built by hand from the parts.

## Decision

**Option C**, with these rules, all in `src/mailguard/parsing/`:

1. **`compat32` policy.** Headers are decoded one by one (RFC 2047), tolerating unknown charsets
   and raw bytes. Bodies are decoded with the declared charset, then UTF-8, then latin-1, which
   cannot fail.
2. **Never raise.** Problems become entries in `Email.defects` (class names only, no content). A
   last-resort guard turns any unexpected exception into a partial `Email` with a
   `ParserFailure:<type>` defect. Property tests assert that this guard *never* fires, so it is a
   safety net, not a way to hide bugs.
3. **Immutable, validated output.** `Email` is a frozen pydantic model in which every field has a
   safe default, so an empty or broken message still yields a valid object that serialises to
   JSON.
4. **Attachments are metadata only**: filename, declared content type, size and SHA-256. The
   payload is hashed and dropped.
5. **HTML is read with the standard library** (`html.parser`): link targets and visible text,
   skipping `script`, `style` and `head`. No external HTML engine, nothing executed, nothing
   fetched. `Email.body_text` is the plain-text body, or the visible HTML text when there is none.
6. **Untrusted values stay untrusted.** `Date` is kept but flagged as untrusted. Addresses are
   lowercased and set to `None` when they have no domain.

## Consequences

- Zero parser exceptions on the 14,084 corpus messages, and two bugs found by `hypothesis` before
  they reached a model (PR #19).
- Callers never handle parser errors. The defect count is available as a feature (#23).
- Accepted gaps, to revisit in Phase 4 (#53): text hidden with CSS (`display:none`, white on
  white) is still counted as visible; only the first `From` address is used; the latin-1 fallback
  can produce mojibake on mislabelled charsets.
- An attachment's type can be checked against its extension only from metadata kept at parse
  time. If a feature needs more (magic bytes), it is computed in the parser and stored as metadata,
  never by keeping the payload.
