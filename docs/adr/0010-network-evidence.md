# 0010. Network evidence: RDAP for domain age, local feed snapshots for reputation, fenced egress

- Status: Accepted
- Date: 2026-11-03
- Issue: #44

## Context

Two facts about a link weigh heavily and are not in the email: how old its domain is, and
whether it is already a known phishing URL. Getting them means talking to the outside world
from a process that reads attacker-controlled text, so the rules come first: the agent never
fetches what the email links to, and nothing it does can be steered to an arbitrary host.

The issue planned WHOIS and a URLhaus or PhishTank snapshot. Checked on 2026-11-03:

- URLhaus's public downloads (`/downloads/csv_recent/`, `/downloads/text_online/`) answer
  `410 Gone`.
- PhishTank's verified-online CSV still downloads without a key: 50,575 URLs with the
  impersonated brand.
- OpenPhish's community feed is public: 300 URLs.
- IANA's RDAP bootstrap maps 1,203 TLDs to their registry's RDAP server. `.de` and `.ch` have
  none.

## Options

### Domain age

- **A. WHOIS (port 43).** Free text in a different format per registry, no TLS, a parsing
  library to trust; many registries now redact or rate-limit it.
- **B. RDAP (RFC 9083).** JSON over HTTPS, the standard that replaces WHOIS for gTLDs, one
  server per TLD from IANA's bootstrap. Not every ccTLD offers it.
- **C. A commercial API.** One call for every TLD, but a key, a cost, and every looked-up domain
  sent to a vendor.

### Reputation

- **D. Live lookup APIs (Safe Browsing and similar).** Fresh, but every URL of every
  investigated email leaves for a third party, and a key is needed.
- **E. Local snapshots of public feeds.** Nothing leaves the machine at query time; freshness
  is as good as the last download, and feeds lag new campaigns.

## Decision

**B and E, behind a fence.**

- `domain_age` asks the registry's RDAP server, chosen from the IANA bootstrap bundled in the
  package (`agent/tools/data/rdap_dns.json`). Every request goes through `AllowlistTransport`,
  which refuses anything but HTTPS to those registry hosts before a connection is made.
  Redirects are not followed. Only registrable domains that appear in the email (sender,
  reply-to, return-path, links) can be looked up. Timeout 5 s, answers cached for a day.
- `lookup_reputation` searches PhishTank and OpenPhish snapshots downloaded by
  `make reputation`. The feeds change hourly, so their SHA-256 cannot be pinned in the repository
  like the corpora's: it is recorded at download and checked at every load. Snapshots older than
  7 days are reported as stale. The feeds are never committed.
- URLs are compared without scheme or fragment and after HTML unescaping (PhishTank writes
  `&amp;`). A listing of the exact URL is evidence. A listing of the same host is reported as a
  count, with `shared_platform` set when the host carries 50 listed URLs or more or is a URL
  shortener: on the 2026-11-03 snapshot the median host carries 1 listed URL and the 99th
  percentile 4, while `docs.google.com` carries 4,176. Matching on the host there would flag
  every Google Form.

## Consequences

- One real check, against live services: a Netflix lure from the PhishTank feed sat on a `.com`
  registered 5 days earlier at a low-cost registrar. Both facts came back in two tool calls,
  without touching the lure.
- Domains under TLDs without RDAP (`.de`, `.ch`) get `no_rdap_service`: the agent must not read
  it as "old" or "safe".
- Not being listed proves little. Reputation helps on known campaigns; new ones are what the
  classifier and the other tools are for.
- PhishTank and OpenPhish data stay under their providers' terms. A commercial deployment would
  need their licences or a different feed.
- The bootstrap is a snapshot refreshed by hand, like the Public Suffix List. A TLD that moves
  registry needs a refresh.
