# Features

Every feature is computed from a parsed `Email` (`parse_email(raw)`), the same object the API will
build in production. Extractors are scikit-learn transformers over a list of `Email`, so the whole
chain, from parsing to the classifier, is one `Pipeline` that is fitted on the training split only.

Rules that apply to every group, from [`docs/data.md`](data.md#known-biases-and-leakage-risks):

- No date, year or anything derived from them.
- No standalone "has HTML" flag: text is read from `Email.body_text` whatever the format.
- No SPF / DKIM / DMARC result in the model: they only exist in recent phishing. They are parsed for
  the Phase 3 agent.
- Nothing touches the network. Reference lists are bundled snapshots with a source and a date.

## URL features

Code: [`src/mailguard/features/urls.py`](../src/mailguard/features/urls.py). One row per message,
from `Email.urls` (text and HTML links) and the anchors of the HTML body.

| Feature | Definition |
|---|---|
| `url_count` | URLs with a usable host |
| `url_distinct_domains` | Distinct registrable domains (see below) |
| `url_ip_hosts` | Hosts written as an IP, including the integer and hex forms browsers accept |
| `url_max_length` | Longest URL, in characters |
| `url_max_host_entropy` | Highest Shannon entropy of a host, in bits per character |
| `url_max_subdomain_depth` | Most labels in front of a registrable domain (`a.b.bank.com` is 2) |
| `url_punycode_hosts` | Hosts with an `xn--` label (internationalised, possible lookalike) |
| `url_userinfo` | URLs with `@` in the authority (`http://bank.com@evil.example/`) |
| `url_shorteners` | Links through a known shortener |
| `url_suspicious_tlds` | Hosts under one of the 20 most abused TLDs |
| `url_nonstandard_ports` | Explicit port other than 80 or 443, or an invalid one |
| `url_keyword_hits` | URLs containing a login-flow word (`login`, `verify`, `account`, ...) |
| `url_anchor_mismatches` | Links whose text names a domain other than their target |

Left out on purpose: the share of `https` links. It mostly separates 2002 from 2020.

**Registrable domains** come from a snapshot of the [Public Suffix List](https://publicsuffix.org)
bundled in the package (`src/mailguard/features/data/public_suffix_list.dat`, MPL 2.0, version
`2026-10-22_08-36-41_UTC`, SHA-256 `0232f4df...49e9ca`). `login.bank.co.uk` and `www.bank.co.uk`
both belong to `bank.co.uk`. A test blocks every socket function and runs the extraction, so a
runtime download cannot slip in. To refresh the snapshot, download it from the URL above, check the
diff, and update the version and checksum here.

**Reference lists** ([`src/mailguard/features/lists.py`](../src/mailguard/features/lists.py)):

- Suspicious TLDs: the 20 TLDs with the highest phishing score (phishing domains per 10,000
  registered) in Interisle's *Phishing Landscape 2025* (May 2024 to April 2025), via the
  [Cybercrime Information Center](https://www.cybercrimeinfocenter.org/phishing-activity-in-tlds-may-april-2025).
  None of them existed in 2002, so on these corpora the feature can only fire on recent phishing.
  It is a real signal for a modern inbox, but on this data it also partly encodes the era. The
  leakage audit (#23) measures how much.
- Shorteners: 19 common public services, checked by hand.

**On the training split** (share of messages where the feature is non-zero):

| Feature | Ham | Spam | Phishing 2004-2007 | Phishing 2015-2025 |
|---|---|---|---|---|
| any URL | 87.5% | 82.1% | 97.8% | 89.9% |
| IP host | 0.2% | 20.2% | 36.5% | 0.5% |
| login-flow keyword | 3.6% | 1.0% | 71.1% | 14.3% |
| anchor text / target mismatch | 1.8% | 0.8% | 52.1% | 13.1% |
| shortener | 0.1% | 0.0% | 0.0% | 5.6% |
| suspicious TLD | 0.0% | 0.0% | 0.0% | 1.3% |
| non-standard port | 0.4% | 3.0% | 7.5% | 0.2% |

The strongest URL signals of 2004-2007 phishing (raw IPs, keywords, mismatched anchors) are four to
seventy times rarer in recent phishing: attackers moved to real domains, shorteners and hosted
pages. Expect URL features to help much more on old phishing than on new, and the temporal check
to show it.
