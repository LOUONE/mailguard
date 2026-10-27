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

## Header features

Code: [`src/mailguard/features/headers.py`](../src/mailguard/features/headers.py). Only five
headers are read: `From`, `Reply-To`, `Return-Path`, `Message-ID` and `Date`. Everything else
(`X-Spam-*`, `Received`, list headers, authentication headers) is removed before extraction, and a
`hypothesis` property checks that adding any other header never changes a feature.

The corpus maintainers replaced real domains with `spamassassin.taint.org` and `monkey.org`. On the
training split, `Return-Path` is under `taint.org` for 24.7% of ham and `From` for 17.0%, and 7.0%
of phishing display names contain `monkey.org`. Those domains count as **unknown**: an unknown
domain never matches or differs from another one.

| Feature | Definition |
|---|---|
| `hdr_from_missing` | No usable `From` address |
| `hdr_reply_to_present` | A `Reply-To` exists |
| `hdr_reply_to_mismatch` | `Reply-To` and `From` have different registrable domains |
| `hdr_return_path_mismatch` | `Return-Path` and `From` have different registrable domains |
| `hdr_display_name_has_address` | The display name contains an address (`"service@bank.com" <x@evil.example>`) |
| `hdr_display_name_domain_mismatch` | A domain written in the display name differs from the sender's |
| `hdr_display_name_brand_mismatch` | The display name names a well-known brand that the sender's domain does not contain |
| `hdr_message_id_missing` | No `Message-ID` |
| `hdr_message_id_domain_mismatch` | The `Message-ID` domain differs from the sender's |
| `hdr_date_missing` | No parsable `Date` (presence only, never the value) |
| `hdr_defects` | Number of MIME defects found by the parser |

**SPF, DKIM and DMARC** are parsed by
[`features/auth.py`](../src/mailguard/features/auth.py) but are not model features (they exist only
in recent phishing). The parser only trusts the `Authentication-Results` header of a given
`authserv-id`, since any sender can write one; the Phase 3 agent will use it on live mail.

## Attachment features

The parser now sniffs each attachment's first bytes and stores the file family (`pdf`, `zip`,
`exe`, `ole`, `html`, ...) next to the hash, then drops the payload as before
([ADR 0002](adr/0002-parser-design.md)).

| Feature | Definition |
|---|---|
| `att_count` | Attachments |
| `att_risky_extension` | Last extension runs code or renders a page (`exe`, `js`, `html`, `iso`, `docm`, ...) |
| `att_archive` | Last extension is an archive |
| `att_double_extension` | A known extension hides in front of a risky or archive one (`invoice.pdf.exe`) |
| `att_type_mismatch` | The sniffed family differs from the one implied by the name or declared type |

Attachments are rare in these corpora (3.2% of ham, 5.7% and 17.5% of old and recent phishing).
Risky extensions appear in 5.6% of recent phishing and almost never elsewhere.

## Leakage audit

`uv run python scripts/audit_features.py` trains a gradient-boosted model to tell 2004-2007 from
2015-2025 phishing, on training phishing only, with folds grouped by duplicate cluster. Every row
is phishing, so any skill comes from time. All ham dates from 2002, so a group that encodes the era
can separate ham from phishing for the wrong reason. Above an AUC of 0.95, the group is
investigated feature by feature.

A high era AUC has two possible causes, and only one is leakage:

1. **The collection pipeline**, meaning how and where the messages were received, stored or
   anonymised. Such a feature describes the corpus, not the sender, and would not transfer to a
   real inbox. These are removed.
2. **Attackers changing their methods.** Such a feature describes the sender, and ham exhibits it at
   realistic rates. These are kept, and their effect is measured by the per-era results and the
   temporal check (#27).

| Group | Era AUC, first run | Era AUC, after removals | Top features (univariate AUC) |
|---|---|---|---|
| headers | 0.937 | 0.854 | `hdr_message_id_domain_mismatch` 0.73, `hdr_reply_to_present` 0.67 |
| attachments | 0.557 | 0.557 | `att_count` 0.56 |
| urls | 0.956 | 0.956 | `url_keyword_hits` 0.80, `url_count` 0.71, `url_distinct_domains` 0.70 |
| all groups | 0.990 | 0.980 | |

**Removed** (cause 1):

- `hdr_received_hops`: each relay on the recipient's side adds a `Received` line. Ham has a median
  of 6 hops (mailing lists and the maintainer's own fetching chain), 2004-2007 phishing 3, recent
  phishing 2. The count measures the receiving infrastructure of each corpus.
- `hdr_recipients`: in 2004-2007 phishing, a `To` header exists in 99.9% of messages but holds an
  address in only 68.8% (ham: 96.2%). The anonymisation removed the others.

**Kept** (cause 2): the URL group, above the threshold at 0.956. No single URL feature exceeds 0.80;
the era shows through their combination: raw IP hosts fall from 36.5% to 0.5% of phishing, login
keywords from 71.1% to 14.3%. These are choices made by the attacker, and ham carries them at
plausible rates. The consequence is accepted and measured: URL features should help more on old
phishing than on recent phishing, and the per-era results will show by how much.

## Text features

Code: [`src/mailguard/features/text.py`](../src/mailguard/features/text.py). The body is read
through `Email.body_text`, plain text or visible HTML text alike, and cut at 20,000 characters
(0.38% of messages are longer; one is over a million characters).

**TF-IDF** (`TextFeatures`): word 1-2 grams (50,000 terms at most) and character 3-5 grams within
word boundaries (100,000 at most), `min_df=3`, sublinear term frequency, float32 sparse output.
Character n-grams hold up better against `v3rify` or `P@yPal` than whole words. This is the only
extractor with state: its vocabulary and IDF weights are learnt in `fit`, which runs inside the
training pipeline on the training split only, and `transform` refuses to run before it. The
vocabulary size is logged at each fit. Fitting the 9,859 training messages takes about 25 seconds.

**Scrubbing before TF-IDF.** A first logistic regression on raw body text ranked these among its
strongest terms: `jose` (the phishing corpus owner, in 26.2% of phishing and 0.4% of ham), `monkey`
and `org` (his domain), `2002` and `2025` (years), `spamassassin` and `xent` (the ham maintainers'
software and mailing lists). They describe who received the message and when, not what the sender
wrote. Before vectorising, the text is scrubbed:

- addresses become `emailaddress` and URLs `urladdress` (URL features already describe them);
- digits become `0`, which removes years;
- the words of the recipients' own addresses are removed (`jane.doe@...` removes `jane` and `doe`);
- the corpus names `jose`, `nazario`, `monkey`, `spamassassin`, `taint` are removed, with any
  domain labels that follow them.

After scrubbing, none of these terms is left in the top weights; `org` drops from 8.2% to 0.3% of
phishing. The strongest phishing terms become `your account`, `verify`, `update your`, `kindly`,
`mailbox`, `upgrade`; the strongest ham terms are mailing-list vocabulary (`wrote`, `mailing list`,
`unsubscribe`). The lexicon below is computed on the unscrubbed text, since its cues
(`within 24 hours`) need the digits.

**Lexicon** (`LexiconFeatures`, stateless), for the body and for the subject:

| Feature | Definition |
|---|---|
| `*_urgency_cues` | Distinct pressure cues present: urgent, suspended, expires, within 24 hours, locked, unusual activity, ... (12 patterns) |
| `*_credential_cues` | Distinct credential cues present: password, sign in, verify, account, card number, billing, ... (12 patterns) |
| `*_uppercase_ratio` | Uppercase share of letters |
| `*_exclamation_ratio` | `!` share of non-space characters |
| `body_empty` | No visible text at all (image or attachment only, 2.1-2.3% of phishing) |

Cues count once per message, so the feature measures how many tactics are used, not how long the
message is. Mean values on the training split:

| Feature | Ham | Spam | Phishing 2004-2007 | Phishing 2015-2025 |
|---|---|---|---|---|
| `body_urgency_cues` | 0.07 | 0.22 | 0.79 | 0.51 |
| `body_credential_cues` | 0.11 | 0.25 | 2.34 | 1.22 |
| `subject_credential_cues` | 0.00 | 0.02 | 0.62 | 0.37 |
| `body_empty` | 0.00 | 0.01 | 0.02 | 0.02 |

The cues are English only. Non-English phishing relies on TF-IDF and the other feature groups.
