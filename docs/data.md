# Data

This document lists the corpora mailguard is trained and evaluated on, their licences, what is
inside them, and the biases that come with them. Raw data is never committed: it is downloaded
into `data/` (gitignored) by `scripts/download_data.py`, which pins every file by SHA-256.

## Selected corpora

| Corpus | Source | Licence | Files used | Messages | Label | Period (`Date` header) |
|---|---|---|---|---|---|---|
| SpamAssassin public corpus | [spamassassin.apache.org/old/publiccorpus](https://spamassassin.apache.org/old/publiccorpus/) | No formal licence, published for spam-filter testing (see below) | `20030228_easy_ham`, `20030228_easy_ham_2`, `20030228_hard_ham` | 4,150 | `ham` | 2002 |
| SpamAssassin public corpus | same | same | `20030228_spam`, `20050311_spam_2` | 1,896 | `spam` | 2002-2003 |
| Nazario phishing corpus | [monkey.org/~jose/phishing](https://monkey.org/~jose/phishing/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `phishing0.mbox` to `phishing3.mbox` | 4,572 | `phishing` | 2004-2007 |
| Nazario phishing corpus | same | same | `phishing-2015` to `phishing-2025` | 3,466 | `phishing` | 2015-2025 |
| **Total** | | | 20 files, ~147 MB | **14,084** | | |

Class balance of the combined set: 29.5% ham, 13.5% spam, 57.1% phishing.

### Licences

- **SpamAssassin.** The corpus [readme](https://spamassassin.apache.org/old/publiccorpus/readme.html)
  states that every message "was posted to public fora, sent [...] in the knowledge that [it] may
  be made public, [...] or originated as newsletters", and that "copyright for the text in the
  messages remains with the original senders". There is no licence file. mailguard therefore
  downloads it from the original host at setup time and never redistributes messages, in the repo
  or in test fixtures.
- **Nazario.** CC BY 4.0 per the corpus `README.txt` and `LICENSE.txt` (both checked on
  2026-10-19). Commercial use and derivative works are allowed; attribution is required:
  *Phishing corpus by Jose Nazario, https://monkey.org/~jose/phishing/, CC BY 4.0.*

### Considered and not selected

| Corpus | Reason |
|---|---|
| Enron-Spam (Metsis et al., 2006) | Archives are still online, but the dataset page that stated the terms of use returns 404. Licence cannot be verified today. |
| CEAS 2008 | No official distribution point. The copies on Kaggle and GitHub have unclear provenance and no licence. |
| `20051114.mbox` (Nazario) | 239 of its 438 messages also appear in `phishing1.mbox`. |
| `private-phishing4.mbox` (Nazario) | Overlaps the yearly files, and the `private-` prefix suggests it was not meant for general use. Excluded by caution. |
| `20021010_*`, `20030228_spam_2` (SpamAssassin) | Older snapshots of the selected sets, superseded by the 2003 and 2005 archives. |

## Labels

Labels come from the corpus a message belongs to, not from per-message annotation:

- `ham`: legitimate mail (mailing lists, newsletters, personal mail) from SpamAssassin.
- `spam`: unsolicited bulk mail from SpamAssassin. Not phishing, but not legitimate either.
- `phishing`: hand-classified phishing from Nazario's own inbox.

The three labels are kept as-is in the parsed dataset. How `spam` enters the binary
phishing-vs-rest target is a modelling decision, recorded as an ADR before the first baseline.

Expected label noise: Nazario notes his classification is manual and may contain errors, and a few
2002-era SpamAssassin spam messages are phishing in all but name.

## Personal data

- SpamAssassin: addresses and hostnames were partially obfuscated by the maintainers (replaced
  with `spamassassin.taint.org`). Most headers are otherwise as received.
- Nazario: early mailboxes (2004-2007) were anonymised. **The 2015-2025 files are not**: recipient
  addresses, receiving hosts and IPs are real. The recipient is the corpus author.
- Policy for this repo: no raw message, header or address from these corpora is committed (code,
  fixtures, notebook outputs, docs). Test fixtures are synthetic. The serving layer, from phase 2,
  never logs email content.

## Known biases and leakage risks

These were measured on the raw files before any modelling. Each one is a shortcut a classifier can
learn instead of learning what phishing looks like.

| Risk | Evidence | Planned mitigation |
|---|---|---|
| **Source artefacts in headers** | `spamassassin.taint.org` appears in 71% of ham headers and 37% of spam headers, never in phishing. `monkey.org` appears in 50% of phishing headers, `X-Spam-*` headers in 52%, both ~0% elsewhere. | Header features use an allowlist of meaningful fields. Known artefacts are stripped during feature extraction. A headers-only "guess the source" model is trained as an audit: if it is near-perfect, the features leak. |
| **Era gap** | All ham is from 2002. Phishing spans 2004-2025. Mail formats, HTML, encodings and clients changed a lot in between: 4.5% of ham messages have an HTML body, against 94.1% of phishing. | `Date` and year-derived values are never used as features. "Has HTML" is not used as a standalone feature, and text features are computed on the same extracted text whether it came from plain text or HTML. Results are reported separately for 2004-2007 and 2015-2025 phishing. |
| **Authentication headers** | `Authentication-Results`: 0% of ham, 0% of 2004-2007 phishing, 61% of 2015-2025 phishing. `DKIM-Signature`: 0% vs 44%. DKIM did not exist in 2002. | SPF/DKIM/DMARC features cannot be validated on these corpora. They are implemented but excluded from the trained model (or measured by ablation), and used by the phase 3 agent on live headers instead. |
| **No modern ham** | No public corpus of recent legitimate mail with full headers and a verified licence was found. | Listed as a limitation in the model card. A global temporal split is impossible (no ham after 2002). Temporal generalisation is measured on phishing only: train on phishing up to 2019, measure recall on 2020-2025 at a threshold fixed on ham. |
| **Unrealistic base rate** | 57% phishing here vs well under 1% in a real mailbox. | Accuracy is not reported. Headline metrics are PR-AUC and recall at a fixed false-positive rate, which do not depend on the class prior. |
| **Duplicates** | 21.3% of messages repeat an earlier body once case, Unicode, URLs and numbers are normalised: 41.5% of 2004-2007 phishing, 20.6% of 2015-2025 phishing, 18.4% of spam, 1.9% of ham. Largest cluster: 71 copies. Details in `notebooks/01_eda.ipynb`. | Splits group messages by normalised body fingerprint (`mailguard.data.dedup`), so a campaign never sits on both sides. |
| **Broken `Date` headers** | Some parsed years are 102, 1970, 2028, 2030 or 2038, and 55 messages in `phishing-2023` have no parsable date. | `Date` is treated as untrusted input. Time-based analysis uses the corpus file period, not the header. |
| **Malformed messages** | At least one message in `phishing3.mbox` makes the stdlib `email` generator raise `UnicodeEncodeError` when re-serialised. | The parser is property-tested with `hypothesis` to never raise on arbitrary bytes (issue #12). |

## Reproducing

```bash
make data   # or: uv run python scripts/download_data.py [--corpus nazario] [--force]
```

Files land in `data/raw/<corpus>/<original filename>`. The manifest of URLs, labels and checksums
lives in [`src/mailguard/data/sources.py`](../src/mailguard/data/sources.py). Re-running is a no-op
when every file is present with the right checksum. A file whose checksum no longer matches upstream
fails loudly instead of being accepted silently: if a corpus is updated upstream, the new checksum
is reviewed and committed like any other change.
