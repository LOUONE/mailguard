# Error analysis and last leakage check

What the LightGBM baseline gets wrong, why, and whether its explanations show any shortcut. This
page holds categories and counts only: messages were read locally with
`scripts/review_errors.py`, which prints scrubbed text to the terminal, and nothing from them is
committed.

## Method

- Model: registered `mailguard-lightgbm` (calibrated), at the 1% operating point, i.e. the
  threshold that flags 1% of validation ham.
- Errors: every false positive and false negative on the validation and test splits (test alone
  holds too few for a taxonomy: 5 ham and 11 spam false positives, 15 false negatives).
- Each message was read and given one category, with the features that pushed it most
  (`mailguard.models.explain`).

## False positives: 24 legitimate messages flagged (7 ham, 17 spam)

| Category | Count | Ham / spam | What drove the score |
|---|---|---|---|
| Notifications with very long tracking URLs (a store newsletter; one group-platform notice sent five times) | 6 | 5 / 1 | `url_max_length` |
| Bulk marketing spam (address lists, cheap sales) | 6 | 0 / 6 | credential words in a marketing text ("your account") |
| Almost no visible text (empty body or a single unsubscribe line) | 6 | 0 / 6 | attachments, a few character n-grams |
| Non-English spam (Korean, Chinese) | 3 | 0 / 3 | character n-grams outside the English vocabulary |
| Advance-fee fraud labelled spam | 1 | 0 / 1 | bank and account vocabulary |
| Commercial email from a real brand about account and payment details | 1 | 1 / 0 | `your account` |
| Delivery failure notice | 1 | 1 / 0 | short text, return-path mismatch |

- **Only 7 of the 24 are ham, and 4 of those are copies of one notice.** At the 1% operating
  point the model mostly confuses spam with phishing, which ADR 0001 rates as the cheaper error.
- **Long tracking URLs are the main ham failure.** See the leakage check below: this is the one
  feature whose signal may come from the era of the corpus.
- 1 of the 17 spam false positives is arguably phishing (an advance-fee scam), which is the label
  noise ADR 0001 anticipated.

## False negatives: 30 phishing messages missed

| Category | Count | Real miss? |
|---|---|---|
| Not phishing: spam, newsletters or a personal message stored in the phishing corpus | 11 | No: label noise |
| Advance-fee or lottery fraud, no credential request (one message sent four times) | 5 | Ambiguous under ADR 0001 |
| Non-English credential phishing (German, Japanese, Chinese) | 3 | Yes |
| Phishing subject over unrelated junk text (filter evasion) | 3 | Yes |
| Delivery lure padded with encyclopedia text | 3 | Yes |
| Lure with almost no text: shared-document notice, content in an attachment or image | 3 | Yes |
| "You received a postcard" malware lure | 2 | Yes |

- **11 of the 30 misses are not phishing.** The phishing corpus is one person's inbox and also
  holds the spam, newsletters and mail he received. The measured recall is therefore a lower
  bound on the real one, and some of the "errors" are the model being right.
- **14 are real misses, and they share a pattern: the phishing signal is diluted or absent from
  the text the model reads.** Other languages (the lexicon is English only), padding with unrelated
  text, and lures that hide in an attachment or image. These are the attacks Phase 4 will build
  deliberately (#50), and the cases the Phase 3 agent should take over.

## Last leakage check

The 20 features with the largest mean absolute contribution on test (TreeSHAP, in
[`reports/evaluation.md`](../reports/evaluation.md)) were checked against every risk in
[`docs/data.md`](data.md#known-biases-and-leakage-risks) and
[`docs/features.md`](features.md#leakage-audit).

| Known risk | In the top 20? |
|---|---|
| Corpus domains (`spamassassin.taint.org`, `monkey.org`) and the recipient's name | No (allowlist, unknown domains, text scrubbing) |
| `X-Spam-*`, `Received` hops, recipient count | No (not read or removed) |
| Dates and years | No (no date feature, digits scrubbed) |
| "Has HTML" | No (no such feature) |
| SPF / DKIM / DMARC | No (kept out of the model) |
| **URL length** | **Yes, rank 1** (see below) |
| Mailing-list vocabulary (`list`, quoted `> `) | Yes, ranks 4 and 19 (see below) |

- **`url_max_length` is the strongest feature and a probable era confound.** 2002 ham has short
  URLs (median 48 characters on the training split); phishing of both eras has long ones (about
  70). Modern legitimate mail is full of long tracking URLs, so in a real inbox this feature would
  push ham towards phishing, exactly what the false positives above show. It cannot be measured
  without modern ham, so it is kept, listed as the first limitation in the model card, and watched
  by the drift report in Phase 2 (#38). A model without it is the first experiment once modern ham
  is available.
- **Mailing-list vocabulary reflects the composition of the ham corpus**, which is mostly mailing
  lists. It is real ham behaviour, not an artefact, but it means the model knows one kind of
  legitimate mail far better than the others (personal mail, transactional mail).
- Nothing else in the top 20 points to a source artefact. The rest is credential and urgency
  vocabulary, URL structure, and sender consistency.
