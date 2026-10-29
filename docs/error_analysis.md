# Error analysis and last leakage check

What the LightGBM baseline gets wrong, why, and whether its explanations show any shortcut. This
page holds categories and counts only: messages were read locally with
`scripts/review_errors.py`, which prints scrubbed text to the terminal, and nothing from them is
committed.

## Method

- Model: registered `mailguard-lightgbm` v0.1.0 (calibrated, still with `url_max_length`), at
  the 1% operating point, i.e. the threshold that flags 1% of validation ham. The label audit
  and the slices were run later and say which version they measure.
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
  holds the spam, newsletters and mail he received. Some of the "errors" are the model being
  right, but the same noise also sits among the messages the model flags, where it counts as
  caught phishing. Reading misses alone cannot say which way recall moves: the
  [label audit](#label-audit-a-random-sample-of-phishing) below measures both sides.
- **14 are real misses, and the phishing signal is diluted or absent from the text the model
  reads**: other languages (the lexicon is English only), padding with unrelated text, and lures
  that hide in an attachment or image. These are the attacks Phase 4 will build deliberately
  (#50), and the cases the Phase 3 agent should take over.
- **"Little text" is not the weak spot, padding is.** The first reading of these misses suggested
  that short or non-English messages were the problem. The slices added to the report do not
  support it: on test, LightGBM catches every phishing message under 20 words (87) and every one
  with little English (78), helped by subject, attachment and header features. Its recall is
  lowest on messages of 100 words or more (98.5% [97.1, 99.6]), where junk or encyclopedia text
  buries a short lure. The slices also show where false alarms come from: no test ham of 100
  words or more was flagged by v0.1.0, and 1 of 394 by v0.1.1; all the others are shorter.

## Label audit: a random sample of phishing

Errors are a biased sample: the model rejects spam, so spam stored in the phishing corpus piles up
among the misses. To measure the noise of the phishing label itself, 100 test messages labelled
phishing were drawn at random (`scripts/review_errors.py --kind sample --splits test`, seed fixed
in the script) and read like the errors.

| Reading | Count | Flagged at 1% |
|---|---|---|
| Phishing (credential, payment, callback, malware lures; padded or not) | 97 | 97 |
| Not phishing: an advertisement for an auction-selling kit | 1 | 1 |
| Cannot tell: a newsletter template under a delivery subject; a body in an encoding that does not decode | 2 | 2 |

- **The phishing label is 1% to 3% wrong** (1 clear case, 3 counting the uncertain ones; the
  exact 95% interval of 1 in 100 is 0.03% to 5.5%). All 100 sampled messages were flagged.
- **On test, the 15 misses split into 6 real misses, 5 advance-fee frauds (one sent four times)
  and 4 messages that are not phishing** (casino, pharmacy and reseller spam, a newsletter).
- **Recall corrected on both sides moves from 98.8% to about 99.1%.** Removing the 4 noisy misses
  and 1% to 3% of the 1,191 flagged messages gives 99.06% to 99.08%; also counting advance-fee
  fraud as out of scope gives about 99.5%. The difference stays inside the bootstrap interval
  of the measured recall, so the measured figure is the one to quote, and label noise does not
  change any conclusion.

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
  push ham towards phishing, exactly what the false positives above show. Its effect on modern
  ham cannot be measured without modern ham; what it does to the model can, and the report now
  includes an ablation, run on two splits:
  - **On the test split, the model does not need it.** Without it, recall at 1% FPR is the same
    or better: +0.1 points [0.0, 0.3] on the first split, +0.5 [0.2, 0.9] on the reproducible one
    (v0.1.2). It flags about 2 points more spam on both.
  - **On new campaigns, the evidence did not hold.** On the first split, the model without it
    caught 97.6% of 2020-2025 phishing instead of 95.3%; on the reproducible split, 96.1% instead
    of 96.7%, inside the intervals. That check sets its threshold on the ham of one validation
    split, and 850 spam messages changing split was enough to reverse it. A single split cannot
    measure an effect of a couple of points.
  - The feature still encodes the era of the ham corpus by construction. It was removed from the
    served model in v0.1.1 ([ADR 0006](adr/0006-drop-url-max-length.md), and its update), and the
    report keeps measuring both variants.
- **Mailing-list vocabulary reflects the composition of the ham corpus**, which is mostly mailing
  lists. It is real ham behaviour, not an artefact, but it means the model knows one kind of
  legitimate mail far better than the others (personal mail, transactional mail).
- Nothing else in the top 20 points to a source artefact. The rest is credential and urgency
  vocabulary, URL structure, and sender consistency.
