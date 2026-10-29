# Model card: mailguard-lightgbm v0.1.1

The phishing classifier behind mailguard, as of release `v0.1.1`. Every metric on this page comes
from [`reports/evaluation.md`](../reports/evaluation.md), which `make evaluate` regenerates
byte for byte; brackets are 95% intervals from a bootstrap over duplicate clusters.

**Changes since v0.1.0.** `url_max_length`, an era marker, is left out of the model
([ADR 0006](adr/0006-drop-url-max-length.md)). Same recall at 1% FPR; recall at the strict
threshold rises from 96.4% to 98.3%, and on unseen 2020-2025 campaigns from 95.3% to 97.6%; spam
flagged rises from 3.9% to 6.0%. The calibrated operating thresholds are now recorded on the
registered model version, for the API.

## Model details

- **What it is.** A scikit-learn pipeline from a parsed email to a probability: header, attachment,
  URL and lexicon features plus word and character TF-IDF on the visible body text, a LightGBM
  classifier, and a sigmoid calibration fitted on validation. Registered in MLflow as
  `mailguard-lightgbm`, saved with skops (loading never runs code, [ADR 0003](adr/0003-model-serialisation.md)).
  Its version tags hold the calibrated threshold of each operating point (`threshold_1pct`,
  `threshold_01pct`).
- **Why this model.** Best recall at both operating points and on new campaigns, structured
  explanations; a text-only logistic regression is kept as a challenger
  ([ADR 0005](adr/0005-model-choice.md)).
- **Inputs.** One raw RFC 5322 message. Only five headers are read (`From`, `Reply-To`,
  `Return-Path`, `Message-ID`, `Date` presence), plus attachment metadata, URLs and the visible
  body text, cut at 20,000 characters ([features](features.md)).
- **Output.** `phishing_probability` at the corpus base rate (57.1% phishing), and up to five
  reasons as additive log-odds contributions ([explanations](error_analysis.md)). A probability
  for a real mailbox is obtained with `adjust_prior` (below).
- **Target.** Phishing against ham and spam ([ADR 0001](adr/0001-binary-target.md)).

## Intended use

- Scoring incoming mail **next to** a spam filter, to flag credential phishing, impersonation and
  malicious links, in real time.
- Routing: confident scores are acted on directly; a band of uncertain, calibrated scores goes to
  the LLM investigation agent (Phase 3).
- A portfolio and research artefact: the training data are public corpora, not a customer's mail.

**Out of scope.** Spam filtering (spam is a negative class, not a target); blocking mail without a
human or a second check; any language other than English for the lexicon features; deciding on
image-only or attachment-only mail, where the model sees almost nothing.

## Data

[`docs/data.md`](data.md): 14,084 messages from the SpamAssassin public corpus (4,150 ham and
1,896 spam, 2002-2003) and the Nazario phishing corpus (8,038 phishing, 2004-2007 and
2015-2025, CC BY 4.0). Split 70 / 15 / 15, stratified, grouped by duplicate cluster so a campaign
never sits on both sides ([ADR 0004](adr/0004-grouped-splits.md)).

Test split: **2,113 messages** (622 ham, 285 spam, 1,206 phishing) in 1,670 duplicate clusters,
never used to choose anything.

## Metrics

Thresholds are fixed on validation ham, then applied to test as they are.

| Metric | Value |
|---|---|
| Recall at the 1% ham-FPR threshold | **98.8% [97.9, 99.6]** |
| Ham FPR measured on test at that threshold | 1.0% [0.0, 2.7] (6 messages) |
| Spam flagged at that threshold | 6.0% [3.3, 9.2] |
| Recall at the strict threshold (0.1% target: no validation ham flagged) | 98.3% [97.3, 99.2] |
| Ham FPR measured on test at that threshold | 0.8% [0.0, 2.4] (5 messages) |
| PR-AUC (depends on the 57% prior; for comparison only) | 0.9992 [0.9986, 0.9997] |

**What 622 test ham can measure.** One ham message is 0.16% of the FPR, so every FPR above is a
count of a handful of messages. A 0.1% FPR can be neither shown nor ruled out: proving it would
take about 3,000 test ham with none flagged (rule of three). The strict threshold is therefore a
stricter setting with a measured 0.8% FPR, not a 0.1% operating point.

**By phishing era**, at the 1% threshold: 98.5% [97.1, 99.7] on 2004-2007 phishing and
99.2% [98.1, 100.0] on 2015-2025 phishing.

**Against the text-only challenger**, paired: +0.9 points of recall at 1% FPR [0.3, 1.8],
McNemar p = 0.46; +3.8 points at the strict threshold [2.6, 5.3], p = 2.4e-9.

**New campaigns.** Retrained on phishing up to 2019 only, the same model catches
**97.6% [96.8, 98.4]** of 1,800 phishing messages from 2020-2025, against 98.8% [97.3, 100.0]
when 2020-2025 campaigns are in its training data: it loses 1.2 points on years it has never
seen, as the challenger does (96.5%). With `url_max_length` (v0.1.0) it lost 3.5.

## Calibration and base rate

Sigmoid calibration, chosen over isotonic by cross-validated log loss on validation. On test, ECE
goes from 1.1% [0.6, 2.0] to **0.9% [0.5, 1.7]**, Brier score from 0.0138 to 0.0136; the
reliability diagram is in the report.

Probabilities hold at the corpus base rate of 57% phishing. In a mailbox where 1% of mail is
phishing, out of 1,000 messages (10 phishing, 990 ham), the 1% operating point would:

- catch **9.9** of the 10 phishing messages;
- flag **9.5 [0.0, 26.8]** legitimate messages, so a precision of 50.9% [27.1, 100.0]. The
  interval is wide because it rests on six flagged test ham: between no false alarm and more
  than two per phishing message caught. At 0.1% phishing, precision falls to 9.3%;
- turn a corpus probability of 0.9 into **0.064**, and 0.99 into 0.429
  (`mailguard.models.calibration.adjust_prior`, Bayes' rule on the odds).

Spam is left out of these counts: it would add false alarms of the cheaper kind (6.0% of spam is
flagged), in a proportion that depends on the spam filter in front of the model.

The serving layer must apply the base rate of its own mailbox before showing a probability.

## Explanations

On test, 76.7% of the model's mean absolute contribution comes from text, 9.7% from the lexicon,
9.3% from URLs, 3.2% from headers and 1.1% from attachments. The strongest features are now
credential cues in the subject and body, login words in URLs and host entropy. The top features and a local
explanation of a synthetic email are in the report; the leakage check of the top 20 is in
[`docs/error_analysis.md`](error_analysis.md#last-leakage-check).

## Error analysis

From the [error taxonomy](error_analysis.md) (validation and test, 1% threshold). The taxonomy
was read on the v0.1.0 model; the slices below are those of v0.1.1.

- **False positives (24: 7 ham, 17 spam).** Mostly spam taken for phishing, the cheaper error.
  The ham errors are notifications with very long tracking URLs, a commercial email about account
  details and a bounce.
- **False negatives (30).** 11 are not phishing at all (spam and newsletters stored in the phishing
  corpus) and 5 are advance-fee fraud without a credential request. In the 14 real misses the
  lure is diluted or hidden: junk or encyclopedia padding, other languages, an attachment or an
  image.
- **Slices.** Short and non-English phishing is not the weak spot: every test phishing message
  under 20 words or with little English is caught. Recall is lowest on long, padded messages
  (98.5% at 100 words or more), and false alarms fall on short ham: 2-4% of ham under 100 words
  is flagged, against 1 of 394 longer ones.
- **Label noise.** In a random sample of 100 test phishing messages, 1 to 3 are not phishing,
  and all 100 were flagged: the noise sits among detections as well as misses. Corrected on both
  sides, recall would be about 99.1% instead of 98.8%, inside the interval, so the measured
  figure stands.

## Limitations

Each risk measured in [`docs/data.md`](data.md#known-biases-and-leakage-risks), and where it
leaves the model:

| Risk | Effect on this model |
|---|---|
| **No modern ham.** All ham is from 2002. | The largest one. Every false-positive figure is measured on 2002 mailing-list and personal mail; modern newsletters, transactional mail and SaaS notifications are untested. Treat the FPR as optimistic. |
| **URL length as an era marker** (found by the leakage check) | `url_max_length` was the strongest v0.1.0 feature: short URLs in 2002 ham, long ones in phishing. Removed in v0.1.1 ([ADR 0006](adr/0006-drop-url-max-length.md)) after an ablation: same test recall, 97.6% instead of 95.3% on unseen campaigns, 2.1 points more spam flagged. Other URL features may still carry era signal; the drift report watches them (#38). |
| **Ham is mostly mailing lists** (found by scoring a made-up personal email through the API) | The `Return-Path` domain differs from `From` in 61% of training ham (a list's bounce address) and 19% of phishing, so a *consistent* return path pushes towards phishing. A plain three-line personal email scores 0.67, above the 1% threshold. 75% of training ham carries mailing-list headers; on test, ham without them is flagged 2.7% of the time (4 of 149) against 0.4% for list mail (2 of 473), so the headline 1% FPR is mostly a mailing-list FPR. Short personal mail is the most likely false alarm in production; tracked in #73. |
| **Era gap** between ham (2002) and phishing (2004-2025) | No date or HTML features; text read the same way for plain text and HTML; digits scrubbed; era leakage audited. URL features still encode the era (audit AUC 0.950), measured by the temporal check. |
| **Source artefacts in headers** | Five allowlisted headers; corpus domains treated as unknown; two collection artefacts removed after an audit (`Received` hops, recipient count). None in the top 20 features. |
| **Authentication headers** exist only in recent phishing | SPF / DKIM / DMARC are parsed but not used by the model; the agent uses them on live mail. The model therefore ignores a strong real-world signal. |
| **Unrealistic base rate** (57% phishing) | Recall and FPR are reported because they do not depend on it; precision and probabilities are converted to real base rates above. |
| **Duplicates** (21.3% of messages) | Splits and intervals are grouped by cluster. Near-duplicates beyond URL and number changes form separate clusters, so overlap is underestimated. |
| **Broken `Date` headers** | Only the presence of a date is a feature, never its value. |
| **Malformed messages** | The parser never raises (property-tested); defects are counted as a feature. |
| **Image- or attachment-only phishing** (3.2% of phishing) | Almost invisible to the model: no text, and attachment content is never read. Left to the agent. |
| **Label noise** | Nazario's corpus is one inbox and holds spam and newsletters: 1% to 3% of the phishing label in a random sample, but 4 of the 15 test misses. It barely moves recall (about +0.3 points once corrected on both sides). |
| **English only** | Lexicon cues are English; non-English phishing relies on TF-IDF and the structured features. All 78 test phishing messages with little English were caught, but 3 non-English misses appear in the error taxonomy. |
| **Small ham test set** (622) | One ham message is 0.16% of the FPR: 0.1% FPR can be neither shown nor ruled out (that takes about 3,000 ham), and the 1% FPR interval is wide. |

## Privacy

- No message, address or URL from the corpora is in the repository, the reports or this card;
  reports hold metrics only, error analysis holds categories and counts.
- Fitted TF-IDF vocabularies contain n-grams of corpus text. Models live in MLflow under
  `mlruns/` (gitignored) and are not published.
- Text is scrubbed before TF-IDF: addresses, URLs, digits and the recipient's own name never enter
  the vocabulary.
- The serving layer (Phase 2) must not log email content.

## Reproducing

```bash
make setup && make data && make dataset && make train && make evaluate
```

Same corpora (SHA-256 pinned), same seeds and same split hash (`bcb685de8cbe...`) give the same
validation metrics and a byte-identical `reports/evaluation.md`.
