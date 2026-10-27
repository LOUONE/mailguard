# 0001. Binary target: phishing against ham and spam

- Status: Accepted
- Date: 2026-10-27
- Issue: #21

## Context

The corpora carry three labels, taken from the file a message comes from (see
[`docs/data.md`](../data.md#labels)): 4,150 `ham`, 1,896 `spam` and 8,038 `phishing`. mailguard
outputs one score, the probability that a message is phishing, so a binary target has to be
defined before any label can be computed.

Three facts from the EDA weigh on the choice:

- `spam` is 13.5% of the corpus. It shares surface traits with phishing (bulk sending, links,
  calls to action) without trying to steal credentials.
- A few 2002 spam messages are phishing in all but name. Whatever the option, this is label noise
  on one side or the other.
- All ham and spam date from 2002, all phishing from 2004 onwards. No choice of target fixes this
  era gap; it is handled by the split and the features (no date, no HTML flag, no SPF/DKIM).

In deployment, mailguard sits next to a spam filter, not in place of one. The costly error is a
legitimate message flagged as phishing. Spam flagged as phishing is an error too, but a cheap one:
the user did not want that message anyway.

## Options

### A. Phishing against ham only (spam dropped)

Cleanest labels: each class is what it says. But the model never sees unsolicited bulk mail. In
production, spam that gets past the spam filter would be scored on traits the model never had to
separate from phishing, and it would likely score high. That false-positive rate could not even be
measured. It also throws away 13.5% of the data.

### B. Phishing against ham and spam

The model must learn what is specific to phishing (impersonation, credential requests, mismatched
links) rather than "unsolicited". The spam false-positive rate becomes measurable. Cost: the
phishing-like 2002 spam is a negative, which slightly caps the measurable recall and precision.

### C. Three classes

Richest output, but the serving contract becomes three probabilities, calibration has to hold per
class, and the Phase 3 grey zone has to be defined in two dimensions. The headline metric (recall
at a fixed false-positive rate) would still need a one-against-rest reduction. Classifying spam is
not the goal of the project.

## Decision

**Option B.** `target = 1` for `phishing`, `target = 0` for `ham` and `spam`. The three-way `label`
stays in the dataset so every metric can be broken down by it.

## Consequences

- **False-positive rate is measured on ham.** "Recall at 1% FPR" means 1% of *ham* flagged.
  Operating thresholds are chosen on validation ham. The rate of spam flagged as phishing is
  reported next to it as a secondary metric, never mixed into the headline FPR.
- **PR-AUC depends on the 57% phishing prior** of the corpus, far above a real mailbox. It is
  reported for comparing models on this test set only, never as an expected production figure.
- **Small negative set.** With a 70 / 15 / 15 split, the test set holds about 620 ham. 1% FPR is
  about 6 messages and 0.1% is less than one, so both are reported with intervals and 1% is the
  headline (see #27).
- **Splits are stratified on the three-way label**, so validation and test both contain spam.
- **Serving contract**: one `phishing_probability`, no spam output. The API documentation says
  mailguard is not a spam filter.
- **Label noise** on phishing-like spam is expected to show up among the false positives; the error
  analysis (#29) counts it as its own category.
- Revisit if a licensed corpus of modern ham or labelled modern spam becomes available, or if the
  spam false-positive rate turns out to dominate the errors.
