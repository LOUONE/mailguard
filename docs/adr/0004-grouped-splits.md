# 0004. Split by duplicate cluster before any feature, with a separate temporal check

- Status: Accepted
- Date: 2026-10-27
- Issues: #22, #27 (fix in PR #63)

## Context

The EDA found that 21.3% of messages repeat an earlier body once case, Unicode, URLs and numbers
are normalised (41.5% of 2004-2007 phishing; largest campaign: 71 copies). It also found that
phishing never overlaps ham in time: all ham is from 2002, phishing from 2004-2007 and 2015-2025.

The roadmap planned a global temporal split and a split after the first baselines. Neither holds:
a temporal split would put no ham in test, and any feature fitted before the split (a TF-IDF
vocabulary, for one) would see test messages.

## Options

### A. Random split of messages

Standard, but copies of a campaign land on both sides, and the model is graded on campaigns it has
memorised. Scores would be inflated by an amount nobody could measure.

### B. Global temporal split

The most realistic for drift, but impossible here: there is no ham after 2002.

### C. Split grouped by duplicate cluster, plus a temporal check on phishing only

Every cluster (normalised body fingerprint) sits in one split. Drift is measured separately: train
on phishing up to 2019, test on 2020-2025 phishing, with ham kept on its main split.

## Decision

**Option C**, in [`data/dataset.py`](../../src/mailguard/data/dataset.py), built once by
`make dataset` before any feature exists:

- 70 / 15 / 15, stratified by label and phishing period, seed fixed;
- clusters shuffled, then placed largest first into the split that is **least full relative to its
  target**;
- the temporal assignment keeps any cluster with phishing up to 2019 on the training side (6
  clusters, 22 messages, cross the cut);
- a manifest records counts and a hash of every assignment; training runs record it.

## Consequences

- Validation and test only contain campaigns the model has never seen, and intervals are
  bootstrapped over clusters, not messages (#27).
- **The placement rule matters.** The first version chose the largest *absolute* shortfall, which
  sent all 1,254 multi-message clusters to train (the biggest target) and left validation and test
  with unique messages only. Proportions were perfect and every test passed; the first evaluation
  report showed as many clusters as messages in test. A regression test and an integration check
  on the share of duplicates per split now guard it (29.8-29.9% in every split).
- Near-duplicates that differ by more than URLs and numbers still form separate clusters. The
  fingerprint is a lower bound on campaign overlap.
- Changing the split changes the assignment hash, and `make evaluate` refuses models trained on
  another one.
