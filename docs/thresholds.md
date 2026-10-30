# Operating thresholds and duplicate clusters (#85)

The 1% operating threshold is the (k+1)-th highest score among validation ham, with
k = floor(0.01 × n). The multi-seed experiment of #73 found a split where it jumped to 0.995 and
recall fell to 88.6%: nine copies of one legitimate email landed in validation, scored high, and
filled the top of the ranking by themselves. Splits and bootstrap intervals treat duplicates by
cluster; the threshold did not. This page measures the obvious fix and why it was not adopted.
Numbers: [`reports/experiments/thresholds.md`](../reports/experiments/thresholds.md) (served
LightGBM, eight split seeds, `scripts/experiment_thresholds.py`).

## Three rules for the same 1% target

| Rule (mean over 8 seeds) | Recall | Ham FPR delivered, per message | Threshold range |
|---|---|---|---|
| Per validation ham message (served) | 97.8% (88.6 to 99.5) | 1.09% (0.0 to 2.1) | 0.23 to 0.99 |
| Per duplicate cluster, each counted once | 99.4% (99.0 to 99.75) | **1.75%** (1.1 to 2.7) | 0.06 to 0.22 |
| On ham that is not list mail | 97.2% (88.6 to 99.4) | 0.68% (0.0 to 1.1) | 0.26 to 0.99 |

- **Counting clusters once removes the failure**: recall never falls below 99.0%, and the
  threshold moves within a narrow band on every split.
- **But it changes the promise.** It allows 1% of ham *campaigns*, and a campaign of nine copies
  is nine emails in someone's inbox. The FPR it delivers per message is above 1% on all eight
  seeds (+0.66 points on average): a user would see more false alarms than the operating point
  says.
- **The per-message rule keeps its promise and is fragile**: one split in eight lost ten points
  of recall, and a second (threshold 0.70) lost one.

## Decision

The served rule stays per message: the FPR an operator sets is the FPR users get. The served
split (seed 20261027) is not one of the fragile ones (threshold 0.25 per message, 0.17 per
cluster).

The fragility is now known and measured, and it gives a concrete check for any future training
data: if the copies of one ham cluster sit at the top of the validation ranking, the threshold is
set by one campaign. A larger, deduplicated validation set of ham, or a threshold averaged over
resamples of clusters, would address it without changing the promise; both need more modern
ham than the corpus has, which is the same conclusion as [the mailing-list study](mailing_lists.md).
