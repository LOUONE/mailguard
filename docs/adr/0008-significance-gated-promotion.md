# 0008. Promote a retrained model only on a significant gain, from a plain-Python flow

- Status: Accepted
- Date: 2026-10-29
- Issue: #39

## Context

Retraining is easy to automate; deciding whether the new model should replace the one in
production is the hard part. Phase 1 measured how noisy this corpus is: one test ham is 0.16% of
the FPR, the paired McNemar test is optimistic because of duplicates, and the ADR 0006 update
showed a temporal "gain" of 2.3 points turning into a loss when 850 spam messages changed split.
A rule that promotes on any improvement would churn production on luck.

Two further constraints:

- **Fair comparison.** Splits move when data is added or when the parser changes
  ([data](../data.md#reproducibility)), so a message in today's test split may have trained the
  production model, which would then look better than it is.
- **Orchestrator.** The roadmap planned Prefect. Prefect 3.8.8 installs, but running a flow
  starts its embedded API server, which fails with FastAPI 0.162 (`Body() got an unexpected
  keyword argument 'min_items'`); the scoring API needs that FastAPI.

## Options

### Promotion rule

- **A. Promote when the candidate's point estimate is better.** Simple; promotes noise half the
  time when the two models are equivalent.
- **B. Promote when the paired bootstrap interval of the recall gain is above zero.** Accounts for
  duplicates; alone, says nothing about false alarms.
- **C. B, plus McNemar agreement, plus a non-inferiority margin on ham FPR.** Two independent
  views of the gain, and a guard on the cost that matters most to users.

### Orchestration

- **D. Prefect in a separate environment or image.** Keeps the planned tool, doubles what must
  be locked, built and patched, for one flow.
- **E. Pin FastAPI to a version Prefect's server accepts.** Couples the API to the orchestrator.
- **F. Plain functions, one per step, runnable by any scheduler.** No new dependency; loses
  Prefect's UI, retries and run history.

## Decision

**C and F.** `mailguard.pipeline.promotion.decide` promotes only when, on messages neither model
has seen and at the thresholds each version serves (recorded on the version at training time):

1. the 95% cluster-bootstrap interval of `recall(candidate) - recall(production)` is above zero;
2. McNemar finds more messages only the candidate gets right, with p < 0.05;
3. the upper bound of the ham FPR difference is at most +0.5 points.

`mailguard.pipeline.retrain` runs dataset, training, comparison and decision as plain functions.
Every training run records the ids of the messages it saw (`seen_ids.json`); the comparison
removes them from the evaluation set. A production version without that record is compared only
if it was trained on the same split, and otherwise kept, with the reason. The decision is a tag
on the candidate version (`promotion`) and a JSON file, and is logged as a warning when
production is kept.

## Consequences

- Most retrainings will keep production: on this corpus, gains of under a point are rarely
  significant. That is the intended behaviour; the missed improvement costs one cycle.
- The margin of half a point of ham FPR is a product choice: an operator who cares more about
  false alarms lowers it.
- MLflow keeps the history Prefect would have kept (runs, versions, tags). Retries and
  scheduling are the scheduler's job: cron, a Kubernetes CronJob, or the drift job when it
  flags the traffic (#40).
- Revisit the orchestrator if flows multiply or if Prefect's server supports current FastAPI.
