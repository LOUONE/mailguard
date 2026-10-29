# 0007. Detect drift with PSI and KS in our own code, not Evidently

- Status: Accepted
- Date: 2026-10-29
- Issue: #38

## Context

The roadmap planned Evidently for the drift report: the API records feature vectors, and a job
compares a recent window with a reference. When the issue was started, `evidently` 0.7.23 (the
latest release uv could resolve) installed but failed at import with pandas 3.1, the version the
project uses (`cannot import name 'infer_dtype_from_object' from 'pandas.core.dtypes.common'`).

The same install showed what it would bring: 26 direct dependencies, including plotly, nltk,
statsmodels, litestar, a second web server, and `iterative-telemetry`, a library that reports
usage. For the statistics the report needs (two per numeric feature), that is most of a data
science platform, and a telemetry client in a project whose selling point is privacy.

## Options

### A. Evidently, with pandas pinned below 3

Recognisable to a reviewer, with polished HTML reports. But it downgrades the data stack of the
whole project for one monitoring job, adds a large attack and maintenance surface, and ships a
telemetry dependency that would have to be disabled and audited.

### B. Evidently in a separate image or virtual environment

Keeps pandas 3 for the rest. Doubles the number of environments to lock, build and patch, for a
job that reads a few thousand rows of numbers.

### C. Our own implementation with scipy

PSI and the two-sample Kolmogorov-Smirnov test are a few lines each on top of numpy and scipy,
both already dependencies. The thresholds are explicit, the binning can handle the small counts
most of our features are, and every line is unit-tested against distributions whose answer is
known (PSI of two unit normals d apart is about d²).

## Decision

**Option C.** `mailguard.monitoring.drift` computes PSI and KS per feature;
`mailguard.monitoring.drift_job` builds the reference (validation split, cached per model
version), reads the recorded traffic, and writes `summary.json` (exposed as Prometheus metrics)
and `report.html`.

## Consequences

- No new dependency, and no telemetry anywhere in the stack.
- The report is a plain table, not Evidently's interactive dashboards; the time series and the
  per-feature PSI live in Grafana instead.
- Fewer statistics than Evidently offers (no Wasserstein, Jensen-Shannon or chi-squared). PSI
  is the one credit-risk and fraud teams use, and the others can be added beside it if needed.
- Revisit if Evidently supports pandas 3 without telemetry, or if the project needs text or
  embedding drift, which this implementation does not attempt.
