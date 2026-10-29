# Monitoring

What the scoring API tells an operator, and what it never tells anyone. Code:
[`monitoring/metrics.py`](../src/mailguard/monitoring/metrics.py),
[`monitoring/logs.py`](../src/mailguard/monitoring/logs.py),
[`serving/access.py`](../src/mailguard/serving/access.py).

## Prometheus metrics

Scraped from `GET /metrics` (text exposition format, not part of the OpenAPI contract). Every
label takes a bounded set of values: route templates, methods, status codes, two labels. Nothing
derived from an email is ever a label.

| Metric | Type | Labels | What it answers |
|---|---|---|---|
| `mailguard_requests_total` | counter | `route`, `method`, `status` | Traffic and error rate. `route` is the template (`/v1/predict`) or `unmatched`. |
| `mailguard_request_duration_seconds` | histogram | `route` | End-to-end latency, refused requests included. |
| `mailguard_scoring_duration_seconds` | histogram | | Parsing, features and model for one email: the part a faster model would change. |
| `mailguard_email_bytes` | histogram | | Size of scored emails. A shift here often explains a latency shift. |
| `mailguard_predictions_total` | counter | `label` | Decisions. `phishing / all` is the predicted phishing rate. |
| `mailguard_score` | histogram | | Calibrated score at the training base rate, in 0.05 buckets. |
| `mailguard_model_info` | gauge | `name`, `version`, `operating_point` | Always 1; which model is serving. |
| `mailguard_model_threshold` | gauge | | Score above which mail is labelled phishing. |
| `mailguard_model_ready` | gauge | | 1 when a model is loaded, 0 when `/health` answers 503. |

The two model metrics are the early warnings. The predicted phishing rate and the score
histogram move when traffic drifts (a new campaign, a new kind of newsletter) or when a model
misbehaves, long before anyone reports a missed phishing email. They cannot say *which* is
happening; the drift report (#38) compares feature distributions for that.

Useful queries:

```promql
# predicted phishing rate over 5 minutes
sum(rate(mailguard_predictions_total{label="phishing"}[5m]))
  / sum(rate(mailguard_predictions_total[5m]))

# p95 latency of the scoring endpoint
histogram_quantile(0.95, sum by (le) (rate(mailguard_request_duration_seconds_bucket{route="/v1/predict"}[5m])))

# share of scores in the uncertain band that Phase 3 sends to the agent
sum(rate(mailguard_score_bucket{le="0.9"}[5m])) - sum(rate(mailguard_score_bucket{le="0.1"}[5m]))
```

## Drift

The metrics above say *that* the decisions moved; the drift report says *which inputs* moved.
Method and thresholds in [`monitoring/drift.py`](../src/mailguard/monitoring/drift.py); why not
Evidently in [ADR 0007](adr/0007-drift-without-evidently.md).

1. **Recording.** With `MAILGUARD_FEATURE_LOG_DIR` set, the API appends one JSON line per scored
   email: time, model version, score, label and the 38 structured features (header flags,
   attachment, URL and wording counts). Never text, addresses, URLs, TF-IDF columns or request
   ids; a test sends a canary through the API and searches the files for it.
2. **Reference.** The served model's features and scores on the validation split: data it was
   not trained on, drawn like its training data. Cached per model version.
3. **Comparison.** For each feature and for the score, the PSI and the two-sample
   Kolmogorov-Smirnov statistic over the last 2,000 emails scored by the same model version
   (at least 200). A feature drifts above PSI 0.25; the traffic drifts when half the features
   do, or when the score does.
4. **Output.** `summary.json` (numbers only) and `report.html` in `MAILGUARD_DRIFT_DIR`. The API
   reads the summary on each scrape and exposes it:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `mailguard_drift_detected` | gauge | | 1 when the last report flagged the traffic |
| `mailguard_drift_share` | gauge | | Share of input features above the PSI threshold |
| `mailguard_feature_psi` | gauge | `feature` | PSI of each recorded feature, and of `score` |

In the compose stack the `drift` service runs the job every minute. Its first verdict on the
demo traffic is "drift", which is correct: made-up short personal mail does not look like the
2002 mailing-list ham and 2004-2025 phishing of the validation split.

PSI reading, from credit scoring: below 0.1 stable, 0.1 to 0.25 a moderate shift worth a look,
above 0.25 a significant shift. For two normal distributions d standard deviations apart, PSI is
about d²: 0.25 is half a standard deviation.

## Logs

One JSON object per line on stdout, from the API and from libraries alike. Fields:

| Field | Where | Meaning |
|---|---|---|
| `timestamp`, `level`, `logger`, `event` | every line | UTC ISO timestamp; `event` is a fixed name (`request`, `model_loaded`, ...) |
| `request_id` | requests | Generated, or the caller's own if it matches `[A-Za-z0-9._-]{1,64}`; also the `X-Request-ID` header |
| `method`, `path`, `status`, `duration_ms` | `request` | `path` is the route template, never the raw URL |
| `email_bytes`, `scoring_ms`, `label`, `score`, `model_version` | `request` on `/v1/predict` | Sizes, timings and the decision |
| `error_type` | `unhandled_error`, `model_not_loaded` | The exception *type*; its message can quote the email |
| `model`, `detail` | `model_loaded`, `model_not_loaded` | Registry name and why loading failed |

**No content, by construction.** A field whose name is not in `ALLOWED_FIELDS` is replaced with
`[redacted]` before rendering. Exception messages and tracebacks are never logged: the outermost
middleware turns any unexpected exception into a 500 and logs its type, so Starlette never
re-raises it to the server's own error log. A test sends a canary string through every path
(display name, addresses, subject, body, URL, link text, attachment name; valid, invalid,
oversized and failing requests) and searches the whole log output for it.

`MAILGUARD_LOG_JSON=false` switches to coloured console lines for local development.
