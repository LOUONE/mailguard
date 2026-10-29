# Performance

How fast the scoring API answers, where the time went, and what changed it. Raw Locust
summaries are in [`reports/perf/`](../reports/perf/).

## Headline

One process, real emails from the test split (median 5 KB, p99 100 KB):

| | Before | After |
|---|---|---|
| p50 latency, one client at a time | 43 ms | **18 ms** |
| p95 latency, one client at a time | 62 ms | **30 ms** |
| Throughput, 8 concurrent clients | 21 req/s | **96 req/s** |

With 4 uvicorn worker processes and 16 concurrent clients: **294 req/s**, p50 48 ms, p95 110 ms,
no error in 13,012 requests.

Hardware: AMD Ryzen 7 9700X (8 cores, 16 threads), 32 GB RAM, Windows 11, CPython 3.14.8. The
load generator ran on the same machine, so the 4-worker figure is a lower bound.

## Method

```bash
make serve                      # one process; or add --workers 4 to the uvicorn command
MAILGUARD_LOADTEST_CORPUS=1 uv run --group perf locust -f scripts/locustfile.py --headless \
    -u 8 -r 8 -t 45s --host http://127.0.0.1:8000 --csv reports/perf/run
```

- **Payloads.** `MAILGUARD_LOADTEST_CORPUS=1` replays the 2,113 messages of the test split, read
  locally and never written anywhere. Without it, the made-up messages of `demo_traffic.py`
  (a few hundred bytes) give a floor for latency on a machine without the corpora.
- **Closed loop.** Each simulated user sends its next request as soon as the previous one
  returns, so throughput is what the service can absorb and latency what it costs at that load.
- **Profile.** `py-spy record --pid <server pid> --rate 200 --format speedscope` during a run,
  then inclusive time per function over the samples inside a request.

## What the profile showed

The first measurement had a telling shape: one client got 21.8 req/s at 43 ms, and eight
clients got 21.0 req/s at 370 ms. Concurrency added no throughput, only queueing. That is the
GIL: feature extraction is Python, so one thread runs at a time whatever the thread pool size.
The question was then what those 43 ms were spent on. Inclusive time inside a request
(6,080 samples):

| Where | Share |
|---|---|
| `explain()`, the five reasons | **81%** |
| - building the names of the 50,000 TF-IDF columns | **38%** |
| - computing the features a second time | 25% |
| - TreeSHAP contributions | 18% |
| `predict_proba()`, the score itself | 17% |

The intuition before profiling was "features are computed twice". True, but the largest single
cost was elsewhere: `get_feature_names_out()` rebuilt tens of thousands of n-gram names on every
request, to label five of them.

## What changed

1. **Feature names are computed once**, when the model is loaded (`ServedModel.names`).
2. **One pass for the score and its reasons.** TreeSHAP contributions are additive: bias plus
   contributions is the model's margin, and the logistic of the margin is its probability. So
   one feature transform and one LightGBM call give both. A test checks that the score equals
   the calibrated pipeline's to 1e-12 and that the reasons equal `explain()`.
3. **Top reasons from the sparse row**, instead of densifying 50,000 columns to sort them.

One subtlety the equivalence test caught: the logistic regression computes its margin in
float32 (scikit-learn casts its coefficients to the dtype of the TF-IDF matrix), so float64
contributions do not reproduce its score to the last digit. For that model the score comes from
its own `predict_proba`, a dot product; the contributions are now summed in float64.

After the change, eight clients get 4.6 times the throughput: LightGBM releases the GIL while
it predicts, and with less Python per request the threads overlap.

## Scaling

The API holds no state between requests, so it scales with processes: uvicorn's `--workers`
(or the `WEB_CONCURRENCY` environment variable) runs one model per process. Each process holds
its own copy of the model and its libraries: 370 MB resident after a load test, measured on
Windows. The compose stack runs one worker; a deployment sizes the count to its cores and memory.
