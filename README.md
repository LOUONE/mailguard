# mailguard

Real-time phishing detection with an LLM investigation agent, built for privacy.

[![CI](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml/badge.svg)](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](pyproject.toml)

> **Status: work in progress.** Phases 1 and 2 are complete. The model is evaluated with intervals and documented ([evaluation report](reports/evaluation.md), [model card](docs/model_card.md)). It runs as a monitored service that starts with one command, detects drift, and retrains itself behind a statistical gate ([architecture](docs/architecture.md)). Phase 3, the LLM investigation agent, is next. Expected final version: `v1.0.0`.

## Results

Test split, 2,113 messages in 1,670 duplicate clusters, never used for any choice. Thresholds are
fixed on validation ham; brackets are 95% intervals from a bootstrap over duplicate clusters.
Full report: [reports/evaluation.md](reports/evaluation.md). What the model is for, where it fails
and why LightGBM was chosen: [model card](docs/model_card.md), [ADR 0005](docs/adr/0005-model-choice.md),
[ADR 0006](docs/adr/0006-drop-url-max-length.md).

| Baseline | PR-AUC | Recall at 1% ham FPR | Ham FPR measured on test | Recall on 2020-2025 phishing, trained up to 2019 |
|---|---|---|---|---|
| Logistic regression, text only | 0.9974 [0.9953, 0.9988] | 97.9% [96.8, 98.9] | 1.1% | **96.8% [96.0, 97.6]** |
| LightGBM, all feature groups but `url_max_length` | 0.9990 [0.9982, 0.9997] | **99.0% [98.1, 99.6]** | 1.1% | 96.1% [95.2, 97.0] |

How to read it:

- Header, URL and attachment features add 1.1 points of recall at 1% FPR (paired interval
  [0.4, 1.8]), and 4.0 points at the strict threshold [2.7, 5.4].
- **0.1% FPR can be neither shown nor ruled out on this data.** One test ham is 0.16% of the
  FPR; proving 0.1% would take about 3,000 ham with none flagged. The strictest threshold
  (no validation ham flagged) still flags 0.8% of test ham (5 messages).
- **New campaigns cost recall, and the text-only model holds up better.** Trained only on
  phishing up to 2019, LightGBM loses 2.7 points on 2020-2025 phishing, the logistic regression
  0.9. This comparison is also the least stable one: its threshold rests on the ham of one
  validation split, and it moved by over a point when 850 spam messages changed split.
- **The strongest v0.1.0 feature was an era marker, and it was removed.** `url_max_length`
  separated 2002 ham from later phishing ([ADR 0006](docs/adr/0006-drop-url-max-length.md),
  with an update on what the evidence turned out to show).
- **In a real mailbox, false alarms would be about as frequent as caught phishing.** Out of
  1,000 messages with 1% phishing, LightGBM would catch 9.9 of 10 phishing messages and flag
  11 legitimate ones [2, 27]: the interval rests on seven flagged test ham. Probabilities are
  calibrated (ECE 0.7% on test) and converted to a real base rate with Bayes' rule, so a "90%"
  in the corpus means 6% in such an inbox.
- **The Python version is part of the data.** A patch release of CPython decoded 4 spam messages
  differently and moved 850 of them to another split; the exact version is now pinned
  ([docs/data.md](docs/data.md#reproducibility)).
- **Label noise does not change these figures.** 1-3% of the phishing label is wrong in a
  random sample; corrected on both sides, recall moves by about 0.3 points.
- Scores are near the ceiling because all ham dates from 2002 and phishing from 2004 onwards.
  Treat these as an upper bound for a real inbox ([docs/data.md](docs/data.md)).

## What this project aims to prove

1. **Ship ML to production.** A real-time API, containerised, monitored, with automated retraining.
2. **Statistical rigour.** Temporal splits, confidence intervals, calibrated probabilities, honest error analysis — not just "99% accuracy".
3. **Agentic AI.** An open-source LLM agent that investigates ambiguous emails with typed tools and produces a human-readable verdict.
4. **Attacker mindset.** Red-team the model (homoglyphs, LLM-rewritten phishing) and measure robustness before and after hardening.
5. **Clean Python.** Typed, tested, packaged, CI green.

## Roadmap

Each phase is a [GitHub milestone](https://github.com/LOUONE/mailguard/milestones) with one issue
per task, ordered by dependency, and ends with a tagged release.

| Phase | Deliverable | Release | Status |
|---|---|---|---|
| 0 — Foundations | Repo scaffolded, CI green, `main` protected | `v0.0.1` | Released 2026-10-18 |
| 1 — Data + classical model | Parsed corpora, grouped splits, features, LightGBM baseline, rigorous eval, SHAP, MLflow | `v0.1.0` | Released 2026-10-28 |
| 2 — Production | FastAPI service, Docker, Prometheus/Grafana, drift reports, gated auto-retraining | `v0.2.0` | Released 2026-10-30 |
| 3 — Agent | LLM agent (Ollama, OpenAI-compatible fallback), typed tools, prompt-injection defence | `v0.3.0` | Next, target 2027-01-10 |
| 4 — Adversarial | Attack suite + hardening, before/after metrics | `v1.0.0` | Target 2027-01-24 |

Current board: [mailguard roadmap](https://github.com/users/LOUONE/projects/1).

## Quickstart

Requires Python 3.14+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/LOUONE/mailguard.git
cd mailguard
make setup   # creates the venv, installs deps, installs pre-commit hook
make test    # runs the test suite with coverage
make lint    # ruff + mypy strict
make data    # downloads the corpora (~147 MB) into data/raw/, checksum-verified
make dataset # parses them into data/processed/ with grouped train/val/test splits
make train   # trains both baselines (~3 min) and registers them in a local MLflow
make evaluate  # test-split report with intervals into reports/ (~3 min)
make promote # gives the latest LightGBM the `production` alias
make serve   # scoring API on http://127.0.0.1:8000 (interactive docs at /docs)
make retrain # trains a candidate, promotes it only on a significant gain (ADR 0008)
make simulate-drift  # holds out 2024-2025 phishing and replays drift and retraining
make eda     # re-runs the exploratory notebook on the downloaded data
```

See the `Makefile` for the full list of targets (`make help`).

### The whole system in one command

Requires Docker only (no Python, no data on the host):

```bash
make demo        # or: docker compose up --build -d
```

| Service | URL | What it is |
|---|---|---|
| API | http://127.0.0.1:8000/docs | The scoring API and its interactive documentation |
| Grafana | http://127.0.0.1:3000 | The `mailguard` dashboard: traffic, latency, predicted phishing rate, scores |
| Prometheus | http://127.0.0.1:9090 | The metrics behind it ([docs/monitoring.md](docs/monitoring.md)) |
| MLflow | http://127.0.0.1:5000 | Runs, registered models and the `production` alias |

The first start downloads the corpora and trains the models inside a container (about 5 minutes,
`docker compose logs -f bootstrap`); later starts take seconds. A `traffic` service sends
made-up emails, so the dashboard fills on its own. Every port listens on 127.0.0.1 only, the API
runs as a non-root user and mounts the model store read-only. `make demo-down` stops it all.

**What the dashboard shows is a real finding, not a demo bug.** About 80% of the demo traffic is
short, made-up personal mail, and the model labels nearly all of it phishing. 75% of the training
ham comes from mailing lists, and the model has partly learnt that mail which is not from a list
is suspicious: on the test split, ham without list headers is flagged 3.4% of the time, against
0.4% for list mail. The predicted phishing rate panel is exactly where an operator would see such
a shift. Measured and tracked in [#73](https://github.com/LOUONE/mailguard/issues/73).

### Latency

On real test-split emails, one process answers in **18 ms at the median and 30 ms at p95**, and
four processes sustain about 300 requests per second on an 8-core desktop
([docs/performance.md](docs/performance.md): method, profile and the 2.4x speed-up it led to).

### Scoring an email

```bash
curl -s http://127.0.0.1:8000/v1/predict -H 'content-type: application/json' \
  -d '{"raw": "From: PayPal <x@evil.example>\nSubject: verify your account\n\nConfirm your password now."}'
```

```json
{
  "schema_version": "v1",
  "label": "phishing",
  "score": 0.9993,
  "probability": 0.9184,
  "base_rate": 0.01,
  "model": {"name": "mailguard-lightgbm", "version": "6", "operating_point": "1pct", "threshold": 0.3622},
  "reasons": [
    {"feature": "headers__hdr_display_name_brand_mismatch", "group": "headers", "contribution": 1.91, "direction": "phishing"},
    {"feature": "lexicon__subject_credential_cues", "group": "lexicon", "contribution": 1.63, "direction": "phishing"},
    {"feature": "lexicon__body_credential_cues", "group": "lexicon", "contribution": 0.94, "direction": "phishing"},
    {"feature": "urls__url_count", "group": "urls", "contribution": -0.49, "direction": "legitimate"},
    {"feature": "text__word:list", "group": "text", "contribution": 0.47, "direction": "phishing"}
  ]
}
```

(Rounded; the brand named in the display name does not match the sender's domain, and the
subject and body ask for credentials.)

`score` is the calibrated probability at the corpus base rate, and `label` compares it with the
threshold recorded on the model version. `probability` converts the score to the share of
phishing in the mailbox (`base_rate`, 1% by default, or per request), so a client can show it to
a person. The contract is versioned (`/v1`, `schema_version`) and published as
[`docs/api/openapi.json`](docs/api/openapi.json).

## Repository layout

```
src/mailguard/
├── data/           corpus manifest, checksummed download, streaming loader, dedup
├── parsing/        raw email -> validated Email model
├── features/       header, URL, text features
├── models/         training, evaluation, registry
├── serving/        FastAPI app, schemas, limits, errors, access log
├── agent/          LLM tools + investigation loop
├── monitoring/     metrics, JSON logs, feature recording, drift (PSI, KS)
├── pipeline/       retraining flow, promotion gate, drift simulation
└── adversarial/    attack suite + robustness eval
```

## Data

Trained and evaluated on two public corpora: ~6,000 ham and spam messages from the
[SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) and ~8,000 phishing
messages (2004-2025) from Jose Nazario's [phishing corpus](https://monkey.org/~jose/phishing/)
(CC BY 4.0). No message is redistributed in this repo.

[docs/data.md](docs/data.md) covers licences, class balance, personal data, and the leakage risks
measured before any modelling (source artefacts in headers, a 2002-only ham set, SPF/DKIM headers
that exist only in recent phishing). [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) adds class
balance, dates, structure and duplicate rates, with a TL;DR of what each finding changes.

## Design decisions

Structural choices are recorded as short [architecture decision records](docs/adr/README.md):
the binary target ([0001](docs/adr/0001-binary-target.md)), the parser design
([0002](docs/adr/0002-parser-design.md)), model serialisation without pickle
([0003](docs/adr/0003-model-serialisation.md)), the grouped splits
([0004](docs/adr/0004-grouped-splits.md)), the model choice
([0005](docs/adr/0005-model-choice.md)), dropping an era-marker feature
([0006](docs/adr/0006-drop-url-max-length.md), with an update on evidence that did not hold),
drift detection without Evidently ([0007](docs/adr/0007-drift-without-evidently.md)) and the
significance-gated promotion of retrained models ([0008](docs/adr/0008-significance-gated-promotion.md)).

How it fits together: [architecture](docs/architecture.md). How fast it is and why:
[performance](docs/performance.md). What it watches: [monitoring](docs/monitoring.md). What a
held-out year of phishing did to it: [drift simulation](docs/drift_simulation.md).

## License

MIT — see [LICENSE](LICENSE).
