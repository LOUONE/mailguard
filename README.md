# mailguard

Real-time phishing detection with an LLM investigation agent, built for privacy.

[![CI](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml/badge.svg)](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](pyproject.toml)

> **Status: work in progress.** Phase 1 is complete: data pipeline, grouped splits, leakage-audited features, two tracked and calibrated baselines, a test-set [evaluation report](reports/evaluation.md) and a [model card](docs/model_card.md). Phase 2 (serving in production) is next. Expected final version: `v1.0.0`.

## Results

Test split, 2,113 messages in 1,670 duplicate clusters, never used for any choice. Thresholds are
fixed on validation ham; brackets are 95% intervals from a bootstrap over duplicate clusters.
Full report: [reports/evaluation.md](reports/evaluation.md). What the model is for, where it fails
and why LightGBM was chosen: [model card](docs/model_card.md), [ADR 0005](docs/adr/0005-model-choice.md).

| Baseline | PR-AUC | Recall at 1% ham FPR | Ham FPR measured on test | Recall on 2020-2025 phishing, trained up to 2019 |
|---|---|---|---|---|
| Logistic regression, text only | 0.9978 [0.9960, 0.9991] | 97.9% [96.7, 98.9] | 1.1% | 96.5% [95.6, 97.4] |
| LightGBM, all feature groups | 0.9993 [0.9985, 0.9998] | **98.8% [97.8, 99.5]** | 0.8% | 95.3% [94.1, 96.3] |

How to read it:

- Header, URL and attachment features add 0.8 points of recall at 1% FPR (paired interval
  [0.2, 1.7]).
- **0.1% FPR can be neither shown nor ruled out on this data.** One test ham is 0.16% of the
  FPR; proving 0.1% would take about 3,000 ham with none flagged. The strictest threshold
  (no validation ham flagged) still flags 0.6-0.8% of test ham (4-5 messages).
- **New campaigns cost recall, and the strongest feature is to blame.** Trained only on phishing
  up to 2019, LightGBM loses 3.5 points on 2020-2025 phishing. Retrained without
  `url_max_length`, an era marker found by the [leakage check](docs/error_analysis.md#last-leakage-check),
  it keeps the same test recall and loses only 1.2 points on new campaigns, at the cost of more
  spam flagged.
- **In a real mailbox, false alarms would be about as frequent as caught phishing.** Out of
  1,000 messages with 1% phishing, LightGBM would catch 9.9 of 10 phishing messages and flag
  8 legitimate ones [0, 24]: the interval rests on five flagged test ham. Probabilities are
  calibrated (ECE 0.7% on test) and converted to a real base rate with Bayes' rule, so a "90%"
  in the corpus means 6% in such an inbox.
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
| 2 — Production | FastAPI service, Docker, Prometheus/Grafana, drift reports, gated auto-retraining | `v0.2.0` | Next, target 2026-12-13 |
| 3 — Agent | LLM agent (Ollama, OpenAI-compatible fallback), typed tools, prompt-injection defence | `v0.3.0` | Target 2027-01-10 |
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
make eda     # re-runs the exploratory notebook on the downloaded data
```

See the `Makefile` for the full list of targets (`make help`).

## Repository layout

```
src/mailguard/
├── data/           corpus manifest, checksummed download, streaming loader, dedup
├── parsing/        raw email -> validated Email model
├── features/       header, URL, text features
├── models/         training, evaluation, registry
├── serving/        FastAPI app, schemas, metrics
├── agent/          LLM tools + investigation loop
├── monitoring/     drift reports, metrics
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
([0004](docs/adr/0004-grouped-splits.md)) and the model choice
([0005](docs/adr/0005-model-choice.md)) so far.

## License

MIT — see [LICENSE](LICENSE).
