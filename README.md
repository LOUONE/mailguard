# mailguard

Real-time phishing detection with an LLM investigation agent, built for privacy.

[![CI](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml/badge.svg)](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](pyproject.toml)

> **Status: work in progress.** Phase 0 (scaffolding + CI) is complete. Phase 1 is in progress: the data pipeline, parser and [exploratory analysis](notebooks/01_eda.ipynb) are done, the classical baseline is next. Expected final version: `v1.0.0`.

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
| 1 — Data + classical model | Parsed corpora, grouped splits, features, LightGBM baseline, rigorous eval, SHAP, MLflow | `v0.1.0` | In progress, target 2026-11-15 |
| 2 — Production | FastAPI service, Docker, Prometheus/Grafana, drift reports, gated auto-retraining | `v0.2.0` | Target 2026-12-13 |
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
the binary target ([0001](docs/adr/0001-binary-target.md)) and the parser design
([0002](docs/adr/0002-parser-design.md)) so far.

## License

MIT — see [LICENSE](LICENSE).
