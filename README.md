# mailguard

Real-time phishing detection with an LLM investigation agent, built for privacy.

[![CI](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml/badge.svg)](https://github.com/LOUONE/mailguard/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](pyproject.toml)

> **Status: work in progress.** Phase 0 (scaffolding + CI) is complete. Phase 1 (data pipeline + classical baseline) is next. Expected final version: `v1.0.0`.

## What this project aims to prove

1. **Ship ML to production.** A real-time API, containerised, monitored, with automated retraining.
2. **Statistical rigour.** Temporal splits, confidence intervals, calibrated probabilities, honest error analysis — not just "99% accuracy".
3. **Agentic AI.** An open-source LLM agent that investigates ambiguous emails with typed tools and produces a human-readable verdict.
4. **Attacker mindset.** Red-team the model (homoglyphs, LLM-rewritten phishing) and measure robustness before and after hardening.
5. **Clean Python.** Typed, tested, packaged, CI green.

## Roadmap

| Phase | Weeks | Deliverable | Release |
|---|---|---|---|
| 0 — Foundations | 1 | Repo scaffolded, CI green, `main` protected | `v0.0.1` |
| 1 — Data + classical model | 2-4 | Parsed corpora, LightGBM baseline, rigorous eval, SHAP, MLflow | `v0.1.0` |
| 2 — Production | 5-8 | FastAPI service, Docker, Prometheus/Grafana, drift reports, auto-retraining | `v0.2.0` |
| 3 — Agent | 9-11 | LLM agent (Ollama/vLLM), typed tools, prompt-injection defence | `v0.3.0` |
| 4 — Adversarial | 12 | Attack suite + hardening, before/after metrics | `v1.0.0` |

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
```

See the `Makefile` for the full list of targets (`make help`).

## Repository layout

```
src/mailguard/
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
that exist only in recent phishing).

## License

MIT — see [LICENSE](LICENSE).
