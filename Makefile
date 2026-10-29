.PHONY: help setup sync data dataset audit train evaluate promote retrain serve openapi demo demo-down mlflow-ui eda lint format test test-unit test-integration coverage pre-commit clean all

help:
	@echo "mailguard — available targets:"
	@echo "  setup           Create venv, install deps, install pre-commit hook"
	@echo "  sync            Sync dependencies from uv.lock"
	@echo "  data            Download corpora into data/raw/ and verify checksums"
	@echo "  dataset         Parse corpora into data/processed/ with grouped splits (needs make data)"
	@echo "  audit           Era-leakage audit of each feature group (needs make dataset)"
	@echo "  train           Train both baselines, log and register them in MLflow (needs make dataset)"
	@echo "  evaluate        Test-split report with cluster-bootstrap intervals into reports/ (needs make train)"
	@echo "  promote         Point the production alias at the latest LightGBM (needs make train)"
	@echo "  retrain         Rebuild, train a candidate, promote it only on a significant gain (ADR 0008)"
	@echo "  serve           Run the scoring API at http://127.0.0.1:8000/docs (needs make promote)"
	@echo "  openapi         Regenerate docs/api/openapi.json after a schema change"
	@echo "  demo            Whole stack in Docker: API, MLflow, Prometheus, Grafana, demo traffic"
	@echo "  demo-down       Stop the stack (volumes kept; add -v to docker compose down to drop them)"
	@echo "  mlflow-ui       Browse local MLflow runs at http://127.0.0.1:5000"
	@echo "  eda          Re-execute notebooks/01_eda.ipynb in place (needs make data)"
	@echo "  lint            Ruff check + format check + mypy"
	@echo "  format          Ruff auto-fix + ruff format"
	@echo "  test            Run full test suite with coverage"
	@echo "  test-unit       Run tests marked 'unit' only"
	@echo "  test-integration  Run tests marked 'integration' only"
	@echo "  coverage        Run tests, emit coverage.xml, fail under 90% coverage"
	@echo "  pre-commit      Run all pre-commit hooks against every file"
	@echo "  clean           Remove caches and build artifacts"
	@echo "  all             lint + test"

setup:
	uv sync --all-extras --dev
	uv run pre-commit install

sync:
	uv sync --all-extras --dev

data:
	uv run python scripts/download_data.py

dataset:
	uv run python scripts/build_dataset.py

audit:
	uv run python scripts/audit_features.py

train:
	uv run python scripts/train.py

evaluate:
	uv run python scripts/evaluate.py

promote:
	uv run python scripts/promote.py

retrain:
	uv run python scripts/retrain.py

serve:
	uv run uvicorn --factory mailguard.serving.app:create_app --host 127.0.0.1 --port 8000 --no-access-log

openapi:
	uv run python scripts/export_openapi.py

demo:
	docker compose up --build -d
	@echo "API http://127.0.0.1:8000/docs  MLflow http://127.0.0.1:5000  Grafana http://127.0.0.1:3000"
	@echo "First start trains the model (about 5 min): docker compose logs -f bootstrap"

demo-down:
	docker compose down

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

eda:
	uv run --group notebooks jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-unit:
	uv run pytest -m unit

test-integration:
	uv run pytest -m integration

coverage:
	uv run pytest --cov=mailguard --cov-report=xml --cov-report=term-missing --cov-fail-under=90

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml build dist *.egg-info

all: lint test
