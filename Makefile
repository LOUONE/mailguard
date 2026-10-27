.PHONY: help setup sync data dataset audit train mlflow-ui eda lint format test test-unit test-integration coverage pre-commit clean all

help:
	@echo "mailguard — available targets:"
	@echo "  setup           Create venv, install deps, install pre-commit hook"
	@echo "  sync            Sync dependencies from uv.lock"
	@echo "  data            Download corpora into data/raw/ and verify checksums"
	@echo "  dataset         Parse corpora into data/processed/ with grouped splits (needs make data)"
	@echo "  audit           Era-leakage audit of each feature group (needs make dataset)"
	@echo "  train           Train both baselines, log and register them in MLflow (needs make dataset)"
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
