.PHONY: help setup sync data eda lint format test test-unit test-integration coverage pre-commit clean all

help:
	@echo "mailguard — available targets:"
	@echo "  setup           Create venv, install deps, install pre-commit hook"
	@echo "  sync            Sync dependencies from uv.lock"
	@echo "  data            Download corpora into data/raw/ and verify checksums"
	@echo "  eda             Re-execute notebooks/01_eda.ipynb in place (needs make data)"
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
