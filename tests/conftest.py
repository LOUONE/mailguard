from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, settings

# "dev" keeps the local loop fast; CI and pre-release runs use "ci" for a deeper search:
#   HYPOTHESIS_PROFILE=ci uv run pytest
settings.register_profile("dev", max_examples=200, deadline=None)
settings.register_profile(
    "ci",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test `unit` or `integration` from its folder, so markers cannot be forgotten."""
    for item in items:
        item.add_marker("integration" if "integration" in item.path.parts else "unit")
