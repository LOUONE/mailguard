from __future__ import annotations

import importlib
import importlib.metadata

import mailguard


def test_package_imports() -> None:
    module = importlib.import_module("mailguard")
    assert module is not None


def test_version_matches_installed_metadata() -> None:
    assert mailguard.__version__ == importlib.metadata.version("mailguard")


def test_subpackages_import() -> None:
    for sub in (
        "data",
        "parsing",
        "features",
        "models",
        "serving",
        "agent",
        "monitoring",
        "adversarial",
    ):
        module = importlib.import_module(f"mailguard.{sub}")
        assert module is not None
