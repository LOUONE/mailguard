from __future__ import annotations

import importlib


def test_package_imports() -> None:
    module = importlib.import_module("mailguard")
    assert module is not None


def test_subpackages_import() -> None:
    for sub in (
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
