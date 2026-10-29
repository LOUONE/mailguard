"""The compose stack's helper scripts: the bootstrap (smoke mode) and the demo traffic."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mailguard.config import Settings
from mailguard.models.registry import aliased_version
from mailguard.parsing import parse_email


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bootstrap = load_script("bootstrap")
demo_traffic = load_script("demo_traffic")


def test_smoke_bootstrap_promotes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlruns' / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MAILGUARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MAILGUARD_MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MAILGUARD_MLFLOW_ARTIFACT_DIR", str(tmp_path / "mlruns" / "artifacts"))
    settings = Settings()

    assert bootstrap.main(["--smoke"]) == 0
    assert aliased_version(settings.model_name, settings).version == "1"
    assert bootstrap.main(["--smoke"]) == 0  # already in production: nothing retrained
    assert aliased_version(settings.model_name, settings).version == "1"


def test_demo_traffic_is_synthetic_and_parses() -> None:
    rng = random.Random(0)

    bodies = [demo_traffic.request(rng) for _ in range(300)]

    raws = [str(b["raw"]) for b in bodies if b["raw"]]
    assert len(raws) < 300  # a few invalid requests on purpose
    for raw in raws:
        email = parse_email(raw.encode())
        assert (email.from_address or "").endswith((".example", "@example.org"))  # RFC 2606
    lures = sum("Dear customer" in r for r in raws)
    assert 0.1 < lures / len(raws) < 0.35
