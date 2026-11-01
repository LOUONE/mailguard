from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mailguard.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for var in (
        "MAILGUARD_ENVIRONMENT",
        "MAILGUARD_LOG_LEVEL",
        "MAILGUARD_DATA_DIR",
        "MAILGUARD_APP_NAME",
        "MAILGUARD_MLFLOW_TRACKING_URI",
        "MAILGUARD_MLFLOW_ARTIFACT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings()

    assert settings.app_name == "mailguard"
    assert settings.environment == "dev"
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")
    assert settings.mlflow_tracking_uri == "sqlite:///mlruns/mlflow.db"
    assert settings.mlflow_artifact_dir == Path("mlruns/artifacts")


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAILGUARD_ENVIRONMENT", "prod")
    monkeypatch.setenv("MAILGUARD_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("MAILGUARD_DATA_DIR", str(tmp_path / "custom"))

    settings = Settings()

    assert settings.environment == "prod"
    assert settings.log_level == "WARNING"
    assert settings.data_dir == tmp_path / "custom"


def test_invalid_environment_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAILGUARD_ENVIRONMENT", "staging")

    with pytest.raises(ValidationError):
        Settings()


def test_llm_key_comes_from_the_environment_and_stays_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAILGUARD_LLM_BACKEND", "openai")
    monkeypatch.setenv("MAILGUARD_LLM_API_KEY", "sk-very-secret")

    settings = Settings()

    assert settings.llm_backend == "openai"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "sk-very-secret"
    assert "sk-very-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("MAILGUARD_LLM_BACKEND", "anthropic"),
        ("MAILGUARD_LLM_CONTEXT_TOKENS", "512"),
        ("MAILGUARD_LLM_MAX_RETRIES", "10"),
    ],
)
def test_invalid_llm_settings_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, var: str, value: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(var, value)

    with pytest.raises(ValidationError):
        Settings()
