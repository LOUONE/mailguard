from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["dev", "test", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MAILGUARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    app_name: str = "mailguard"
    environment: Environment = "dev"
    log_level: LogLevel = "INFO"
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    # Local experiment tracking (gitignored). Phase 2 points this at the compose MLflow server.
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    mlflow_artifact_dir: Path = Field(default_factory=lambda: Path("mlruns/artifacts"))


def get_settings() -> Settings:
    return Settings()
