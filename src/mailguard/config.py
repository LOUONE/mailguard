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

    # Serving (Phase 2). The API serves the registered version behind `model_alias`.
    model_name: str = "mailguard-lightgbm"
    model_alias: str = "production"
    operating_point: Literal["1pct", "01pct"] = "1pct"
    # Share of phishing in the mailboxes served, to turn scores into probabilities (ADR 0001).
    base_rate: float = Field(default=0.01, gt=0, lt=1)
    # Request limits. Only text and attachment metadata are read, so a bigger email adds cost
    # without adding signal; 2 MiB holds all but one of the 14,084 corpus messages.
    max_email_bytes: int = Field(default=2 * 1024 * 1024, gt=0)
    scoring_timeout_s: float = Field(default=2.0, gt=0)


def get_settings() -> Settings:
    return Settings()
