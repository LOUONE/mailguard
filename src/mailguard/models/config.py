"""Typed, validated hyperparameters for the baselines, loaded from `configs/baselines.toml`."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG: Final = Path("configs/baselines.toml")


class _Strict(BaseModel):
    # A misspelt key in the TOML file must fail loudly, not fall back to a default.
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextConfig(_Strict):
    word_max_features: int = Field(default=50_000, gt=0)
    char_max_features: int = Field(default=100_000, gt=0)
    min_df: int = Field(default=3, ge=1)


class LogRegConfig(_Strict):
    C: float = Field(default=4.0, gt=0)
    max_iter: int = Field(default=2000, gt=0)
    text: TextConfig = TextConfig()


class LightGBMConfig(_Strict):
    n_estimators: int = Field(default=500, gt=0)
    learning_rate: float = Field(default=0.05, gt=0)
    num_leaves: int = Field(default=31, gt=1)
    min_child_samples: int = Field(default=20, ge=1)
    colsample_bytree: float = Field(default=0.3, gt=0, le=1)
    reg_lambda: float = Field(default=1.0, ge=0)
    n_jobs: int = Field(default=4, ge=1)
    text: TextConfig = TextConfig(word_max_features=20_000, char_max_features=30_000)


class BaselineConfig(_Strict):
    seed: int = 20261027
    logreg: LogRegConfig = LogRegConfig()
    lightgbm: LightGBMConfig = LightGBMConfig()


def load_config(path: Path = DEFAULT_CONFIG) -> BaselineConfig:
    with path.open("rb") as f:
        return BaselineConfig.model_validate(tomllib.load(f))


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """`{"a": {"b": 1}}` -> `{"a.b": 1}`, the shape MLflow expects for parameters."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= flatten(value, f"{name}.")
        else:
            out[name] = value
    return out
