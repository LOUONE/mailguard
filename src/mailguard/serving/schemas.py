"""The API contract, version 1: what a client sends and what it gets back.

The schemas are strict (unknown fields are refused) and versioned in the URL (`/v1/...`) and in
every response (`schema_version`). A breaking change gets `/v2` next to `/v1`, never an edit of
these classes; the OpenAPI document generated from them is committed in `docs/api/openapi.json`.
"""

from __future__ import annotations

import base64
import binascii
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["v1"] = "v1"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredictRequest(_Strict):
    raw: str = Field(
        min_length=1,
        description="The whole message, headers and body, as RFC 5322 text; or the base64 of its "
        "bytes when `encoding` is `base64` (for messages that are not valid UTF-8).",
    )
    encoding: Literal["text", "base64"] = "text"
    base_rate: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Share of phishing in the mailbox this message comes from, used to convert "
        "the score into `probability`. Defaults to the service setting.",
    )

    @model_validator(mode="after")
    def _decodable(self) -> PredictRequest:
        self.raw_bytes()  # refuse bad base64 at validation time, as a 422
        return self

    def raw_bytes(self) -> bytes:
        if self.encoding == "text":
            return self.raw.encode("utf-8")
        try:
            return base64.b64decode(self.raw, validate=True)
        except binascii.Error as exc:
            raise ValueError("raw is not valid base64") from exc


class Reason(_Strict):
    feature: str = Field(description="`<group>__<name>`; TF-IDF terms are `text__word:...`.")
    group: str
    contribution: float = Field(description="Additive contribution to the log-odds.")
    direction: Literal["phishing", "legitimate"]


class ModelRef(_Strict):
    name: str
    version: str
    operating_point: str = Field(description="FPR target the threshold was chosen for.")
    threshold: float = Field(description="Scores strictly above it are labelled phishing.")


class PredictResponse(_Strict):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    label: Literal["phishing", "legitimate"]
    score: float = Field(
        ge=0,
        le=1,
        description="Calibrated probability at the base rate of the training data. The label "
        "compares this score with the threshold.",
    )
    probability: float = Field(ge=0, le=1, description="The score converted to `base_rate`.")
    base_rate: float
    model: ModelRef
    reasons: list[Reason] = Field(description="Features that moved the score most, largest first.")


class Health(_Strict):
    status: Literal["ok", "unavailable"]
    model: ModelRef | None = None
    detail: str | None = Field(default=None, description="Why the model is not ready.")
