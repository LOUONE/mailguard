"""The scoring API: a raw email in, a calibrated score, a label and its reasons out.

    uv run uvicorn --factory mailguard.serving.app:create_app

The model is loaded once, at startup, from the MLflow registry. If loading fails the process
stays up and says so on `/health` (503), instead of crashing in a restart loop: an orchestrator
can tell "starting" from "broken", and an operator can read why.

The endpoint receives hostile input by design, so every request is bounded:

- size: the body is capped while it is read (`RequestLimits`), the decoded email again after;
- time: scoring runs in a worker thread, off the event loop, and the request gives up with a 503
  after `scoring_timeout_s`. Python cannot kill a thread, so a timed-out scoring still finishes
  in the background; the cap on size is what bounds that work;
- errors: one JSON shape for every failure, without stack traces or echoed input (`errors.py`).

Logs are JSON, one access line per request with sizes, timings, the score and the model version,
and nothing the sender wrote (`monitoring/logs.py`, `access.py`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Final

import structlog
from fastapi import FastAPI, HTTPException, Request, Response, status
from starlette.concurrency import run_in_threadpool

from mailguard import __version__
from mailguard.config import Settings, get_settings
from mailguard.monitoring.logs import configure_logging
from mailguard.parsing import parse_email
from mailguard.serving import errors
from mailguard.serving.access import RequestLog
from mailguard.serving.limits import RequestLimits
from mailguard.serving.model import Prediction, ServedModel, load_production
from mailguard.serving.schemas import (
    ErrorResponse,
    Health,
    PredictRequest,
    PredictResponse,
    Reason,
)

log = structlog.get_logger("mailguard.serving")

Loader = Callable[[Settings], ServedModel]
# Base64 is 4/3 of the bytes it encodes, plus room for the JSON around it.
JSON_OVERHEAD: Final = 64 * 1024


def score(model: ServedModel, raw: bytes, base_rate: float) -> Prediction:
    """Parse and score one message: the whole CPU-bound part of a request."""
    return model.predict(parse_email(raw), base_rate)


def _errors(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {code: {"model": ErrorResponse} for code in codes}


def create_app(settings: Settings | None = None, loader: Loader = load_production) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model, app.state.load_error = None, None
        try:
            model = await run_in_threadpool(loader, settings)
            app.state.model = model
            log.info("model_loaded", model=model.name, model_version=model.version)
        except Exception as exc:  # any failure must leave the API up, and not ready
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            log.error("model_not_loaded", error_type=type(exc).__name__, detail=str(exc))
        yield

    app = FastAPI(
        title="mailguard",
        version=__version__,
        summary="Phishing detection for raw email, with calibrated scores and reasons.",
        lifespan=lifespan,
    )
    errors.install(app)
    app.add_middleware(
        RequestLimits, max_body_bytes=settings.max_email_bytes * 4 // 3 + JSON_OVERHEAD
    )
    app.add_middleware(RequestLog)  # added last, so it is the outermost: it sees everything

    def served(request: Request) -> ServedModel:
        model: ServedModel | None = request.app.state.model
        if model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return model

    @app.get("/health", responses={503: {"model": Health, "description": "No model loaded."}})
    async def health(request: Request, response: Response) -> Health:
        """Liveness (the process answers) and readiness (a model is loaded)."""
        model: ServedModel | None = request.app.state.model
        if model is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="unavailable", detail=request.app.state.load_error)
        return Health(status="ok", model=model.ref)

    @app.post("/v1/predict", responses=_errors(413, 422, 500, 503))
    async def predict(body: PredictRequest, request: Request) -> PredictResponse:
        """Score one raw email."""
        model = served(request)
        raw = body.raw_bytes()
        request.state.log["email_bytes"] = len(raw)
        if len(raw) > settings.max_email_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"email over {settings.max_email_bytes} bytes",
            )
        base_rate = body.base_rate or settings.base_rate
        start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(score, model, raw, base_rate), settings.scoring_timeout_s
            )
        except TimeoutError:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"scoring took over {settings.scoring_timeout_s} s",
            ) from None
        label = "phishing" if result.is_phishing else "legitimate"
        request.state.log |= {
            "scoring_ms": round(1000 * (time.perf_counter() - start), 2),
            "label": label,
            "score": round(result.score, 4),
            "model_version": model.version,
        }
        return PredictResponse(
            label=label,
            score=result.score,
            probability=result.probability,
            base_rate=base_rate,
            model=model.ref,
            reasons=[Reason.model_validate(r) for r in result.reasons],
        )

    return app
