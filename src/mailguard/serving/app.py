"""The scoring API: a raw email in, a calibrated score, a label and its reasons out.

    uv run uvicorn --factory mailguard.serving.app:create_app

The model is loaded once, at startup, from the MLflow registry. If loading fails the process
stays up and says so on `/health` (503), instead of crashing in a restart loop: an orchestrator
can tell "starting" from "broken", and an operator can read why.

Scoring is CPU-bound (parsing, features, trees), so it runs in a worker thread and never blocks
the event loop that serves other requests.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from starlette.concurrency import run_in_threadpool

from mailguard import __version__
from mailguard.config import Settings, get_settings
from mailguard.parsing import parse_email
from mailguard.serving.model import Prediction, ServedModel, load_production
from mailguard.serving.schemas import Health, PredictRequest, PredictResponse, Reason

logger = logging.getLogger(__name__)

Loader = Callable[[Settings], ServedModel]


def score(model: ServedModel, raw: bytes, base_rate: float) -> Prediction:
    """Parse and score one message: the whole CPU-bound part of a request."""
    return model.predict(parse_email(raw), base_rate)


def create_app(settings: Settings | None = None, loader: Loader = load_production) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model, app.state.load_error = None, None
        try:
            app.state.model = await run_in_threadpool(loader, settings)
            logger.info("serving %s v%s", app.state.model.name, app.state.model.version)
        except Exception as exc:  # any failure must leave the API up, and not ready
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.error("model not loaded: %s", app.state.load_error)
        yield

    app = FastAPI(
        title="mailguard",
        version=__version__,
        summary="Phishing detection for raw email, with calibrated scores and reasons.",
        lifespan=lifespan,
    )

    def served(request: Request) -> ServedModel:
        model: ServedModel | None = request.app.state.model
        if model is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model not loaded")
        return model

    @app.get(
        "/health",
        responses={503: {"model": Health, "description": "The model is not loaded."}},
    )
    async def health(request: Request, response: Response) -> Health:
        """Liveness (the process answers) and readiness (a model is loaded)."""
        model: ServedModel | None = request.app.state.model
        if model is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return Health(status="unavailable", detail=request.app.state.load_error)
        return Health(status="ok", model=model.ref)

    @app.post(
        "/v1/predict",
        responses={503: {"description": "The model is not loaded."}},
    )
    async def predict(body: PredictRequest, request: Request) -> PredictResponse:
        """Score one raw email."""
        model = served(request)
        base_rate = body.base_rate or settings.base_rate
        result = await run_in_threadpool(score, model, body.raw_bytes(), base_rate)
        return PredictResponse(
            label="phishing" if result.is_phishing else "legitimate",
            score=result.score,
            probability=result.probability,
            base_rate=base_rate,
            model=model.ref,
            reasons=[Reason.model_validate(r) for r in result.reasons],
        )

    return app
