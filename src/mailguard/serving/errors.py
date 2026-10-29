"""One error shape for every failure, and nothing in it that the client did not need.

FastAPI's default 422 body echoes the offending input back; for this API the input is someone's
email, so validation errors keep the location and the message and drop the value. Unexpected
exceptions become a generic 500: the type goes to the logs, never a stack trace to the client.
"""

from __future__ import annotations

import logging
from typing import Final

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from mailguard.serving.schemas import ErrorBody, ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

CODES: Final = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "invalid_request",
    500: "internal_error",
    503: "unavailable",
}


def request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def error_response(
    request: Request, status_code: int, message: str, details: list[ErrorDetail] | None = None
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=CODES.get(status_code, "error"),
            message=message,
            request_id=request_id(request),
            details=details or [],
        )
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)


async def on_http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)
    return error_response(request, exc.status_code, str(exc.detail))


async def on_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = [
        ErrorDetail(location=[str(part) for part in e.get("loc", ())], message=str(e.get("msg")))
        for e in exc.errors()
    ]
    return error_response(
        request, status.HTTP_422_UNPROCESSABLE_CONTENT, "the request is invalid", details
    )


async def on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # The exception type only: its message could quote the email.
    logger.error("unexpected %s on %s", type(exc).__name__, request.url.path)
    return error_response(request, status.HTTP_500_INTERNAL_SERVER_ERROR, "internal error")


def install(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, on_http_error)
    app.add_exception_handler(RequestValidationError, on_validation_error)
    app.add_exception_handler(Exception, on_unexpected_error)
