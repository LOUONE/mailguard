"""The outermost middleware: request id, one access log line per request, the last-resort 500.

Why the 500 lives here and not in a FastAPI exception handler: Starlette runs a handler
registered for `Exception` and then re-raises, so the server logs the exception with its
traceback, and an exception message can quote the email (reproduced with uvicorn while writing
this module). Catching here, inside Starlette's own error middleware, means nothing reaches it.

Endpoints add fields to the access line through `request.state.log` (sizes, scores, versions),
never through free text.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Final

import structlog
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mailguard.serving.errors import error_response

REQUEST_ID_HEADER: Final = "x-request-id"
# A client may pass its own id to correlate logs; anything else is replaced, never echoed.
_CLIENT_ID_RE: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

log = structlog.get_logger("mailguard.access")


def request_id_from(scope: Scope) -> str:
    for name, value in scope.get("headers", ()):
        if name == REQUEST_ID_HEADER.encode():
            candidate: str = value.decode("latin-1")
            if _CLIENT_ID_RE.match(candidate):
                return candidate
    return uuid.uuid4().hex


class RequestLog:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = request_id_from(scope)
        state: dict[str, Any] = scope.setdefault("state", {})
        state["request_id"], state["log"] = rid, {}
        start = time.perf_counter()
        response: dict[str, Any] = {"status": 500, "started": False}

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                response["status"], response["started"] = message["status"], True
                headers = [*message.get("headers", []), (REQUEST_ID_HEADER.encode(), rid.encode())]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception as exc:  # the type only: the message could quote the email
            log.error("unhandled_error", request_id=rid, error_type=type(exc).__name__)
            if not response["started"]:
                await error_response(Request(scope), 500, "internal error")(
                    scope, receive, send_with_id
                )
        finally:
            # The route template, not the raw path: a client can put anything in a URL.
            route = scope.get("route")
            log.info(
                "request",
                request_id=rid,
                method=scope["method"],
                path=getattr(route, "path", "unmatched"),
                status=response["status"],
                duration_ms=round(1000 * (time.perf_counter() - start), 2),
                **state.get("log", {}),
            )
