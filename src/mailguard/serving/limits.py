"""Request ids and a cap on the request body, as plain ASGI middleware.

The size check runs while the body is read from the socket, before any JSON parsing: a client
announcing 1 GB with `Content-Length` is refused at once, and one streaming a body without a
length is cut off as soon as it passes the cap. Checking `len(body.raw)` in the endpoint would
come after the whole body had been buffered and parsed.
"""

from __future__ import annotations

import re
import uuid
from typing import Final

from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mailguard.serving.errors import error_response

REQUEST_ID_HEADER: Final = "x-request-id"
# A client may pass its own id to correlate logs; anything else is replaced, never echoed.
_CLIENT_ID_RE: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def request_id_from(scope: Scope) -> str:
    for name, value in scope.get("headers", ()):
        if name == REQUEST_ID_HEADER.encode():
            candidate: str = value.decode("latin-1")
            if _CLIENT_ID_RE.match(candidate):
                return candidate
    return uuid.uuid4().hex


def content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", ()):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class RequestLimits:
    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        rid = request_id_from(scope)
        scope.setdefault("state", {})["request_id"] = rid

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), rid.encode()))
                message = {**message, "headers": headers}
            await send(message)

        announced = content_length(scope)
        if announced is not None and announced > self.max_body_bytes:
            response = error_response(Request(scope), 413, self._too_large())
            await response(scope, receive, send_with_id)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # FastAPI re-raises HTTPException from body reading as is.
                    raise HTTPException(413, self._too_large())
            return message

        await self.app(scope, limited_receive, send_with_id)

    def _too_large(self) -> str:
        return f"request body over {self.max_body_bytes} bytes"
