"""Structured request logging middleware for the serving API."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_LOGGER_NAME = "hf_finetuning_lab.serving.requests"


def get_request_logger() -> logging.Logger:
    """Return the dedicated request logger (configurable from the outside)."""
    return logging.getLogger(REQUEST_LOGGER_NAME)


class StructuredRequestLogger(BaseHTTPMiddleware):
    """Emit one JSON-formatted log line per request with latency + status."""

    def __init__(self, app: Any, model_version: str | None = None) -> None:
        super().__init__(app)
        self._model_version = model_version
        self._logger = get_request_logger()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000.0
            payload: dict[str, Any] = {
                "event": "request",
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
                "latency_ms": round(latency_ms, 3),
            }
            if self._model_version is not None:
                payload["model_version"] = self._model_version
            client = request.client
            if client is not None:
                payload["client"] = f"{client.host}:{client.port}"
            self._logger.info(json.dumps(payload, sort_keys=True))
