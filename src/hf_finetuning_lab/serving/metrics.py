"""Optional Prometheus metrics for the serving API.

The ``prometheus_client`` dependency is imported lazily so the rest of the
serving layer stays usable without it. ``install_metrics`` mounts a
``/metrics`` endpoint on the supplied app and returns the counter / histogram
pair used by the request-tracking middleware.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI


class MetricsMiddleware(BaseHTTPMiddleware):
    """Increment a Prometheus counter and histogram on every request."""

    def __init__(self, app: Any, request_counter: Any, latency_histogram: Any) -> None:
        super().__init__(app)
        self._counter = request_counter
        self._histogram = latency_histogram

    @staticmethod
    def _route_template(request: Request) -> str:
        """Return the matched route's path template, or ``"unmatched"``.

        Labelling by the template (e.g. ``/predict``) rather than the raw URL
        path keeps Prometheus label cardinality bounded: 404 probes and random
        scan paths all collapse into a single ``"unmatched"`` series instead of
        minting a new time series each.
        """
        for route in request.app.routes:
            try:
                match, _ = route.matches(request.scope)
            except Exception:  # pragma: no cover - defensive against odd routes
                continue
            if match == Match.FULL:
                return getattr(route, "path", request.url.path)
        return "unmatched"

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
            elapsed = time.perf_counter() - start
            path = self._route_template(request)
            labels = {
                "method": request.method,
                "path": path,
                "status": str(status_code),
            }
            self._counter.labels(**labels).inc()
            self._histogram.labels(method=labels["method"], path=path).observe(elapsed)


def install_metrics(app: FastAPI) -> tuple[Any, Any]:
    """Install Prometheus instrumentation on ``app`` and return ``(counter, histogram)``.

    Raises ``ImportError`` when ``prometheus_client`` is unavailable so callers
    can surface a clear message instead of a generic attribute error.
    """
    try:
        from prometheus_client import (  # type: ignore[import-not-found]
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Histogram,
            generate_latest,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install `prometheus-client` to enable serving metrics "
            "(e.g. `pip install prometheus-client`)."
        ) from exc

    registry = CollectorRegistry()
    request_counter = Counter(
        "hf_lab_requests_total",
        "Number of API requests handled, labelled by method/path/status.",
        labelnames=("method", "path", "status"),
        registry=registry,
    )
    latency_histogram = Histogram(
        "hf_lab_request_latency_seconds",
        "Request latency in seconds, labelled by method/path.",
        labelnames=("method", "path"),
        registry=registry,
    )

    app.add_middleware(
        MetricsMiddleware,
        request_counter=request_counter,
        latency_histogram=latency_histogram,
    )

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return request_counter, latency_histogram
