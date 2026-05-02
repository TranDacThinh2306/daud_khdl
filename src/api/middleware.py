"""
middleware.py - Logging, auth, rate limiting
==============================================
Custom middleware for the API server.
"""

import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log request/response details."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        logger.info(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code} ({duration:.3f}s)"
        )
        response.headers["X-Process-Time"] = f"{duration:.3f}"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting."""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )

        self._requests[client_ip].append(now)
        return await call_next(request)
