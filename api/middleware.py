"""
FastAPI middleware for request logging, error handling, and timing.
"""

import time
import traceback
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every incoming request with timing, status, and metadata.
    
    Captures:
    - Request method, path, client IP
    - Response status code
    - Processing time in milliseconds
    - Error details on failure
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        request_id = f"{time.time_ns()}"

        # Log incoming request
        logger.info(
            f"→ {request.method} {request.url.path}",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "client": request.client.host if request.client else "unknown",
                    "query_params": str(request.query_params),
                }
            },
        )

        try:
            response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000

            # Add timing header
            response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
            response.headers["X-Request-Id"] = request_id

            # Log response
            logger.info(
                f"← {request.method} {request.url.path} → {response.status_code} "
                f"({process_time:.2f}ms)",
                extra={
                    "extra_data": {
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "process_time_ms": round(process_time, 2),
                    }
                },
            )
            return response

        except Exception as e:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"✖ {request.method} {request.url.path} → 500 ({process_time:.2f}ms): {e}",
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(e) if logger.level <= 10 else "An unexpected error occurred",
                    "request_id": request_id,
                },
            )


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"ValueError: {exc}")
        return JSONResponse(
            status_code=400,
            content={"error": "Bad request", "detail": str(exc)},
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning(f"FileNotFoundError: {exc}")
        return JSONResponse(
            status_code=404,
            content={"error": "Resource not found", "detail": str(exc)},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        logger.error(f"RuntimeError: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error", "detail": str(exc)},
        )
