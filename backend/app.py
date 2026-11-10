"""FastAPI application factory and middleware configuration for MedFusion-RAFT."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import router as api_router
from backend.schemas.responses import ErrorResponse
from backend.services.pipeline_service import PipelineService
from src.utils.logging import request_context, setup_logging

logger = logging.getLogger(__name__)


def create_app(config_dir: str = "./configs", mock: bool = False, allowed_origins: list[str] | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Args:
        config_dir: Filesystem directory path containing configuration YAML files.
        mock: If True, boots models in mock mode for CPU-only testing.
        allowed_origins: List of allowed CORS origins. If None, uses restrictive defaults.

    Returns:
        Configured FastAPI application instance.
    """
    setup_logging(log_level="INFO")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application startup and shutdown lifespan context."""
        logger.info("Initializing MedFusion-RAFT backend (mock=%s)...", mock)
        service = PipelineService.get_instance(config_dir=config_dir, mock=mock)
        yield
        logger.info("Shutting down MedFusion-RAFT backend...")
        service.shutdown()

    app = FastAPI(
        title="MedFusion-RAFT Clinical QA API",
        description=(
            "Production-grade Medical Question-Answering API with Retrieval-Augmented "
            "Fine-Tuning, Independent Verification, and Selective Abstention routing."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. CORS Middleware - Restrictive by default for security
    # For development, set allowed_origins=["http://localhost:3000", "http://localhost:8080"]
    # For production, specify exact frontend domains
    cors_origins = allowed_origins or [
        "http://localhost:3000",  # Default React dev server
        "http://localhost:8080",  # Alternative dev port
    ]
    
    # Security: Only allow wildcard in development with explicit opt-in
    if "*" in cors_origins:
        logger.warning(
            "CORS configured with wildcard origin '*' - only use in development! "
            "For production, specify exact allowed origins."
        )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
        allow_headers=["Content-Type", "Authorization", "X-Request-ID"],  # Explicit headers
    )

    # 2. Request Logging & Correlation Middleware
    @app.middleware("http")
    async def log_and_trace_requests(
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """Trace incoming HTTP requests with latency logging and unique correlation IDs."""
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()

        with request_context(req_id):
            logger.info(
                "Started %s %s [request_id=%s]",
                request.method,
                request.url.path,
                req_id,
            )
            try:
                response = await call_next(request)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                response.headers["X-Request-ID"] = req_id
                response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
                logger.info(
                    "Completed %s %s status=%d duration=%.2fms",
                    request.method,
                    request.url.path,
                    response.status_code,
                    latency_ms,
                )
                return response
            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                logger.error(
                    "Unhandled exception in %s %s: %s (duration=%.2fms)",
                    request.method,
                    request.url.path,
                    exc,
                    latency_ms,
                    exc_info=True,
                )
                raise

    # 3. Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle schema validation errors."""
        logger.warning("Request validation failed on %s: %s", request.url.path, exc.errors())
        err = ErrorResponse(
            error="Validation Error",
            detail=str(exc.errors()),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=err.model_dump(mode="json"),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        """Handle explicit HTTP exceptions."""
        err = ErrorResponse(
            error=exc.detail if isinstance(exc.detail, str) else "HTTP Exception",
            detail=str(exc.detail),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=err.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Catch-all unexpected error handler."""
        logger.error("Unhandled 500 error on %s: %s", request.url.path, exc, exc_info=True)
        err = ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=err.model_dump(mode="json"),
        )

    # 4. Attach API routes
    app.include_router(api_router)

    return app
