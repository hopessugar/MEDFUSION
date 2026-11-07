"""Structured logging infrastructure for MedFusion-RAFT.

Configures structlog with JSON formatting for production or rich console formatting
for development, including context variable propagation (e.g., request_id) and
performance timing context managers.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar, cast

import structlog
from structlog.types import EventDict, WrappedLogger

# Context variable for tracing request IDs across async boundaries
_REQUEST_ID_CTX: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


def set_request_id(request_id: Optional[str]) -> None:
    """Set the current context's request identifier.

    Args:
        request_id: Unique trace or request ID.
    """
    _REQUEST_ID_CTX.set(request_id)


def get_request_id() -> Optional[str]:
    """Retrieve the current context's request identifier.

    Returns:
        The active request_id string, or None if not set.
    """
    return _REQUEST_ID_CTX.get()


@contextmanager
def request_context(request_id: str) -> Generator[None, None, None]:
    """Context manager setting an active request ID for the block.

    Args:
        request_id: Unique trace or request identifier.
    """
    token = _REQUEST_ID_CTX.set(request_id)
    try:
        yield
    finally:
        _REQUEST_ID_CTX.reset(token)


def _add_request_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Structlog processor adding active request_id to log records."""
    req_id = _REQUEST_ID_CTX.get()
    if req_id is not None:
        event_dict["request_id"] = req_id
    return event_dict


def setup_logging(
    log_level: Optional[str] = None,
    json_format: Optional[bool] = None,
) -> None:
    """Initialize structured logging system.

    Args:
        log_level: Log level name (DEBUG, INFO, WARNING, ERROR).
                   Defaults to MEDFUSION_LOG_LEVEL env var or 'INFO'.
        json_format: Whether to output JSON lines. If None, inspects
                     MEDFUSION_LOG_JSON env var or defaults to False in dev.
    """
    if log_level is None:
        log_level = os.environ.get("MEDFUSION_LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, log_level, logging.INFO)

    if json_format is None:
        json_format = os.environ.get("MEDFUSION_LOG_JSON", "false").lower() in (
            "true",
            "1",
            "yes",
        )

    # Standard library root logging configuration
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_format:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Attach structlog formatter to the default root handler
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Obtain a structured logger instance.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Configured structlog BoundLogger.
    """
    return structlog.get_logger(name)


@contextmanager
def log_execution_time(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    log_start: bool = False,
    **extra_kwargs: Any,
) -> Generator[None, None, None]:
    """Context manager measuring and logging elapsed duration of an operation.

    Args:
        logger: Bound logger to emit timing events.
        operation: Human-readable name of the action being measured.
        log_start: Whether to emit an entry event at operation start.
        extra_kwargs: Additional metadata key-values to include in log output.
    """
    if log_start:
        logger.debug(f"Starting {operation}", operation=operation, **extra_kwargs)

    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"Completed {operation}",
            operation=operation,
            duration_ms=round(elapsed_ms, 2),
            **extra_kwargs,
        )


def timed(
    operation: Optional[str] = None,
    logger: Optional[structlog.stdlib.BoundLogger] = None,
) -> Callable[[F], F]:
    """Decorator measuring synchronous or asynchronous function duration.

    Args:
        operation: Operation label. Defaults to decorated function name.
        logger: Logger to use. If None, dynamically resolves logger for function.

    Returns:
        Decorated callable.
    """

    def decorator(fn: F) -> F:
        op_name = operation or fn.__name__
        fn_logger = logger or get_logger(fn.__module__)

        @wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with log_execution_time(fn_logger, op_name):
                return fn(*args, **kwargs)

        return cast(F, sync_wrapper)

    return decorator
