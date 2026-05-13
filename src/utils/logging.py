"""Structured logging via structlog.

Use :func:`get_logger` everywhere instead of stdlib ``logging`` directly.
Call :func:`configure_logging` exactly once at process start (done in main.py).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

import structlog
from structlog.types import Processor

from src.config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """Initialise structlog + stdlib logging. Idempotent."""
    settings = settings or get_settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, settings.log_level.value)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.format_exc_info,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(file=_open_log_sink(settings.logs_dir)),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level, format="%(message)s", handlers=[logging.StreamHandler()])


def _open_log_sink(logs_dir: Path) -> Any:
    """Tee logs to stdout. Replace with rotating file handler when needed."""
    return sys.stdout


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a logger, optionally bound with starting context."""
    logger = structlog.get_logger(name)
    bound = logger.bind(**initial_context) if initial_context else logger
    # structlog.get_logger returns a bound proxy whose static type is too
    # loose; the runtime object IS a BoundLogger.
    return cast(structlog.stdlib.BoundLogger, bound)
