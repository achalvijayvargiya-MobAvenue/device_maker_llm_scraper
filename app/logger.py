"""
Structured logging setup with console + rotating file handlers.
Uses structlog for machine-readable JSON logs alongside human-readable console output.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Configure root logger with:
    - Colored console output (human-readable).
    - Rotating JSON-structured file handler when log_file is provided.

    Parameters
    ----------
    log_level:
        One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    log_file:
        Path to the log file. Rotated at 10 MB, keeps 5 backups.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    console_fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_fmt = (
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
        '"message":%(message)s}'
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any existing handlers (e.g., when called multiple times in tests)
    root.handlers.clear()

    # --- Console handler ---
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric_level)
    ch.setFormatter(logging.Formatter(console_fmt, datefmt="%H:%M:%S"))
    root.addHandler(ch)

    # --- File handler ---
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(logging.Formatter(file_fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        root.addHandler(fh)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance."""
    return logging.getLogger(name)
