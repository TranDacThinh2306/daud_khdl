"""
logger.py - Logging configuration
====================================
Centralized logging setup for the project.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "depression_alert",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for log output
        fmt: Log message format

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
