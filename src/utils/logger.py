"""
Centralized logging configuration for RL Meter Analyst project.

Usage in other modules:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.debug("Detailed debug information")
"""

import logging
import sys

from src import config


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for the specified module.

    Creates a logger with:
    - Console handler (INFO level): User-facing progress messages
    - File handler (DEBUG level): Detailed debugging in logs/
    - Consistent formatting with timestamps, level, module, message

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing query: %s", query)
        >>> logger.debug("Intermediate result: %s", result)
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured (avoid duplicate handlers)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter

    # Log format: 2026-02-11 16:30:45 | INFO | src.data.generator | Message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (DEBUG and above)
    log_file = config.Paths.LOGS_DIR / "rl_meter_analyst.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False

    return logger
