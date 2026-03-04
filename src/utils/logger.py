"""
ACMGS Logger — Centralized logging for all modules.

WHY THIS EXISTS:
- print() statements get lost and can't be filtered
- In production, you need log levels (DEBUG, INFO, WARNING, ERROR)
- A central logger writes to both console AND a log file
- Every module imports this instead of using print()

HOW IT WORKS:
- get_logger(name) creates a named logger for each module
- Logs go to: console (colored) + logs/acmgs.log (file)
- Log levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
"""

import logging
import os
import sys
from datetime import datetime

# Import project paths from config
from config.settings import LOGS_DIR


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Name of the logger (typically the module name, e.g., 'energy_dna')
    level : int
        Minimum log level to capture (default: DEBUG = capture everything)

    Returns
    -------
    logging.Logger
        Configured logger instance

    Usage
    -----
        from src.utils.logger import get_logger
        logger = get_logger("energy_dna")
        logger.info("Training started")
        logger.warning("High reconstruction loss detected")
    """

    # Create the logger with the given name
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    # ── Format: timestamp | level | module | message ──
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console Handler: prints to terminal ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)   # Console shows INFO and above
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File Handler: writes to logs/acmgs.log ──
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "acmgs.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)      # File captures everything
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
