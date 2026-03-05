"""
TARS Logging Configuration — Centralized setup for all modules.

Usage:
    from utils.logging_config import setup_logging
    setup_logging(level="INFO")  # Call once at startup
    
All TARS loggers (Tars.Agent, Tars.GIE, Tars.MoIRA, Tars.LEANN, etc.)
will use consistent formatting and output.
"""
import logging
import sys
import os
from datetime import datetime


# All known TARS logger names
TARS_LOGGERS = [
    "Tars",
    "Tars.Agent",
    "Tars.GIE",
    "Tars.MoIRA",
    "Tars.Actions",
    "Tars.LEANN",
    "Tars.Memory",
    "Tars.Titans",
    "Tars.Memo",
    "Tars.Tools",
    "Tars.ToolRegistry",
    "Tars.SubAgents",
    "Tars.Cron",
    "Tars.RRN",
    "Tars.Reflex",
]


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    console: bool = True,
):
    """
    Configure all TARS loggers with consistent formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        console: Whether to output to console (default: True)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Format: timestamp [level] logger — message
    fmt = "%(asctime)s [%(levelname).1s] %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    
    handlers = []
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root Tars logger (all child loggers inherit)
    root_logger = logging.getLogger("Tars")
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    for h in handlers:
        root_logger.addHandler(h)
    
    # Prevent duplicate logs from propagation
    root_logger.propagate = False
    
    root_logger.info(f"Logging configured: level={level}, file={log_file or 'none'}")
