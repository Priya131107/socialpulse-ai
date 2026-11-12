"""
Structured logger for SocialPulse AI.
Provides consistent formatting across CLI, API, background tasks, and dashboards.
"""

import logging
import sys
from pathlib import Path

def setup_logger(name: str = "socialpulse", level: str = "INFO") -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Default logger instance
logger = setup_logger("socialpulse")

