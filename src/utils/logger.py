"""Logging configuration and utilities."""

from loguru import logger
from config.settings import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger():
    """Configure logger for NextSight."""
    logger.remove()  # Remove default logger
    
    # Console logger
    logger.add(
        sink=lambda message: print(message, end=""),
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        colorize=True
    )
    
    # File logger
    logger.add(
        sink=LOG_FILE,
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        rotation="1 day",
        retention="7 days",
        compression="zip"
    )
    
    return logger


def get_logger(name=None):
    """Get configured logger instance."""
    if name:
        return logger.bind(name=name)
    return logger