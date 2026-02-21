"""
Central logger configuration. Import logger.get_logger() from other modules.
"""
import logging
from configurations.config import LOG_LEVEL, LOG_FORMAT

def get_logger(name: str = "rag_legal_assistant"):
    """Create and return a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger

# convenience
logger = get_logger()
