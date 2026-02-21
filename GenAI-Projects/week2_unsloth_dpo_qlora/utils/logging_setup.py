import logging
import os
from pathlib import Path
from configurations.config import LOG_LEVEL

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "fitness_bot.log"

# Logging Formatter
formatter = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# File Handler
file_handler = logging.FileHandler(log_file, mode="a")
file_handler.setFormatter(formatter)

# Logger
logger = logging.getLogger("FitnessBot")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger.addHandler(console_handler)
logger.addHandler(file_handler)
