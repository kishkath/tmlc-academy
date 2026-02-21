import json
import logging
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# Dataset config
TRAIN_FILE = CONFIG.get("train_file")
VAL_FILE = CONFIG.get("val_file")
TRAIN_LIMIT = CONFIG.get("train_limit", 700)
TEST_LIMIT = CONFIG.get("test_limit", 100)

# Model config
MODEL_CONFIG = CONFIG.get("model", {})
MODEL_NAME = MODEL_CONFIG.get("name")
MAX_SEQ_LENGTH = MODEL_CONFIG.get("max_seq_length", 512)
DTYPE = MODEL_CONFIG.get("dtype", None)
LOAD_IN_4BIT = MODEL_CONFIG.get("load_in_4bit", True)
LORA_CONFIG = MODEL_CONFIG.get("lora", {})

# Trainer config
TRAINER_CONFIG = CONFIG.get("trainer", {})

# Inference config
INFERENCE_CONFIG = CONFIG.get("inference", {})
INFERENCE_MODEL_PATH = INFERENCE_CONFIG.get("model_path", "small-qwen-exp")
INFERENCE_MAX_SEQ_LENGTH = INFERENCE_CONFIG.get("max_seq_length", 512)
INFERENCE_LOAD_IN_4BIT = INFERENCE_CONFIG.get("load_in_4bit", True)

# WandB config
WANDB_CONFIG = CONFIG.get("wandb", {})
USE_WANDB = WANDB_CONFIG.get("use", False)
WANDB_PROJECT = WANDB_CONFIG.get("project", "fitness-bot")
WANDB_NAME = WANDB_CONFIG.get("name", "qwen-finetuning")

# API config
API_CONFIG = CONFIG.get("api", {})

# Logging config
LOGGING_CONFIG = CONFIG.get("logging", {})
LOG_LEVEL = LOGGING_CONFIG.get("level", "INFO")
LOG_FILE = LOGGING_CONFIG.get("log_file", "fitness_bot.log")

# Setup logger
logger = logging.getLogger("fitness_bot")
logger.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
log_path = Path(LOG_FILE)
log_path.parent.mkdir(exist_ok=True, parents=True)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
