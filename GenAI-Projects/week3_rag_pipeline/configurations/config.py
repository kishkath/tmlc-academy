"""
Load and expose configuration variables from config.json
"""

import json
import os
import loggers

# Path to config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    """Load and validate configuration file."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in config file: {str(e)}")

# Load the config
config = load_config()

# ------------------------------
# Dataset
# ------------------------------
DATASET_NAME = config["dataset"]["name"]
DATASET_SPLIT = config["dataset"]["split"]

# ------------------------------
# Embedding model
# ------------------------------
EMBEDDING_MODEL_NAME = config["embedding"]["model_name"]

# ------------------------------
# Chroma settings
# ------------------------------
CHROMA_COLLECTION_NAME = config["chroma"]["collection_name"]
CHROMA_PERSIST_DIR = config["chroma"]["persist_directory"]
CHROMA_BATCH_SIZE = config["chroma"].get("batch_size", 5000)

# ------------------------------
# LLM settings
# ------------------------------
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = config["llm"]["model_name"]
LLM_TEMPERATURE = config["llm"]["temperature"]
OPENAI_API_KEY = config["llm"]["api_key"]

# ------------------------------
# Retriever settings
# ------------------------------
TOP_K = config["retriever"].get("top_k", 3)

# ------------------------------
# Logging setup
# ------------------------------
# loggers.basicConfig(
#     level=getattr(loggers, config["loggers"]["level"], loggers.INFO),
#     format=config["loggers"]["format"]
# )

# Logging defaults (used in core/logger.py)
LOG_LEVEL = config.get("loggers", {}).get("level", "INFO")
LOG_FORMAT = config.get("loggers", {}).get("format", "%(asctime)s - %(levelname)s - %(message)s")
