import json
from pathlib import Path
import torch

# ------------------- Paths -------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configurations" / "config.json"

# ------------------- Load JSON -------------------
with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

# ------------------- Device -------------------
DEVICE = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0

# ------------------- Model & Data -------------------
MODEL_NAME = cfg["model_name"]
TRAIN_FILE = ROOT_DIR / cfg["train_file"]
EVAL_FILE = ROOT_DIR / cfg["eval_file"]
MAX_LENGTH = cfg.get("max_length", 512)

# ------------------- LoRA -------------------
LORA_R = cfg["lora"]["r"]
LORA_ALPHA = cfg["lora"]["lora_alpha"]
LORA_TARGET_MODULES = cfg["lora"]["target_modules"]
LORA_DROPOUT = cfg["lora"]["lora_dropout"]
LORA_BIAS = cfg["lora"]["bias"]
LORA_TASK_TYPE = cfg["lora"]["task_type"]

# ------------------- Training Args -------------------
TRAINING_ARGS = cfg["training"]

# ------------------- Inference -------------------
MODE = cfg["inference"]["mode"]
BASE_MODEL_NAME = cfg["inference"]["base_model_name"]
ADAPTER_PATH = ROOT_DIR / cfg["inference"]["adapter_path"]
MERGED_MODEL_PATH = ROOT_DIR / cfg["inference"]["merged_model_path"]
GENERATION_PARAMS = cfg["inference"]["generation"]
QUESTIONS = cfg["inference"]["questions"]
