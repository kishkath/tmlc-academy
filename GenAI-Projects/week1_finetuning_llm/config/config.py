import json
import os
from dotenv import load_dotenv

load_dotenv()

# Load settings.json
with open("config/settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

# Combine environment variables with settings file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FINE_TUNE_SUFFIX = settings.get("FINE_TUNE_SUFFIX")
DEFAULT_TIMEOUT = settings.get("DEFAULT_TIMEOUT", 1800)
POLL_INTERVAL = settings.get("POLL_INTERVAL", 30)
