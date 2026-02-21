import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file into process env.
# This allows local development without exporting variables manually.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration container for the application.

    Why frozen=True?
    - Prevents accidental mutation at runtime
    - Makes behavior deterministic across Streamlit reruns
    - Safe to share as a global singleton
    """

    openai_api_key: str = os.environ.get(
        "OPENAI_API_KEY", ""
    )  # OpenAI API key used by OpenAITriageClient.
    openai_model: str = os.environ.get(
        "OPENAI_MODEL", "gpt-4.1-mini"
    )  # Default LLM model name.
    db_path: str = os.environ.get(
        "TRIAGE_DB_PATH", os.path.join("data", "triage.db")
    )  # SQLite DB path for triage system.


# Global, read-only settings instance.
# Safe to import anywhere without side effects.
SETTINGS = Settings()
