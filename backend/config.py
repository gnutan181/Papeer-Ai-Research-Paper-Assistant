"""Runtime configuration for Papeer.

Secrets are deliberately read only from the environment. Hugging Face Spaces
injects these values through Space secrets; local development may use `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv


class ConfigurationError(RuntimeError):
    """Raised when the application cannot be started safely."""


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    tavily_api_key: str
    groq_model: str = "openai/gpt-oss-20b"
    qdrant_collection: str = "papeer_chunks_v2"


def load_settings() -> Settings:
    """Load and validate required settings before serving any user request."""
    load_dotenv()
    values = {
        "groq_api_key": os.getenv("GROQ_API_KEY", "").strip(),
        "qdrant_url": os.getenv("QDRANT_URL", "").strip(),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", "").strip(),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", "").strip(),
    }
    missing = [name.upper() for name, value in values.items() if not value]
    if missing:
        raise ConfigurationError(
            "Missing required configuration: " + ", ".join(missing)
        )

    parsed = urlparse(values["qdrant_url"])
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigurationError("QDRANT_URL must be a valid HTTPS URL.")

    return Settings(
        **values,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
    )
