"""Validated, centralized runtime configuration."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic.dataclasses import dataclass


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or unsafe."""


def _flag(value: str | bool | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _threshold(value: str) -> float:
    threshold = float(value)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("GUARDRAIL_INJECTION_THRESHOLD must be in (0, 1].")
    return threshold


@dataclass(frozen=True)
class Settings:
    """Settings model; the four leading fields retain legacy positional use."""

    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    tavily_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    qdrant_collection: str = "papeer_chunks_v2"
    multimodal_collection: str = "papeer_multimodal_v1"
    portkey_api_key: str = ""
    portkey_provider: str = ""
    portkey_config: str = ""
    asset_store: str = "local"
    local_asset_dir: str = "./data/assets"
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = ""
    r2_public_base_url: str = ""
    enable_ocr: bool = True
    enable_vision: bool = True
    enable_image_embeddings: bool = True
    enable_audio_input: bool = True
    enable_browser_tts: bool = True
    max_file_size_mb: int = 20
    max_pdf_pages: int = 40
    max_figures_per_document: int = 30
    max_audio_size_mb: int = 10
    max_urls_per_request: int = 5
    debug: bool = False
    guardrails_enabled: bool = True
    guardrail_injection_threshold: float = 0.8
    guardrail_timeout_seconds: float = 10.0
    prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    safety_guard_model: str = "openai/gpt-oss-safeguard-20b"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_audio_size_bytes(self) -> int:
        return self.max_audio_size_mb * 1024 * 1024

    @property
    def r2_is_configured(self) -> bool:
        return bool(
            self.r2_endpoint_url
            and self.r2_access_key_id
            and self.r2_secret_access_key
            and self.r2_bucket_name
        )


def load_settings() -> Settings:
    """Load required core secrets and optional feature settings from the environment."""
    load_dotenv()
    values = {
        name: os.getenv(env, "").strip()
        for name, env in {
            "groq_api_key": "GROQ_API_KEY",
            "qdrant_url": "QDRANT_URL",
            "qdrant_api_key": "QDRANT_API_KEY",
            "tavily_api_key": "TAVILY_API_KEY",
        }.items()
    }
    missing = [name.upper() for name, value in values.items() if not value]
    if missing:
        raise ConfigurationError(
            "Missing required configuration: " + ", ".join(missing)
        )
    parsed = urlparse(values["qdrant_url"])
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigurationError("QDRANT_URL must be a valid HTTPS URL.")
    try:
        return Settings(
            **values,
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", "papeer_chunks_v2"
            ).strip(),
            multimodal_collection=os.getenv(
                "MULTIMODAL_QDRANT_COLLECTION", "papeer_multimodal_v1"
            ).strip(),
            portkey_api_key=os.getenv("PORTKEY_API_KEY", "").strip(),
            portkey_provider=os.getenv("PORTKEY_PROVIDER", "").strip(),
            portkey_config=os.getenv("PORTKEY_CONFIG", "").strip(),
            asset_store=os.getenv("ASSET_STORE", "local").strip().lower(),
            local_asset_dir=os.getenv("LOCAL_ASSET_DIR", "./data/assets").strip(),
            r2_endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip(),
            r2_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            r2_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            r2_bucket_name=os.getenv("R2_BUCKET_NAME", "").strip(),
            r2_public_base_url=os.getenv("R2_PUBLIC_BASE_URL", "").strip(),
            enable_ocr=_flag(os.getenv("ENABLE_OCR"), True),
            enable_vision=_flag(os.getenv("ENABLE_VISION"), True),
            enable_image_embeddings=_flag(os.getenv("ENABLE_IMAGE_EMBEDDINGS"), True),
            enable_audio_input=_flag(os.getenv("ENABLE_AUDIO_INPUT"), True),
            enable_browser_tts=_flag(os.getenv("ENABLE_BROWSER_TTS"), True),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "20")),
            max_pdf_pages=int(os.getenv("MAX_PDF_PAGES", "40")),
            max_figures_per_document=int(os.getenv("MAX_FIGURES_PER_DOCUMENT", "30")),
            max_audio_size_mb=int(os.getenv("MAX_AUDIO_SIZE_MB", "10")),
            max_urls_per_request=int(os.getenv("MAX_URLS_PER_REQUEST", "5")),
            debug=_flag(os.getenv("DEBUG")),
            guardrails_enabled=_flag(os.getenv("GUARDRAILS_ENABLED"), True),
            guardrail_injection_threshold=_threshold(
                os.getenv("GUARDRAIL_INJECTION_THRESHOLD", "0.8")
            ),
            guardrail_timeout_seconds=float(
                os.getenv("GUARDRAIL_TIMEOUT_SECONDS", "10")
            ),
            prompt_guard_model=os.getenv(
                "PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m"
            ).strip(),
            safety_guard_model=os.getenv(
                "SAFETY_GUARD_MODEL", "openai/gpt-oss-safeguard-20b"
            ).strip(),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Invalid Papeer configuration value.") from exc
