"""Central LLM construction with optional Portkey gateway."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from backend.config import Settings, load_settings

PORTKEY_GATEWAY_URL = "https://api.portkey.ai/v1"
logger = logging.getLogger(__name__)


def _direct_groq(settings: Settings) -> BaseChatModel:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        timeout=45,
        max_retries=2,
    )


def make_llm(settings: Settings | None = None) -> BaseChatModel:
    """Use Portkey only with a saved provider; otherwise safely use Groq."""
    settings = settings or load_settings()
    provider = settings.portkey_provider
    if not settings.portkey_api_key:
        return _direct_groq(settings)
    if not provider.startswith("@"):
        logger.warning(
            "Portkey disabled: PORTKEY_PROVIDER must be a saved @slug; using direct Groq."
        )
        return _direct_groq(settings)

    # Portkey exposes an OpenAI-compatible API, so ChatOpenAI is only
    # acting as the compatible HTTP client. No OpenAI key is required.
    from langchain_openai import ChatOpenAI

    headers = {
        "x-portkey-api-key": settings.portkey_api_key,
        "x-portkey-provider": provider,
        "x-portkey-metadata": (
            '{"_environment":"production","service":"papeer"}'
        ),
    }

    if settings.portkey_config:
        headers["x-portkey-config"] = settings.portkey_config

    return ChatOpenAI(
        model=settings.groq_model,

        # ChatOpenAI requires a key value; Portkey authenticates using its
        # x-portkey-api-key header, not an OpenAI key.
        api_key=settings.portkey_api_key,

        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers,
        timeout=45,

        # Prefer Portkey gateway retries to avoid multiplying retries.
        max_retries=0,
    )
