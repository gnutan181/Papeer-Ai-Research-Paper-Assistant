import pytest

from backend import config
from backend.config import ConfigurationError, load_settings


def test_configuration_requires_all_secrets(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for name in ("GROQ_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        load_settings()


def test_configuration_rejects_non_https_qdrant_url(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.test")
    monkeypatch.setenv("QDRANT_API_KEY", "test")
    monkeypatch.setenv("TAVILY_API_KEY", "test")

    with pytest.raises(ConfigurationError, match="HTTPS"):
        load_settings()
