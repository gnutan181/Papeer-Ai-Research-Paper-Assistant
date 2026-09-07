"""Groq Whisper adapter with bounded retryable failure handling."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

from backend.config import Settings
from backend.ingestion.validators import validate_audio


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transcript:
    text: str
    model: str


def normalize_chat_input(value: object) -> tuple[str | None, object | None]:
    """Support Streamlit's str and ChatInputValue return types without internals."""
    if isinstance(value, str):
        return value.strip() or None, None
    return (
        str(getattr(value, "text", "") or "").strip() or None,
        getattr(value, "audio", None),
    )


def transcribe_audio(
    data: bytes,
    filename: str,
    mime_type: str | None,
    settings: Settings,
    accuracy_sensitive: bool = False,
    client: object | None = None,
) -> Transcript:
    validate_audio(filename, data, mime_type, settings)
    model = "whisper-large-v3" if accuracy_sensitive else "whisper-large-v3-turbo"
    if client is None:
        try:
            from groq import Groq

            client = Groq(api_key=settings.groq_api_key, timeout=30)
        except ImportError as exc:
            raise TranscriptionError(
                "Speech transcription is not installed in this deployment."
            ) from exc
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.audio.transcriptions.create(
                file=(filename, io.BytesIO(data)),
                model=model,
                response_format="json",
                temperature=0.0,
            )
            text = str(getattr(response, "text", "")).strip()
            if not text:
                raise TranscriptionError("No speech was detected in that recording.")
            return Transcript(text=text, model=model)
        except Exception as exc:  # noqa: BLE001 - provider SDK has many transport exception types
            last_error = exc
            if attempt == 0 and ("429" in str(exc) or "timeout" in str(exc).lower()):
                time.sleep(0.4)
                continue
            break
    raise TranscriptionError(
        "Transcription is temporarily unavailable. Please type your question."
    ) from last_error
