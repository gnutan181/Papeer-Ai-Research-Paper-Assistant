"""Input and SSRF validation before parsers or network clients are invoked."""

from __future__ import annotations

import ipaddress
import mimetypes
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

from backend.config import Settings

PDF_MAGIC = b"%PDF-"
ALLOWED_AUDIO_MIME = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/webm",
    "video/webm",
}


class ValidationError(ValueError):
    pass


def sanitize_filename(name: str) -> str:
    candidate = Path(name).name
    candidate = re.sub(r"[^A-Za-z0-9._ -]", "_", candidate).strip(". ")
    if not candidate:
        raise ValidationError("The file name is invalid.")
    return candidate[:150]


def detect_document_kind(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if data.startswith(PDF_MAGIC):
        return "pdf"
    if suffix in {".txt", ".md", ".markdown"}:
        if b"\x00" in data[:4096]:
            raise ValidationError("Text files may not contain binary content.")
        return "markdown" if suffix in {".md", ".markdown"} else "text"
    raise ValidationError("Only valid PDF, text, and Markdown files are supported.")


def validate_upload(filename: str, data: bytes, settings: Settings) -> str:
    if not data:
        raise ValidationError("The uploaded file is empty.")
    if len(data) > settings.max_file_size_bytes:
        raise ValidationError(
            f"Files must be {settings.max_file_size_mb} MB or smaller."
        )
    sanitize_filename(filename)
    return detect_document_kind(filename, data)


def validate_audio(
    filename: str, data: bytes, mime_type: str | None, settings: Settings
) -> None:
    if not data or len(data) > settings.max_audio_size_bytes:
        raise ValidationError(
            f"Audio must be between 1 byte and {settings.max_audio_size_mb} MB."
        )
    guessed, _ = mimetypes.guess_type(filename)
    if (mime_type or guessed or "").lower() not in ALLOWED_AUDIO_MIME:
        raise ValidationError(
            "Unsupported audio format. Use WAV, MP3, M4A, OGG, or WebM."
        )


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValidationError("Only public HTTPS URLs are allowed.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValidationError("The URL host could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValidationError("Private, local, and metadata addresses are blocked.")
    return url
