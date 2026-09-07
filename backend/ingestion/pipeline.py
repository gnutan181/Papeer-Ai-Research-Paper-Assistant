"""Safe document ingestion entry point with single-flight processing."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import Settings
from backend.ingestion.asset_store import AssetStore, LocalAssetStore, R2AssetStore
from backend.ingestion.docling_parser import parse_pdf
from backend.ingestion.element_schema import (
    DocumentElement,
    deterministic_element_id,
    stable_content_hash,
)
from backend.ingestion.validators import sanitize_filename, validate_upload

logger = logging.getLogger(__name__)
_ingestion_lock = threading.Lock()


@dataclass
class IngestionResult:
    elements: list[DocumentElement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    document_id: str = ""


def make_asset_store(settings: Settings) -> AssetStore:
    if settings.asset_store == "r2" and settings.r2_is_configured:
        return R2AssetStore(
            settings.r2_endpoint_url,
            settings.r2_access_key_id,
            settings.r2_secret_access_key,
            settings.r2_bucket_name,
            settings.r2_public_base_url,
        )
    return LocalAssetStore(settings.local_asset_dir)


def ingest_upload(
    filename: str,
    data: bytes,
    session_id: str,
    settings: Settings,
    asset_store: AssetStore | None = None,
) -> IngestionResult:
    """Validate first, then parse under a one-at-a-time ingestion lock."""
    kind = validate_upload(filename, data, settings)
    filename = sanitize_filename(filename)
    document_id = stable_content_hash(data)
    with _ingestion_lock:
        if kind == "pdf":
            import tempfile

            path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as file:
                    file.write(data)
                    path = file.name
                elements, warnings = parse_pdf(
                    path,
                    document_id,
                    session_id,
                    settings,
                    asset_store or make_asset_store(settings),
                )
                return IngestionResult(elements, warnings, document_id)
            finally:
                if path:
                    Path(path).unlink(missing_ok=True)
        text = data.decode("utf-8", errors="replace")
        content_hash = stable_content_hash(text)
        element = DocumentElement(
            element_id=deterministic_element_id(document_id, "text", 1, content_hash),
            document_id=document_id,
            session_id=session_id,
            content_type="text",
            text=text,
            page_number=1,
            source_filename=filename,
            extraction_method=kind,
            extraction_confidence=1.0,
            content_hash=content_hash,
        )
        return IngestionResult([element], [], document_id)
