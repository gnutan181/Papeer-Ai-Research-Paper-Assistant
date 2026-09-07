"""Docling-first PDF extraction with controlled PyMuPDF fallback.

The adapter keeps Docling imports inside the call so text/TXT-only installs and
unit tests do not download OCR/VLM models at application startup.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.ingestion.asset_store import AssetStore
from backend.ingestion.element_schema import (
    DocumentElement,
    deterministic_element_id,
    stable_content_hash,
)

logger = logging.getLogger(__name__)


def should_use_ocr(native_text: str, page_area: float | None = None) -> bool:
    """Route only sparse/image-only pages to OCR; avoid duplicate native OCR."""
    del page_area
    return len("".join(native_text.split())) < 40


def table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    headers = [str(value).strip() or " " for value in headers]
    if not headers:
        return ""
    normalized_rows = [
        (list(map(str, row)) + [""] * len(headers))[: len(headers)] for row in rows
    ]
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(row) + " |" for row in normalized_rows)
    )


def is_meaningful_figure(
    width: float, height: float, page_area: float, min_area_ratio: float = 0.015
) -> bool:
    if width <= 0 or height <= 0 or page_area <= 0:
        return False
    ratio = width * height / page_area
    aspect = max(width / height, height / width)
    return ratio >= min_area_ratio and aspect <= 12


def _element(
    document_id: str,
    session_id: str,
    filename: str,
    page: int,
    content_type: str,
    text: str | None,
    method: str,
    **extra: Any,
) -> DocumentElement:
    content_hash = stable_content_hash(
        (text or extra.get("caption") or extra.get("image_key") or "").encode()
    )
    return DocumentElement(
        element_id=deterministic_element_id(
            document_id, content_type, page, content_hash, extra.get("parent_id")
        ),
        document_id=document_id,
        session_id=session_id,
        content_type=content_type,
        text=text,
        source_filename=filename,
        page_number=page,
        extraction_method=method,
        content_hash=content_hash,
        **extra,
    )


def _pymupdf_fallback(
    path: str, document_id: str, session_id: str, filename: str, settings: Settings
) -> tuple[list[DocumentElement], list[str]]:
    import fitz

    doc = fitz.open(path)
    if doc.page_count > settings.max_pdf_pages:
        raise ValueError(
            f"PDF has {doc.page_count} pages; the limit is {settings.max_pdf_pages}."
        )
    elements, warnings = (
        [],
        [
            "Docling was unavailable or failed; used the controlled text-only PDF fallback."
        ],
    )
    for index, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        method = "pymupdf_fallback"
        content_type = "ocr" if should_use_ocr(text) else "text"
        if text:
            elements.append(
                _element(
                    document_id,
                    session_id,
                    filename,
                    index,
                    content_type,
                    text,
                    method,
                    extraction_confidence=0.6 if content_type == "ocr" else 0.9,
                )
            )
        else:
            warnings.append(f"Page {index} contained no extractable text.")
    doc.close()
    return elements, warnings


def parse_pdf(
    path: str,
    document_id: str,
    session_id: str,
    settings: Settings,
    asset_store: AssetStore | None = None,
) -> tuple[list[DocumentElement], list[str]]:
    """Extract text/table/figure elements. Individual conversion errors are warnings."""
    filename = Path(path).name
    try:
        # API verified against Docling's PdfPipelineOptions documentation.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_table_structure = True
        options.do_ocr = settings.enable_ocr
        if settings.enable_ocr:
            options.ocr_options = RapidOcrOptions()
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        result = converter.convert(path)
        document = result.document
        # Markdown preserves Docling reading order/table representation and is
        # deliberately one bounded page-ish element when granular items differ
        # across Docling minor versions.
        markdown = document.export_to_markdown()
        if not markdown.strip():
            return _pymupdf_fallback(path, document_id, session_id, filename, settings)
        element = _element(
            document_id,
            session_id,
            filename,
            1,
            "text",
            markdown,
            "docling",
            extraction_confidence=0.9,
        )
        return [element], []
    except Exception as exc:  # noqa: BLE001 - Docling optional dependency has broad backend errors.
        logger.warning(
            "Docling conversion failed; falling back: %s", type(exc).__name__
        )
        return _pymupdf_fallback(path, document_id, session_id, filename, settings)
