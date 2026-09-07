"""Canonical element contract shared by ingestion, storage, and citations."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ContentType = Literal["text", "ocr", "table", "figure", "chart", "diagram"]


def stable_content_hash(value: bytes | str) -> str:
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.encode("utf-8")
    ).hexdigest()


def deterministic_element_id(
    document_id: str,
    content_type: str,
    page_number: int,
    content_hash: str,
    parent_id: str | None = None,
) -> str:
    material = "|".join(
        (document_id, content_type, str(page_number), content_hash, parent_id or "")
    )
    return hashlib.sha256(material.encode()).hexdigest()


class DocumentElement(BaseModel):
    element_id: str
    document_id: str
    session_id: str
    content_type: ContentType
    text: str | None = None
    caption: str | None = None
    image_key: str | None = None
    page_number: int = Field(ge=1)
    section_title: str | None = None
    figure_number: str | None = None
    table_number: str | None = None
    bbox: list[float] | None = None
    source_filename: str
    source_url: str | None = None
    extraction_method: str
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    content_hash: str
    parent_id: str | None = None
    table_markdown: str | None = None
    table_csv: str | None = None

    @field_validator("bbox")
    @classmethod
    def valid_bbox(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 4:
            raise ValueError("bbox must contain [left, top, right, bottom]")
        return value

    def searchable_text(self) -> str:
        return "\n".join(
            part
            for part in (
                self.section_title,
                self.caption,
                self.text,
                self.table_markdown,
            )
            if part
        ).strip()

    def payload(self) -> dict:
        data = self.model_dump(exclude_none=True)
        # Qdrant payloads only hold metadata/small text, never image bytes.
        return json.loads(json.dumps(data))
