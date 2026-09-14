from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
    """A source-neutral retrieval candidate used before answer-context selection."""

    element_id: str
    content_type: str
    page_number: int
    score: float
    rank: int
    retrieval_sources: list[str] = Field(default_factory=list)
    text: str | None = None
    image_key: str | None = None
    citation_label: str
    document_id: str = ""
    parent_id: str | None = None
    section_title: str | None = None
    figure_number: str | None = None
    table_number: str | None = None
    source_filename: str | None = None
    source_url: str | None = None
    original_ranks: dict[str, int] = Field(default_factory=dict)
    original_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def evidence_key(self) -> str:
        """Stable page-level key shared by the legacy and multimodal schemas."""
        if self.document_id and self.page_number:
            return f"{self.document_id}:page:{self.page_number}"
        if self.parent_id:
            return f"{self.document_id or 'unknown'}:parent:{self.parent_id}"
        return f"{self.document_id}:element:{self.element_id}"
