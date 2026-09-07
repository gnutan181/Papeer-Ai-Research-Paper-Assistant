from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalHit(BaseModel):
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
