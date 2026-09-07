"""Citation labels are generated/validated from retrieved metadata only."""

from __future__ import annotations

import re

from backend.ingestion.element_schema import DocumentElement


def citation_label(element: DocumentElement | dict) -> str:
    data = element.model_dump() if isinstance(element, DocumentElement) else element
    title = data.get("document_title") or data.get("source_filename") or "Source"
    label = f"{title}, p. {data.get('page_number', 1)}"
    if data.get("figure_number"):
        label += f", Figure {data['figure_number']}"
    if data.get("table_number"):
        label += f", Table {data['table_number']}"
    return label


def validate_citations(
    answer: str, evidence: list[DocumentElement | dict]
) -> tuple[str, list[str]]:
    allowed = {citation_label(item) for item in evidence}
    seen: list[str] = []

    def retain(match: re.Match[str]) -> str:
        label = match.group(1)
        if label in allowed:
            seen.append(label)
            return match.group(0)
        return ""

    return re.sub(r"\[([^\]]+)\]", retain, answer), seen
