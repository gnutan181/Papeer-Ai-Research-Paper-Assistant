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


_SOURCE_ID = re.compile(r"S\d+")
# A bracket followed by "(" is a markdown link, not a citation.
_BRACKET = re.compile(r"\[([^\]]+)\](?!\()")


def _accepted_labels(element: DocumentElement | dict) -> set[str]:
    """Every form in which an answer may cite this piece of evidence.

    Retrieved chunks already carry the label shown to the model
    (``citation_label``), and the answer prompt cites sources by position
    (``citation_id`` such as "S1"); both are accepted alongside the label
    rebuilt from title and page.
    """
    data = element.model_dump() if isinstance(element, DocumentElement) else element
    labels = {citation_label(data)}
    for key in ("citation_label", "citation_id"):
        if data.get(key):
            labels.add(str(data[key]))
    return labels


def validate_citations(
    answer: str, evidence: list[DocumentElement | dict]
) -> tuple[str, list[str]]:
    allowed: set[str] = set()
    for item in evidence:
        allowed |= _accepted_labels(item)
    seen: list[str] = []

    def retain(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        if label in allowed:
            seen.append(label)
            return match.group(0)
        # Grouped source ids such as [S1, S3]: keep only the real ones.
        parts = [part.strip() for part in re.split(r"[,;]", label)]
        if len(parts) > 1 and all(_SOURCE_ID.fullmatch(part) for part in parts):
            kept = [part for part in parts if part in allowed]
            seen.extend(kept)
            return f"[{', '.join(kept)}]" if kept else ""
        return ""

    return _BRACKET.sub(retain, answer), seen
