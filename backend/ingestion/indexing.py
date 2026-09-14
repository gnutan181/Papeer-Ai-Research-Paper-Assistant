"""Keep reliable text indexing independent from optional multimodal indexing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from backend.config import Settings
from backend.ingestion.element_schema import DocumentElement
from backend.retrieval.multimodal_store import MultimodalStore
from backend.vector_store import add_paper

logger = logging.getLogger(__name__)


def index_uploaded_document(
    docs: list[Document],
    elements: list[DocumentElement],
    asset_store: Any,
    session_id: str,
    document_id: str,
    document_title: str,
    source_type: str,
    settings: Settings,
    legacy_indexer: Callable[..., bool] = add_paper,
    multimodal_factory: Callable[[Settings], MultimodalStore] = MultimodalStore,
) -> tuple[bool, str | None]:
    """Index text first; report optional multimodal failure without losing text."""
    inserted = legacy_indexer(
        docs=docs,
        session_id=session_id,
        document_id=document_id,
        document_title=document_title,
        source_type=source_type,
    )
    if not inserted:
        return False, None
    try:
        multimodal_factory(settings).index(
            elements,
            asset_store,
            document_title=document_title,
            source_type=source_type,
        )
    except Exception as exc:  # noqa: BLE001 - optional index must not strand text.
        logger.warning("Optional multimodal indexing failed (%s)", type(exc).__name__)
        return True, "Multimodal indexing was unavailable; text search is ready."
    return True, None
