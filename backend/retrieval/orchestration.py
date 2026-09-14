"""Canonical, failure-tolerant retrieval orchestration for uploaded documents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from backend.citations.formatter import citation_label
from backend.config import Settings
from backend.retrieval.fusion import deduplicate_hits, reciprocal_rank_fusion
from backend.retrieval.models import RetrievalHit
from backend.retrieval.multimodal_store import MultimodalStore
from backend.vector_store import rerank_documents, search_candidates

logger = logging.getLogger(__name__)


def _legacy_hit(document: Document, rank: int) -> RetrievalHit:
    metadata = dict(document.metadata)
    page = metadata.get("page_number") or metadata.get("page") or 1
    if isinstance(page, int) and metadata.get("page") == page:
        page += 1
    document_id = str(metadata.get("document_id") or "")
    element_id = str(
        metadata.get("element_id")
        or metadata.get("point_id")
        or f"legacy:{document_id}:{metadata.get('parent_id') or metadata.get('chunk_index', rank)}"
    )
    return RetrievalHit(
        element_id=element_id,
        content_type=str(metadata.get("content_type") or "text"),
        page_number=int(page) if isinstance(page, int) else 1,
        score=float(metadata.get("retrieval_score") or 0),
        rank=rank,
        retrieval_sources=["legacy_hybrid"],
        text=document.page_content,
        citation_label=str(metadata.get("citation_label") or citation_label(metadata)),
        document_id=document_id,
        parent_id=metadata.get("parent_id"),
        source_filename=metadata.get("document_title") or metadata.get("title"),
        source_url=metadata.get("source_url"),
        original_ranks={"legacy_hybrid": rank},
        original_scores={"legacy_hybrid": float(metadata.get("retrieval_score") or 0)},
        metadata=metadata,
    )


def _rerank_textual_hits(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Replace only text slots, leaving visual-only fused evidence in place."""
    textual = [hit for hit in hits if hit.text and hit.text.strip()]
    if not textual:
        return hits
    documents = [
        Document(
            page_content=hit.text or "", metadata={"evidence_key": hit.evidence_key()}
        )
        for hit in textual
    ]
    reranked = rerank_documents(query, documents, len(documents))
    by_key = {hit.evidence_key(): hit for hit in textual}
    reranked_hits = [
        by_key[document.metadata["evidence_key"]]
        for document in reranked
        if document.metadata.get("evidence_key") in by_key
    ]
    iterator = iter(reranked_hits)
    selected = [
        next(iterator) if hit.text and hit.text.strip() else hit for hit in hits
    ]
    return [
        hit.model_copy(update={"rank": index}) for index, hit in enumerate(selected, 1)
    ]


def _as_answer_document(hit: RetrievalHit) -> Document:
    """Expand legacy parent pages while retaining the matched child for debugging."""
    metadata = dict(hit.metadata)
    metadata.update(
        {
            "document_id": hit.document_id,
            "element_id": hit.element_id,
            "parent_id": hit.parent_id,
            "page_number": hit.page_number,
            "content_type": hit.content_type,
            "citation_label": hit.citation_label,
            "retrieval_sources": hit.retrieval_sources,
            "retrieval_rank": hit.rank,
            "retrieval_score": hit.score,
            "original_ranks": hit.original_ranks,
            "original_scores": hit.original_scores,
            "source_url": hit.source_url,
        }
    )
    parent_content = metadata.get("parent_content")
    if parent_content:
        metadata["matched_chunk"] = hit.text
    # An empty page_content deliberately represents visual-only evidence: we do
    # not fabricate a caption or textual description that was not retrieved.
    return Document(
        page_content=str(parent_content or hit.text or ""), metadata=metadata
    )


def retrieve_uploaded_evidence(
    queries: list[str],
    session_id: str,
    settings: Settings,
    k: int = 4,
    filters: dict[str, Any] | None = None,
    multimodal_factory: Callable[[Settings], MultimodalStore] = MultimodalStore,
) -> list[Document]:
    """Fuse legacy hybrid and multimodal candidates without either bypassing the other."""
    unique_queries = list(
        dict.fromkeys(query.strip() for query in queries if query.strip())
    )
    if not unique_queries:
        return []

    rankings: dict[str, list[RetrievalHit]] = {}
    try:
        legacy = search_candidates(unique_queries, session_id, k, filters)
        rankings["legacy_hybrid"] = [
            _legacy_hit(document, rank) for rank, document in enumerate(legacy, 1)
        ]
    except Exception as exc:  # noqa: BLE001 - multimodal can still provide evidence.
        logger.warning("Legacy hybrid retrieval failed (%s)", type(exc).__name__)

    try:
        multimodal = multimodal_factory(settings).search_many(
            unique_queries, session_id, limit=max(k * 3, 12), filters=filters
        )
        rankings["multimodal"] = multimodal
    except Exception as exc:  # noqa: BLE001 - the hybrid path remains available.
        logger.warning("Multimodal retrieval failed (%s)", type(exc).__name__)

    fused = deduplicate_hits(reciprocal_rank_fusion(rankings))
    if not fused:
        return []
    selected = _rerank_textual_hits(unique_queries[0], fused)
    return [_as_answer_document(hit) for hit in selected[:k]]
