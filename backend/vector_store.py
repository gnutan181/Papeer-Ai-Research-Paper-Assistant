"""Session-isolated storage in one Qdrant collection.

Qdrant collections are application-level resources, not chat-session resources.
Every point carries the session identifier and an expiry timestamp so anonymous
documents remain isolated and can be removed without retaining user accounts.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from backend.config import Settings, load_settings

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
DOCUMENT_TTL = timedelta(hours=24)

_collection_lock = threading.Lock()
_embeddings: HuggingFaceEmbeddings | None = None
_client: QdrantClient | None = None


def document_id_for_bytes(content: bytes) -> str:
    """Return a stable ID used to detect a repeated upload in one session."""
    return hashlib.sha256(content).hexdigest()


def _settings() -> Settings:
    return load_settings()


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        settings = _settings()
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
    return _client


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _collection_name() -> str:
    return _settings().qdrant_collection


def _ensure_collection() -> None:
    """Create and verify the sole collection before reading or writing points."""
    client = _get_client()
    collection_name = _collection_name()
    with _collection_lock:
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            for field_name, field_type in (
                ("session_id", PayloadSchemaType.KEYWORD),
                ("document_id", PayloadSchemaType.KEYWORD),
                ("expires_at", PayloadSchemaType.DATETIME),
            ):
                client.create_payload_index(collection_name, field_name, field_type)
            return

        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors
        configured_dimension = (
            vectors.size if isinstance(vectors, VectorParams) else None
        )
        if configured_dimension != EMBEDDING_DIM:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' uses {configured_dimension} dimensions; "
                f"{EMBEDDING_MODEL} requires {EMBEDDING_DIM}. Create or migrate the collection first."
            )


def _session_filter(session_id: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    )


def _page_number(document: Document) -> int | None:
    page = document.metadata.get("page")
    return page + 1 if isinstance(page, int) else document.metadata.get("page_number")


def _contextual_content(document: Document) -> str:
    """Text embedded and ranked for retrieval, keeping display text pristine."""
    return str(document.metadata.get("contextual_content") or document.page_content)


def _metadata_filter(session_id: str, filters: dict[str, Any] | None = None) -> Filter:
    conditions = list(_session_filter(session_id).must or [])
    filters = filters or {}
    if title := filters.get("document_title"):
        conditions.append(FieldCondition(key="document_title", match=MatchValue(value=title)))
    if source_type := filters.get("source_type"):
        conditions.append(FieldCondition(key="source_type", match=MatchValue(value=source_type)))
    if page_number := filters.get("page_number"):
        conditions.append(FieldCondition(key="page_number", match=MatchValue(value=page_number)))
    return Filter(must=conditions)


def _point_to_document(point: Any) -> Document:
    payload = point.payload or {}
    return Document(
        page_content=str(payload.get("content", "")),
        metadata={key: value for key, value in payload.items() if key not in {"content", "contextual_content"}},
    )


def add_paper(
    docs: list[Document],
    session_id: str,
    document_id: str,
    document_title: str,
    source_type: str = "pdf",
) -> bool:
    """Store chunks once, returning ``False`` when the same file is already present."""
    if not docs:
        return False
    _ensure_collection()
    client = _get_client()
    collection_name = _collection_name()
    duplicate_filter = Filter(
        must=[
            FieldCondition(key="session_id", match=MatchValue(value=session_id)),
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
        ]
    )
    existing, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=duplicate_filter,
        limit=1,
        with_payload=False,
    )
    if existing:
        return False

    expires_at = datetime.now(UTC) + DOCUMENT_TTL
    vectors = _get_embeddings().embed_documents(
        [_contextual_content(document) for document in docs]
    )
    points = []
    for chunk_index, (document, vector) in enumerate(zip(docs, vectors, strict=True)):
        # A UUID5 gives retries a stable point ID without leaking the file hash.
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{document_id}:{chunk_index}")
        )
        source_url = document.metadata.get("url") or document.metadata.get("source")
        if not isinstance(source_url, str) or not source_url.startswith(("https://", "http://")):
            source_url = None
        payload = {
            "content": document.page_content,
            "contextual_content": _contextual_content(document),
            "session_id": session_id,
            "document_id": document_id,
            "document_title": document_title,
            "page_number": _page_number(document),
            "chunk_index": chunk_index,
            "source_type": source_type,
            "parent_id": document.metadata.get("parent_id"),
            "parent_content": document.metadata.get("parent_content"),
            "source_url": source_url,
            "expires_at": expires_at.isoformat(),
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))
    client.upsert(collection_name=collection_name, points=points, wait=True)
    return True


def list_papers(session_id: str) -> list[str]:
    _ensure_collection()
    points, _ = _get_client().scroll(
        collection_name=_collection_name(),
        scroll_filter=_session_filter(session_id),
        with_payload=["document_title"],
        limit=1_000,
    )
    return list(
        dict.fromkeys(
            point.payload["document_title"]
            for point in points
            if point.payload and point.payload.get("document_title")
        )
    )


def _dense_search(
    query: str, session_id: str, filters: dict[str, Any] | None, limit: int
) -> list[Document]:
    response = _get_client().query_points(
        collection_name=_collection_name(),
        query=_get_embeddings().embed_query(query),
        query_filter=_metadata_filter(session_id, filters),
        limit=limit,
        with_payload=True,
    )
    return [_point_to_document(point) for point in response.points if point.payload]


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.lower())


def _lexical_search(
    query: str, session_id: str, filters: dict[str, Any] | None, limit: int
) -> list[Document]:
    """Per-session BM25 over payload text.

    Anonymous sessions are deliberately bounded in the UI; this avoids a
    global inverted index that could leak or retain document terms across
    users. Qdrant remains the durable source of truth.
    """
    points, _ = _get_client().scroll(
        collection_name=_collection_name(),
        scroll_filter=_metadata_filter(session_id, filters),
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return []
    query_terms = _tokens(query)
    if not query_terms:
        return []
    documents = [_point_to_document(point) for point in points if point.payload]
    corpus = [_tokens(str(point.payload.get("contextual_content", ""))) for point in points if point.payload]
    total = len(corpus)
    average_length = sum(map(len, corpus)) / max(total, 1)
    frequencies: dict[str, int] = {}
    for terms in corpus:
        for term in set(terms):
            frequencies[term] = frequencies.get(term, 0) + 1
    scored: list[tuple[float, int]] = []
    for index, terms in enumerate(corpus):
        term_counts: dict[str, int] = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            if not (count := term_counts.get(term)):
                continue
            idf = math.log(1 + (total - frequencies.get(term, 0) + 0.5) / (frequencies.get(term, 0) + 0.5))
            score += idf * (count * 2.0) / (count + 1.2 * (1 - 0.75 + 0.75 * len(terms) / max(average_length, 1)))
        if score:
            scored.append((score, index))
    return [documents[index] for _, index in sorted(scored, reverse=True)[:limit]]


def _rrf(rankings: list[list[Document]], limit: int, k: int = 60) -> list[Document]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for position, document in enumerate(ranking, start=1):
            key = str(document.metadata.get("point_id") or document.metadata.get("parent_id") or (document.metadata.get("document_id"), document.metadata.get("chunk_index")))
            scores[key] = scores.get(key, 0.0) + 1 / (k + position)
            docs[key] = document
    return [docs[key] for key, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


_reranker: Any | None = None
_reranker_lock = threading.Lock()


def _cross_encoder_rerank(query: str, docs: list[Document], limit: int) -> list[Document]:
    """Rerank candidates, degrading safely when model download is unavailable."""
    global _reranker
    if not docs:
        return []
    try:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder

                _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = _reranker.predict([(query, _contextual_content(doc)) for doc in docs])
        return [doc for _, doc in sorted(zip(scores, docs, strict=True), key=lambda item: item[0], reverse=True)[:limit]]
    except (ImportError, OSError, RuntimeError, ValueError):
        # Retrieval remains available in restricted/offline deployments.
        return docs[:limit]


def _parents_for_matches(matches: list[Document], limit: int) -> list[Document]:
    """Return page-sized parent documents while retaining child-hit provenance."""
    parents: list[Document] = []
    seen: set[str] = set()
    for match in matches:
        parent_id = str(match.metadata.get("parent_id") or f"{match.metadata.get('document_id')}:{match.metadata.get('chunk_index')}")
        if parent_id in seen:
            continue
        seen.add(parent_id)
        parent_content = match.metadata.get("parent_content")
        content = str(parent_content or match.page_content)
        metadata = dict(match.metadata)
        metadata["matched_chunk"] = match.page_content
        metadata["citation_label"] = _citation_label(metadata)
        parents.append(Document(page_content=content, metadata=metadata))
        if len(parents) >= limit:
            break
    return parents


def _citation_label(metadata: dict[str, Any]) -> str:
    title = str(metadata.get("document_title") or "Source")
    page = metadata.get("page_number")
    return f"{title}, p. {page}" if page else title


def search_many(
    queries: list[str], session_id: str, k: int = 4, filters: dict[str, Any] | None = None
) -> list[Document]:
    """Multi-query hybrid retrieval: dense + BM25 → RRF → cross encoder → parents."""
    _ensure_collection()
    rankings: list[list[Document]] = []
    for query in dict.fromkeys(q.strip() for q in queries if q.strip()):
        rankings.append(_dense_search(query, session_id, filters, max(k * 4, 12)))
        rankings.append(_lexical_search(query, session_id, filters, max(k * 4, 12)))
    fused = _rrf(rankings, limit=max(k * 6, 24))
    return _parents_for_matches(_cross_encoder_rerank(queries[0], fused, max(k * 3, 12)), k)


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Backward-compatible single-query entry point for hybrid retrieval."""
    return search_many([query], session_id=session_id, k=k)


def delete_session_documents(session_id: str) -> None:
    """Immediately erase all vectors owned by the browser session."""
    _ensure_collection()
    _get_client().delete(
        collection_name=_collection_name(),
        points_selector=_session_filter(session_id),
        wait=True,
    )


def delete_expired_documents() -> None:
    """Delete anonymous documents whose 24-hour retention period has elapsed."""
    _ensure_collection()
    expired = Filter(
        must=[
            FieldCondition(key="expires_at", range=DatetimeRange(lt=datetime.now(UTC)))
        ]
    )
    _get_client().delete(
        collection_name=_collection_name(), points_selector=expired, wait=True
    )
