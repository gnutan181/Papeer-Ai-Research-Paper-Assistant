"""Session-isolated storage in one Qdrant collection.

Qdrant collections are application-level resources, not chat-session resources.
Every point carries the session identifier and an expiry timestamp so anonymous
documents remain isolated and can be removed without retaining user accounts.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from datetime import UTC, datetime, timedelta

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
        [document.page_content for document in docs]
    )
    points = []
    for chunk_index, (document, vector) in enumerate(zip(docs, vectors, strict=True)):
        # A UUID5 gives retries a stable point ID without leaking the file hash.
        point_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{document_id}:{chunk_index}")
        )
        payload = {
            "content": document.page_content,
            "session_id": session_id,
            "document_id": document_id,
            "document_title": document_title,
            "page_number": _page_number(document),
            "chunk_index": chunk_index,
            "source_type": source_type,
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


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    _ensure_collection()
    response = _get_client().query_points(
        collection_name=_collection_name(),
        query=_get_embeddings().embed_query(query),
        query_filter=_session_filter(session_id),
        limit=k,
        with_payload=True,
    )
    return [
        Document(
            page_content=str(point.payload.get("content", "")),
            metadata={
                key: value for key, value in point.payload.items() if key != "content"
            },
        )
        for point in response.points
        if point.payload
    ]


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
