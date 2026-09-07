"""Versioned named-vector Qdrant collection; legacy collection remains untouched."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from backend.citations.formatter import citation_label
from backend.config import Settings
from backend.embeddings.text_embeddings import embed_texts, text_dimension
from backend.embeddings.visual_embeddings import (
    embed_images,
    embed_visual_texts,
    visual_dimension,
)
from backend.ingestion.element_schema import DocumentElement
from backend.retrieval.fusion import (
    deduplicate_hits,
    modality_diverse,
    reciprocal_rank_fusion,
)
from backend.retrieval.models import RetrievalHit


def session_filter(session_id: str) -> Filter:
    return Filter(
        must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
    )


class MultimodalStore:
    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings, self.client = (
            settings,
            client
            or QdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30
            ),
        )

    def ensure_collection(
        self, text_dim: int | None = None, visual_dim: int | None = None
    ) -> None:
        name = self.settings.multimodal_collection
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config={
                    "text_dense": VectorParams(
                        size=text_dim or text_dimension(), distance=Distance.COSINE
                    ),
                    "visual_dense": VectorParams(
                        size=visual_dim or visual_dimension(), distance=Distance.COSINE
                    ),
                },
            )
            for field in ("session_id", "document_id", "element_id", "parent_id"):
                self.client.create_payload_index(name, field, PayloadSchemaType.KEYWORD)

    def index(
        self, elements: list[DocumentElement], asset_store: Any | None = None
    ) -> None:
        textable = [element for element in elements if element.searchable_text()]
        texts = (
            embed_texts([element.searchable_text() for element in textable])
            if textable
            else []
        )
        text_map = dict(zip((item.element_id for item in textable), texts, strict=True))
        visual_map: dict[str, list[float]] = {}
        if self.settings.enable_image_embeddings and asset_store:
            visual_items = [item for item in elements if item.image_key]
            if visual_items:
                visual_map = dict(
                    zip(
                        (item.element_id for item in visual_items),
                        embed_images(
                            [asset_store.get(item.image_key) for item in visual_items]
                        ),
                        strict=True,
                    )
                )
        self.ensure_collection(
            len(texts[0]) if texts else None,
            len(next(iter(visual_map.values()))) if visual_map else None,
        )
        points = []
        for element in elements:
            vectors = {}
            if element.element_id in text_map:
                vectors["text_dense"] = text_map[element.element_id]
            if element.element_id in visual_map:
                vectors["visual_dense"] = visual_map[element.element_id]
            if vectors:
                points.append(
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, element.element_id)),
                        vector=vectors,
                        payload=element.payload(),
                    )
                )
        if points:
            self.client.upsert(
                self.settings.multimodal_collection, points=points, wait=True
            )

    def _query(
        self, vector: list[float], using: str, session_id: str, limit: int
    ) -> list[RetrievalHit]:
        response = self.client.query_points(
            collection_name=self.settings.multimodal_collection,
            query=vector,
            using=using,
            query_filter=session_filter(session_id),
            limit=limit,
            with_payload=True,
        )
        result = []
        for index, point in enumerate(response.points, 1):
            p = point.payload or {}
            result.append(
                RetrievalHit(
                    element_id=p["element_id"],
                    content_type=p.get("content_type", "text"),
                    page_number=p.get("page_number", 1),
                    score=float(getattr(point, "score", 0)),
                    rank=index,
                    retrieval_sources=[using],
                    text=p.get("text") or p.get("table_markdown") or p.get("caption"),
                    image_key=p.get("image_key"),
                    citation_label=citation_label(p),
                    document_id=p.get("document_id", ""),
                    parent_id=p.get("parent_id"),
                    section_title=p.get("section_title"),
                    figure_number=p.get("figure_number"),
                    table_number=p.get("table_number"),
                )
            )
        return result

    def search(self, query: str, session_id: str, limit: int = 6) -> list[RetrievalHit]:
        rankings = {
            "text_dense": self._query(
                embed_texts([query])[0], "text_dense", session_id, limit * 3
            )
        }
        if self.settings.enable_image_embeddings:
            rankings["visual_dense"] = self._query(
                embed_visual_texts([query])[0], "visual_dense", session_id, limit * 3
            )
        return modality_diverse(
            deduplicate_hits(reciprocal_rank_fusion(rankings)), limit
        )

    def delete_session(self, session_id: str) -> None:
        """Erase new-schema vectors for one browser session only."""
        from qdrant_client.models import FilterSelector

        if self.client.collection_exists(self.settings.multimodal_collection):
            self.client.delete(
                collection_name=self.settings.multimodal_collection,
                points_selector=FilterSelector(filter=session_filter(session_id)),
                wait=True,
            )
