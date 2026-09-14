from types import SimpleNamespace

from langchain_core.documents import Document

from backend.config import Settings
from backend.retrieval import orchestration
from backend.retrieval.models import RetrievalHit
from backend.retrieval.multimodal_store import MultimodalStore


def _settings(**overrides):
    values = {
        "groq_api_key": "g",
        "qdrant_url": "https://qdrant.example.test",
        "qdrant_api_key": "q",
        "tavily_api_key": "t",
        "enable_image_embeddings": False,
    }
    values.update(overrides)
    return Settings(**values)


def _legacy(page=1, document_id="paper", text="legacy evidence"):
    return Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "document_title": "Paper",
            "page_number": page,
            "parent_id": f"parent-{page}",
            "parent_content": f"Full parent page {page}",
        },
    )


def _hit(page=2, document_id="paper", text="multimodal evidence", kind="text"):
    return RetrievalHit(
        element_id=f"element-{page}-{kind}",
        document_id=document_id,
        content_type=kind,
        page_number=page,
        text=text,
        score=0.8,
        rank=1,
        citation_label=f"Paper, p. {page}",
        source_filename="Paper",
    )


class Store:
    def __init__(self, hits=None, error=None):
        self.hits = hits or []
        self.error = error
        self.calls = []

    def search_many(self, queries, session_id, limit, filters):
        self.calls.append((queries, session_id, limit, filters))
        if self.error:
            raise self.error
        return self.hits


def _without_model(monkeypatch):
    monkeypatch.setattr(
        orchestration, "rerank_documents", lambda _query, docs, _limit: docs
    )


def test_multimodal_hits_do_not_bypass_legacy_and_all_variants_are_consumed(
    monkeypatch,
):
    _without_model(monkeypatch)
    legacy_calls = []
    store = Store([_hit()])

    def legacy(queries, session_id, k, filters):
        legacy_calls.append((queries, session_id, k, filters))
        return [_legacy()]

    monkeypatch.setattr(orchestration, "search_candidates", legacy)
    docs = orchestration.retrieve_uploaded_evidence(
        ["original", "variant one", "variant two"],
        "session-a",
        _settings(),
        k=3,
        filters={"page_number": 2},
        multimodal_factory=lambda _settings: store,
    )

    assert legacy_calls == [
        (["original", "variant one", "variant two"], "session-a", 3, {"page_number": 2})
    ]
    assert store.calls == [
        (
            ["original", "variant one", "variant two"],
            "session-a",
            12,
            {"page_number": 2},
        )
    ]
    assert {doc.metadata["retrieval_sources"][0] for doc in docs} == {
        "legacy_hybrid",
        "multimodal",
    }


def test_shared_page_is_fused_once_and_parent_citation_metadata_survives(monkeypatch):
    _without_model(monkeypatch)
    monkeypatch.setattr(orchestration, "search_candidates", lambda *_args: [_legacy()])
    docs = orchestration.retrieve_uploaded_evidence(
        ["question"],
        "session-a",
        _settings(),
        multimodal_factory=lambda _settings: Store([_hit(page=1)]),
    )

    assert len(docs) == 1
    assert docs[0].page_content == "Full parent page 1"
    assert docs[0].metadata["matched_chunk"] == "legacy evidence"
    assert docs[0].metadata["citation_label"] == "Paper, p. 1"
    assert docs[0].metadata["retrieval_sources"] == ["legacy_hybrid", "multimodal"]


def test_each_source_degrades_independently_and_empty_is_safe(monkeypatch):
    _without_model(monkeypatch)
    monkeypatch.setattr(orchestration, "search_candidates", lambda *_args: [_legacy()])
    text_docs = orchestration.retrieve_uploaded_evidence(
        ["question"],
        "session-a",
        _settings(),
        multimodal_factory=lambda _settings: Store(error=RuntimeError("offline")),
    )
    assert [doc.page_content for doc in text_docs] == ["Full parent page 1"]

    monkeypatch.setattr(
        orchestration,
        "search_candidates",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    multimodal_docs = orchestration.retrieve_uploaded_evidence(
        ["question"],
        "session-a",
        _settings(),
        multimodal_factory=lambda _settings: Store([_hit()]),
    )
    assert [doc.page_content for doc in multimodal_docs] == ["multimodal evidence"]

    empty_docs = orchestration.retrieve_uploaded_evidence(
        ["question"],
        "session-a",
        _settings(),
        multimodal_factory=lambda _settings: Store(),
    )
    assert empty_docs == []


def test_text_candidates_fill_k_without_a_modality_quota(monkeypatch):
    _without_model(monkeypatch)
    monkeypatch.setattr(
        orchestration,
        "search_candidates",
        lambda *_args: [
            _legacy(page=index, text=f"text {index}") for index in range(1, 5)
        ],
    )
    docs = orchestration.retrieve_uploaded_evidence(
        ["question"],
        "session-a",
        _settings(),
        k=4,
        multimodal_factory=lambda _settings: Store(),
    )
    assert len(docs) == 4
    assert all(doc.metadata["content_type"] == "text" for doc in docs)


class QueryClient:
    def __init__(self):
        self.filters = []

    def query_points(self, **kwargs):
        self.filters.append(kwargs["query_filter"])
        return SimpleNamespace(points=[])


def test_multimodal_queries_keep_session_and_explicit_filters(monkeypatch):
    client = QueryClient()
    store = MultimodalStore(_settings(), client=client)
    monkeypatch.setattr(
        "backend.retrieval.multimodal_store.embed_texts",
        lambda values: [[0.1] * 384 for _ in values],
    )
    store.search_many(
        ["one", "two", "three"],
        "session-a",
        filters={"document_title": "Paper", "source_type": "pdf", "page_number": 3},
    )

    assert len(client.filters) == 3
    for query_filter in client.filters:
        values = {
            condition.key: condition.match.value for condition in query_filter.must
        }
        assert values == {
            "session_id": "session-a",
            "document_title": "Paper",
            "source_type": "pdf",
            "page_number": 3,
        }
