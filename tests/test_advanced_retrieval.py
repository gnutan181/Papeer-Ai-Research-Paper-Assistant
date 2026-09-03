from types import SimpleNamespace

from langchain_core.documents import Document

from backend import vector_store
from backend.paper_loader import _contextual_chunks, set_document_title


class HybridClient:
    def __init__(self):
        self.filters = []
        self.points = [
            SimpleNamespace(
                payload={
                    "content": "Attention uses token interactions.",
                    "contextual_content": "Document: Transformer study\nAttention uses token interactions.",
                    "parent_content": "Full PDF page about attention and token interactions.",
                    "parent_id": "page-1",
                    "document_id": "paper-1",
                    "document_title": "Transformer study",
                    "page_number": 3,
                    "source_type": "pdf",
                }
            ),
            SimpleNamespace(
                payload={
                    "content": "An unrelated appendix.",
                    "contextual_content": "Document: Transformer study\nAn unrelated appendix.",
                    "parent_content": "Appendix page.",
                    "parent_id": "page-9",
                    "document_id": "paper-1",
                    "document_title": "Transformer study",
                    "page_number": 9,
                    "source_type": "pdf",
                }
            ),
        ]

    def query_points(self, **kwargs):
        self.filters.append(kwargs["query_filter"])
        return SimpleNamespace(points=self.points)

    def scroll(self, **kwargs):
        self.filters.append(kwargs["scroll_filter"])
        return self.points, None


class Embeddings:
    def embed_query(self, query):
        return [0.1] * vector_store.EMBEDDING_DIM


def test_contextual_chunks_keep_page_parent_and_child_context():
    chunks = _contextual_chunks(
        [Document(page_content="Page text", metadata={"page": 2})], "Paper title"
    )

    assert chunks[0].metadata["page_number"] == 3
    assert chunks[0].metadata["parent_content"] == "Page text"
    assert "Document: Paper title" in chunks[0].metadata["contextual_content"]
    set_document_title(chunks, "Uploaded paper")
    assert chunks[0].metadata["title"] == "Uploaded paper"
    assert "Document: Uploaded paper" in chunks[0].metadata["contextual_content"]


def test_hybrid_search_returns_parent_page_with_page_citation(monkeypatch):
    client = HybridClient()
    monkeypatch.setattr(vector_store, "_ensure_collection", lambda: None)
    monkeypatch.setattr(vector_store, "_get_client", lambda: client)
    monkeypatch.setattr(vector_store, "_get_embeddings", lambda: Embeddings())
    # Isolate fusion/parent behavior from the downloaded model in a unit test.
    monkeypatch.setattr(vector_store, "_cross_encoder_rerank", lambda _q, docs, _k: docs)

    results = vector_store.search_many(
        ["How does attention work?", "attention token interactions"],
        "session-a",
        k=1,
        filters={"document_title": "Transformer study", "page_number": 3},
    )

    assert len(results) == 1
    assert results[0].page_content == "Full PDF page about attention and token interactions."
    assert results[0].metadata["citation_label"] == "Transformer study, p. 3"
    assert all(condition.must[0].match.value == "session-a" for condition in client.filters)
