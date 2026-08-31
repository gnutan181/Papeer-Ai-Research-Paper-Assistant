from types import SimpleNamespace

from langchain_core.documents import Document

from backend import vector_store
from backend.config import Settings


class FakeEmbeddings:
    def embed_documents(self, documents):
        return [[0.1] * vector_store.EMBEDDING_DIM for _ in documents]

    def embed_query(self, query):
        return [0.1] * vector_store.EMBEDDING_DIM


class FakeClient:
    def __init__(self):
        self.created_collection = None
        self.indexes = []
        self.upserted = []
        self.scroll_filter = None
        self.query_filter = None
        self.deleted_filter = None

    def collection_exists(self, name):
        return False

    def create_collection(self, **kwargs):
        self.created_collection = kwargs

    def create_payload_index(self, *args):
        self.indexes.append(args)

    def scroll(self, **kwargs):
        self.scroll_filter = kwargs["scroll_filter"]
        return [], None

    def upsert(self, **kwargs):
        self.upserted = kwargs["points"]

    def query_points(self, **kwargs):
        self.query_filter = kwargs["query_filter"]
        return SimpleNamespace(points=[])

    def delete(self, **kwargs):
        self.deleted_filter = kwargs["points_selector"]


def _configure(monkeypatch, client):
    settings = Settings("groq", "https://qdrant.example.test", "qdrant", "tavily")
    monkeypatch.setattr(vector_store, "_settings", lambda: settings)
    monkeypatch.setattr(vector_store, "_get_client", lambda: client)
    monkeypatch.setattr(vector_store, "_get_embeddings", lambda: FakeEmbeddings())


def test_add_paper_writes_required_payload_and_single_collection(monkeypatch):
    client = FakeClient()
    _configure(monkeypatch, client)
    documents = [
        Document(page_content="first", metadata={"page": 0}),
        Document(page_content="second", metadata={"page": 1}),
    ]

    inserted = vector_store.add_paper(
        documents,
        session_id="session-a",
        document_id="document-a",
        document_title="A paper",
        source_type="pdf",
    )

    assert inserted is True
    assert client.created_collection["collection_name"] == "papeer_chunks_v2"
    assert [point.payload["chunk_index"] for point in client.upserted] == [0, 1]
    payload = client.upserted[0].payload
    assert payload["session_id"] == "session-a"
    assert payload["document_id"] == "document-a"
    assert payload["document_title"] == "A paper"
    assert payload["page_number"] == 1
    assert payload["source_type"] == "pdf"
    assert payload["expires_at"]


def test_search_always_filters_to_the_current_session(monkeypatch):
    client = FakeClient()
    _configure(monkeypatch, client)

    vector_store.search("question", session_id="session-a")

    condition = client.query_filter.must[0]
    assert condition.key == "session_id"
    assert condition.match.value == "session-a"


def test_document_hash_is_stable_and_content_dependent():
    assert vector_store.document_id_for_bytes(
        b"paper"
    ) == vector_store.document_id_for_bytes(b"paper")
    assert vector_store.document_id_for_bytes(
        b"paper"
    ) != vector_store.document_id_for_bytes(b"other paper")
