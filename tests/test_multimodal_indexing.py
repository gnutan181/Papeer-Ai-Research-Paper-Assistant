from types import SimpleNamespace

from langchain_core.documents import Document

from backend.config import Settings
from backend.ingestion.element_schema import DocumentElement
from backend.ingestion.indexing import index_uploaded_document
from backend.retrieval.multimodal_store import MultimodalStore


def _settings(**overrides):
    values = {
        "groq_api_key": "g",
        "qdrant_url": "https://qdrant.example.test",
        "qdrant_api_key": "q",
        "tavily_api_key": "t",
    }
    values.update(overrides)
    return Settings(**values)


def _element(**overrides):
    values = {
        "element_id": "element",
        "document_id": "document",
        "session_id": "session",
        "content_type": "text",
        "text": "evidence",
        "page_number": 1,
        "source_filename": "paper.pdf",
        "extraction_method": "pdf",
        "content_hash": "hash",
    }
    values.update(overrides)
    return DocumentElement(**values)


def test_multimodal_failure_does_not_prevent_legacy_indexing():
    calls = []

    def legacy(**kwargs):
        calls.append(kwargs)
        return True

    class BrokenStore:
        def index(self, *_args, **_kwargs):
            raise RuntimeError("optional collection unavailable")

    inserted, warning = index_uploaded_document(
        [Document(page_content="text")],
        [_element()],
        object(),
        "session",
        "document",
        "Paper",
        "pdf",
        _settings(),
        legacy_indexer=legacy,
        multimodal_factory=lambda _settings: BrokenStore(),
    )

    assert inserted is True
    assert warning
    assert calls[0]["document_id"] == "document"


def test_duplicate_legacy_document_does_not_reupsert_multimodal():
    class Store:
        def __init__(self):
            self.called = False

        def index(self, *_args, **_kwargs):
            self.called = True

    store = Store()
    inserted, warning = index_uploaded_document(
        [Document(page_content="text")],
        [_element()],
        object(),
        "session",
        "document",
        "Paper",
        "pdf",
        _settings(),
        legacy_indexer=lambda **_kwargs: False,
        multimodal_factory=lambda _settings: store,
    )
    assert (inserted, warning, store.called) == (False, None, False)


class CollectionClient:
    def __init__(self):
        self.created = None
        self.upserted = []

    def collection_exists(self, _name):
        return False

    def create_collection(self, **kwargs):
        self.created = kwargs

    def create_payload_index(self, *_args):
        pass

    def upsert(self, *_args, **kwargs):
        self.upserted.extend(kwargs["points"])


class NoVisualSearchClient:
    def __init__(self):
        self.scroll_filters = []

    def scroll(self, **kwargs):
        self.scroll_filters.append(kwargs["scroll_filter"])
        return [], None

    def query_points(self, **_kwargs):
        return SimpleNamespace(points=[])


def test_siglip_is_not_loaded_when_disabled_or_no_visual_elements(monkeypatch):
    client = CollectionClient()
    disabled = MultimodalStore(_settings(enable_image_embeddings=False), client=client)
    disabled.ensure_collection()
    assert "visual_dense" in client.created["vectors_config"]

    client = CollectionClient()
    enabled = MultimodalStore(_settings(enable_image_embeddings=True), client=client)
    monkeypatch.setattr(
        "backend.retrieval.multimodal_store.embed_texts",
        lambda values: [[0.1] * 384 for _ in values],
    )
    monkeypatch.setattr(
        "backend.retrieval.multimodal_store.embed_images",
        lambda _values: (_ for _ in ()).throw(AssertionError("SigLIP should not load")),
    )
    enabled.index([_element()], asset_store=SimpleNamespace(get=lambda _key: b""))
    assert len(client.upserted) == 1

    search_client = NoVisualSearchClient()
    searchable = MultimodalStore(_settings(enable_image_embeddings=True), search_client)
    monkeypatch.setattr(
        "backend.retrieval.multimodal_store.embed_texts",
        lambda values: [[0.1] * 384 for _ in values],
    )
    monkeypatch.setattr(
        "backend.retrieval.multimodal_store.embed_visual_texts",
        lambda _values: (_ for _ in ()).throw(AssertionError("SigLIP should not load")),
    )
    searchable.search_many(["question"], "session")
    condition = search_client.scroll_filters[0].must[0]
    assert condition.key == "session_id" and condition.match.value == "session"
