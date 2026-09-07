import pytest

from backend.audio.browser_tts import markdown_to_speech
from backend.audio.transcription import normalize_chat_input
from backend.citations.formatter import citation_label, validate_citations
from backend.config import Settings
from backend.ingestion.docling_parser import (
    is_meaningful_figure,
    should_use_ocr,
    table_to_markdown,
)
from backend.ingestion.element_schema import (
    DocumentElement,
    deterministic_element_id,
    stable_content_hash,
)
from backend.ingestion.validators import ValidationError, validate_upload
from backend.retrieval.fusion import (
    deduplicate_hits,
    modality_diverse,
    reciprocal_rank_fusion,
)
from backend.retrieval.models import RetrievalHit
from backend.retrieval.multimodal_store import session_filter


def _element(**overrides):
    values = {
        "element_id": "e1",
        "document_id": "d1",
        "session_id": "s1",
        "content_type": "text",
        "text": "evidence",
        "page_number": 2,
        "source_filename": "paper.pdf",
        "extraction_method": "docling",
        "content_hash": "hash",
    }
    values.update(overrides)
    return DocumentElement(**values)


def _hit(element_id, kind="text", parent_id=None):
    return RetrievalHit(
        element_id=element_id,
        content_type=kind,
        page_number=2,
        score=0,
        rank=0,
        citation_label="paper.pdf, p. 2",
        parent_id=parent_id,
        text=element_id,
    )


def test_document_schema_and_deterministic_ids():
    item = _element()
    assert item.searchable_text() == "evidence"
    assert deterministic_element_id(
        "d", "text", 1, stable_content_hash("x")
    ) == deterministic_element_id("d", "text", 1, stable_content_hash("x"))
    with pytest.raises(ValueError):
        _element(page_number=0)


def test_upload_validation_and_ocr_table_figure_helpers():
    settings = Settings("g", "https://qdrant.example.test", "q", "t")
    assert validate_upload("paper.pdf", b"%PDF-1.7 body", settings) == "pdf"
    with pytest.raises(ValidationError):
        validate_upload("paper.pdf", b"not a pdf", settings)
    assert should_use_ocr("almost empty")
    assert not should_use_ocr(
        "This is enough native text to avoid unnecessary OCR on a PDF page."
    )
    assert "| A | B |" in table_to_markdown(["A", "B"], [["1", "2"]])
    assert is_meaningful_figure(100, 100, 500_000) is True
    assert is_meaningful_figure(5, 5, 500_000) is False


def test_rrf_deduplication_diversity_and_session_filter():
    fused = reciprocal_rank_fusion(
        {
            "text_dense": [_hit("a"), _hit("b", "table")],
            "visual_dense": [_hit("b", "table"), _hit("c", "figure")],
        }
    )
    assert [hit.element_id for hit in fused][:2] == ["b", "a"]
    unique = deduplicate_hits(
        [_hit("child", "table", "parent"), _hit("other", "table", "parent")]
    )
    assert len(unique) == 1
    diverse = modality_diverse(
        [_hit("a"), _hit("b"), _hit("c", "figure")], 3, per_modality=1
    )
    assert [hit.content_type for hit in diverse] == ["text", "figure"]
    condition = session_filter("tenant-a").must[0]
    assert condition.key == "session_id" and condition.match.value == "tenant-a"


def test_citation_and_audio_presentation_helpers():
    item = _element(figure_number="3")
    label = citation_label(item)
    assert label == "paper.pdf, p. 2, Figure 3"
    cleaned, citations = validate_citations(
        f"Supported [{label}] but not [made up].", [item]
    )
    assert "made up" not in cleaned and citations == [label]
    assert (
        markdown_to_speech("**Answer** [paper](https://example.test) [S1]")
        == "Answer paper"
    )
    assert normalize_chat_input("  hello  ") == ("hello", None)
