import os

os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("QDRANT_URL", "https://qdrant.example.test")
os.environ.setdefault("QDRANT_API_KEY", "test")
os.environ.setdefault("TAVILY_API_KEY", "test")

from langchain_core.messages import ToolMessage

from backend import rag_graph


def test_empty_uploaded_evidence_preserves_corrective_rag_flow(monkeypatch):
    monkeypatch.setattr(
        rag_graph, "retrieve_uploaded_evidence", lambda *_args, **_kwargs: []
    )
    responses = rag_graph.retrieve_from_vectorstore.func(
        query="question",
        k=2,
        session_id="session-a",
        current_docs=[],
        search_queries=["question", "variant"],
        search_filters={},
        tool_call_id="call-a",
    )

    assert isinstance(responses[0], ToolMessage)
    assert "No relevant documents" in responses[0].content
    assert (
        rag_graph.after_relevancy_routing(
            {"is_relevant": False, "rewrite_count": 0, "web_fallback_attempted": False}
        )
        == "query_rewrite"
    )
    assert (
        rag_graph.after_relevancy_routing(
            {"is_relevant": False, "rewrite_count": 1, "web_fallback_attempted": False}
        )
        == "web_fallback"
    )
