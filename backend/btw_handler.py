from collections.abc import Generator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily import TavilyClient

from backend.config import load_settings
from backend.guardrails import (
    RETRY_MESSAGE,
    filter_untrusted_documents,
    guard_input,
)
from backend.models import BtwRouteDecision

settings = load_settings()
llm = ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key)


# Route → Should this question use web search?
# Retrieve → If yes, get information from Tavily.
# Generate + stream → Give the LLM the appropriate context and stream its answer.

# ChatPromptTemplate
#         +
# ChatGroq
#         +
# with_structured_output()
#         +
# invoke()
#         +
# stream()
#         +
# Generator / yield


def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel — never touches the vector store or checkpointer."""
    # Same input rail as the main graph: block before any LLM or web call.
    verdict = guard_input(query, settings)
    if not verdict.allowed:
        yield verdict.message
        return

    route_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Decide if answering this question requires a real-time web search (recent events, "
                    "current prices, breaking news) or if your general knowledge is sufficient."
                ),
            ),
            ("human", "{query}"),
        ]
    )
    decision = (route_prompt | llm.with_structured_output(BtwRouteDecision)).invoke(
        {"query": query}
    )

    if decision.needs_web_search:
        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(query, max_results=3).get("results", [])
        # Retrieval rail: drop web results carrying injected instructions.
        screened = filter_untrusted_documents(
            [
                Document(page_content=r["content"], metadata={"source_url": r["url"]})
                for r in results
            ],
            settings,
        )
        if screened.error:
            yield RETRY_MESSAGE
            return
        context = "\n\n".join(doc.page_content for doc in screened.kept)
        sources = "\n".join(f"- {doc.metadata['source_url']}" for doc in screened.kept)

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "Answer the question using the web search results below. Be concise.\n\n"
                        f"Results:\n{context}\n\nSources:\n{sources}"
                    ),
                ),
                ("human", "{query}"),
            ]
        )
    else:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question concisely from your general knowledge.",
                ),
                ("human", "{query}"),
            ]
        )

    for chunk in (answer_prompt | llm).stream({"query": query}):
        if chunk.content:
            yield chunk.content
