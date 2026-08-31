from collections.abc import Generator

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily import TavilyClient

from backend.config import load_settings
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
        results = client.search(query, max_results=3)
        context = "\n\n".join(r["content"] for r in results["results"])
        sources = "\n".join(f"- {r['url']}" for r in results["results"])

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
