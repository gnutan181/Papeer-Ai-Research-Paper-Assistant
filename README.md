# Papeer — production RAG research assistant

Papeer answers questions over session-isolated PDF, ArXiv, text, Markdown, and
web sources. PDF results are returned as their original page-sized parent
document, so tables, figure captions, and surrounding multimodal page context
stay available to the answer model.

## Retrieval pipeline

`query → explicit metadata filters → 3 query variants → dense + BM25 → RRF →
cross-encoder reranking → parent page context → cited answer`

- Child chunks are contextualized with title and page location while the source
  page stays intact as a parent document.
- All Qdrant reads are filtered by the anonymous browser session. The lexical
  BM25 stage operates only over that same session.
- Every grounded answer includes stable `[S#]` citations with PDF page numbers
  or web URLs. It abstains when the retrieved sources cannot support a claim.

## LangGraph safeguards

The graph relevance-grades retrieved context, rewrites once when needed, then
uses Tavily as a bounded web fallback. Source text is explicitly isolated as
untrusted data in the answer prompt, preventing document-embedded instructions
from controlling tool use or the assistant.

## Portkey routing

Set `PORTKEY_API_KEY` to send every LLM call (router, retrieval expansion,
grading, rewriting, verification, and generation) through Portkey. Set
`PORTKEY_PROVIDER` to a saved Model Catalog provider such as
`@production-groq`. Production Portkey policies commonly block the inline
value `groq`; when no saved `@slug` is configured Papeer safely uses direct
Groq instead. `PORTKEY_CONFIG` is optional and can select a Portkey
retry/fallback/cache policy.

```powershell
Copy-Item .env.example .env
uv sync --group dev
uv run streamlit run app.py
```

Run checks with `uv run pytest -q` and `uv run ruff check backend app.py tests`.
