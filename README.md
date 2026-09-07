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

## Multimodal and voice demo

Papeer is a production-style recruiter demo, not a claim of unlimited-scale
production hosting. It adds a typed multimodal ingestion path alongside the
existing, session-isolated text collection. Existing `papeer_chunks_v2` data is
never deleted or mutated; new elements use `papeer_multimodal_v1` with named
`text_dense` and `visual_dense` vectors.

```mermaid
flowchart TD
  A[PDF / ArXiv / Web] --> B[Docling ingestion]
  B --> C[Text, OCR and tables]
  B --> D[Figures and charts]
  C --> E[BGE text vectors]
  D --> F[SigLIP2 image vectors]
  E --> G[Qdrant text search]
  F --> H[Qdrant visual search]
  G --> I[RRF fusion]
  H --> I
  I --> J[Grounded answer with citations]
```

- PDFs use Docling first for layout, table, and OCR-capable extraction. A
  controlled PyMuPDF fallback preserves text-only operation if Docling is
  unavailable. OCR is routed only for sparse native text, not indiscriminately.
- Each extracted item uses a stable content hash and a `DocumentElement` schema
  containing page, section, table/figure IDs, bounding box, extraction method,
  and optional asset key. Image bytes are kept out of Qdrant.
- Tables retain a Markdown form with headers; the retrieval helpers support
  parent-table expansion. Text and visual ranking are fused deterministically
  with reciprocal-rank fusion, then deduplicated and modality-balanced.
- Streamlit's audio chat input uses Groq Whisper Turbo by default and Whisper
  Large V3 when the accuracy setting is enabled. Recordings are transcribed,
  shown for edit/confirm, and are not retained. The optional Read Answer
  control uses the browser's Web Speech API; it never auto-plays.
- Figure analysis is on-demand and uses Groq Llama 4 Scout only after visual
  retrieval. Its prompt requires visible-evidence JSON and explicit uncertainty.
  If it is unavailable, the UI must rely on caption/surrounding text instead.

### Run locally

```powershell
Copy-Item .env.example .env
uv sync --group dev
uv run streamlit run app.py
```

Set `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `TAVILY_API_KEY` in
`.env`; never commit it. `ASSET_STORE=local` works without object-storage
credentials. For Cloudflare R2, set `ASSET_STORE=r2` plus every `R2_*` setting.
The full variable list and safe defaults are in `.env.example`.

### Streamlit Community Cloud

1. Push this repository (without `.env`) and create an app pointed at `app.py`.
2. Add the required values above in the app's Secrets UI, plus optional R2
   settings only if using R2.
3. Keep optional model flags enabled only when the deployment has enough CPU,
   memory, and provider quota. Models are loaded lazily; first visual retrieval
   can be slow on free hosting.

The application enforces 20 MB files, 40 PDF pages, 30 figures, 10 MB audio,
five URLs, local single-flight ingestion, filename/MIME checks, temporary-file
cleanup, strict HTTPS public URL checks, and session filters on every new
multimodal vector query. It does not log keys, raw recordings, or complete
private document contents.

### Evaluation and limitations

`tests/test_multimodal.py` covers canonical schemas and IDs, file validation,
OCR routing, table serialization, figure filtering, RRF, deduplication,
diversity, citations, speech cleanup, and tenant filters. Provider integration
tests are intentionally not run without credentials or downloaded models, so no
performance, faithfulness, or OCR-accuracy metric is claimed here. Add a small
fixture corpus before reporting Recall@k, MRR/nDCG, latency, citation-validity,
or answer-faithfulness values.

Good demo questions include: “Summarize the method on page 4,” “What does
Table 2 compare?”, “Which chart supports the reported trend?”, and “What is
insufficiently supported by this paper?” Web results are supplementary and are
not represented as uploaded-paper evidence.
