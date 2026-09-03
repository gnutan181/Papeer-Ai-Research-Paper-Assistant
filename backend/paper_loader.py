import os
import re
import tempfile
import urllib.parse
import urllib.request
from hashlib import sha256
from pathlib import Path

# langchain-community reads this at import time.  Identify the application to
# sites loaded by users while preserving an explicit deployment override.
os.environ.setdefault("USER_AGENT", "Papeer/1.0 (+)https://github.com/gnutan181/Papeer-Ai-Research-Paper-Assistant")

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


def _contextual_chunks(
    docs: list[Document], title: str, splitter: RecursiveCharacterTextSplitter = _splitter
) -> list[Document]:
    """Split source pages while preserving a retrievable parent document.

    Parent text is a page for PDFs (and the source document for web/text).  A
    page is small enough to safely put back into the answer context, unlike an
    entire paper, and retains tables/captions that a child chunk may omit.
    """
    chunks: list[Document] = []
    for source_index, parent in enumerate(docs):
        page_number = parent.metadata.get("page")
        page_number = page_number + 1 if isinstance(page_number, int) else parent.metadata.get("page_number")
        parent_id = sha256(
            f"{title}:{page_number or source_index}:{parent.page_content}".encode()
        ).hexdigest()
        children = splitter.split_documents([parent])
        for child in children:
            child.metadata.update(
                {
                    "title": title,
                    "page_number": page_number,
                    "parent_id": parent_id,
                    "parent_content": parent.page_content,
                }
            )
            # Contextual retrieval improves standalone chunk meaning without
            # contaminating what is shown to users as the original passage.
            location = f"page {page_number}" if page_number else "source document"
            child.metadata["contextual_content"] = (
                f"Document: {title}\nLocation: {location}\n\n{child.page_content}"
            )
            chunks.append(child)
    return chunks


def set_document_title(docs: list[Document], title: str) -> list[Document]:
    """Apply a user-facing title after a temporary upload path has been used."""
    for chunk_index, doc in enumerate(docs):
        page_number = doc.metadata.get("page_number")
        parent_content = str(doc.metadata.get("parent_content") or doc.page_content)
        doc.metadata["title"] = title
        doc.metadata["parent_id"] = sha256(
            f"{title}:{page_number or chunk_index}:{parent_content}".encode()
        ).hexdigest()
        location = f"page {page_number}" if page_number else "source document"
        doc.metadata["contextual_content"] = (
            f"Document: {title}\nLocation: {location}\n\n{doc.page_content}"
        )
    return docs


def load_pdf(file_path: str) -> list[Document]:
    docs = PyMuPDFLoader(file_path).load()
    return _contextual_chunks(docs, Path(file_path).stem)


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _contextual_chunks(docs, Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _contextual_chunks(docs, Path(file_path).stem, _md_splitter)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _contextual_chunks(docs, title)


def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download and chunk an ArXiv paper PDF by its bare ID."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        docs = PyMuPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(
            arxiv_id
        )
        return _contextual_chunks(docs, title)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
