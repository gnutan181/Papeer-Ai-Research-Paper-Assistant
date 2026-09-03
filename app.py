"""Streamlit entry point for the anonymous Papeer v1 experience."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from backend.btw_handler import handle_btw
from backend.config import ConfigurationError, load_settings
from backend.paper_loader import (
    load_arxiv,
    load_document,
    load_webpage,
    set_document_title,
)
from backend.rag_graph import build_graph
from backend.vector_store import (
    add_paper,
    delete_expired_documents,
    delete_session_documents,
    document_id_for_bytes,
    list_papers,
)

logger = logging.getLogger(__name__)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="Papeer", page_icon="📚", layout="centered")

try:
    load_settings()
except ConfigurationError:
    # Configuration details go only to server logs, never the public UI.
    logger.exception("Papeer configuration validation failed")
    st.error("Papeer is temporarily unavailable. Please try again later.")
    st.stop()


@st.cache_resource
def get_graph():
    return build_graph()


@st.cache_data(ttl=3600, show_spinner=False)
def cleanup_expired_documents() -> bool:
    """Run the shared cleanup at most once per process per hour."""
    delete_expired_documents()
    return True


def _initialize_session() -> None:
    # This identifier exists only in this browser's Streamlit session state.
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _conversation_messages() -> list[HumanMessage | AIMessage]:
    messages: list[HumanMessage | AIMessage] = []
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        else:
            messages.append(AIMessage(content=message["content"]))
    return messages


def _add_documents(
    docs, document_id: str, document_title: str, source_type: str
) -> bool:
    return add_paper(
        docs=docs,
        session_id=st.session_state.session_id,
        document_id=document_id,
        document_title=document_title,
        source_type=source_type,
    )


_initialize_session()
try:
    cleanup_expired_documents()
except Exception:
    # An unsuccessful cleanup must not prevent a user from deleting their own data.
    logger.exception("Document expiry cleanup failed")

graph = get_graph()
session_id = st.session_state.session_id

with st.sidebar:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Delete my documents", use_container_width=True, type="secondary"):
        try:
            delete_session_documents(session_id)
            st.session_state.chat_history = []
            # A fresh ID prevents any late request from reusing deleted vectors.
            st.session_state.session_id = str(uuid.uuid4())
            st.success("Your uploaded documents were deleted.")
        except Exception:
            logger.exception("Unable to delete session documents")
            st.error("We could not delete your documents right now. Please try again.")

    st.divider()
    st.markdown("## 📄 Documents")
    st.caption("Anonymous uploads are automatically deleted after 24 hours.")

    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key="uploader",
        label_visibility="collapsed",
    )
    if st.button("Add Files", use_container_width=True, key="btn_add_files"):
        if not uploaded_files:
            st.warning("No files selected.")
        else:
            with st.spinner("Processing files…"):
                for uploaded_file in uploaded_files:
                    file_bytes = uploaded_file.getvalue()
                    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
                        st.warning(
                            f"Skipped {uploaded_file.name}: files must be 10 MB or smaller."
                        )
                        continue
                    suffix = Path(uploaded_file.name).suffix.lower()
                    temp_path: str | None = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as temporary_file:
                            temporary_file.write(file_bytes)
                            temp_path = temporary_file.name
                        docs = load_document(temp_path)
                        title = Path(uploaded_file.name).stem
                        set_document_title(docs, title)
                        inserted = _add_documents(
                            docs,
                            document_id_for_bytes(file_bytes),
                            title,
                            suffix.removeprefix("."),
                        )
                        if inserted:
                            st.success(f"Added: {uploaded_file.name}")
                        else:
                            st.info(f"Already loaded: {uploaded_file.name}")
                    except Exception:
                        logger.exception(
                            "Upload processing failed for %s", uploaded_file.name
                        )
                        st.error(f"Could not process {uploaded_file.name}.")
                    finally:
                        if temp_path:
                            Path(temp_path).unlink(missing_ok=True)
            st.rerun()

    st.markdown("**Web Pages**")
    url_input = st.text_area(
        "URLs (one per line)",
        key="url_area",
        height=80,
        label_visibility="collapsed",
        placeholder="https://example.com/paper",
    )
    if st.button("Load URLs", use_container_width=True, key="btn_load_urls"):
        urls = [url.strip() for url in url_input.splitlines() if url.strip()]
        if not urls:
            st.warning("Enter at least one URL.")
        else:
            with st.spinner("Loading web pages…"):
                for url in urls:
                    try:
                        docs = load_webpage(url)
                        inserted = _add_documents(
                            docs,
                            document_id_for_bytes(url.encode()),
                            docs[0].metadata.get("title", url) if docs else url,
                            "web",
                        )
                        st.success(
                            f"{'Loaded' if inserted else 'Already loaded'}: {url[:60]}"
                        )
                    except Exception:
                        logger.exception("Web page loading failed")
                        st.error(f"Could not load {url[:60]}.")
            st.rerun()

    st.markdown("**ArXiv Papers**")
    arxiv_query = st.text_input(
        "Paper title or ArXiv ID",
        key="arxiv_input",
        label_visibility="collapsed",
        placeholder="1706.03762 or Attention Is All You Need",
    )
    if st.button("Load ArXiv Paper", use_container_width=True, key="btn_load_arxiv"):
        if not arxiv_query.strip():
            st.warning("Enter a paper title or ArXiv ID.")
        else:
            with st.spinner("Loading from ArXiv…"):
                try:
                    docs = load_arxiv(arxiv_query.strip())
                    title = (
                        docs[0].metadata.get("title", arxiv_query.strip())
                        if docs
                        else arxiv_query.strip()
                    )
                    inserted = _add_documents(
                        docs,
                        document_id_for_bytes(arxiv_query.strip().lower().encode()),
                        title,
                        "pdf",
                    )
                    st.success(f"{'Loaded' if inserted else 'Already loaded'}: {title}")
                except Exception:
                    logger.exception("ArXiv loading failed")
                    st.error("Could not load that ArXiv paper.")
            st.rerun()

    st.divider()
    st.markdown("### Loaded Documents")
    try:
        document_titles = list_papers(session_id)
    except Exception:
        logger.exception("Document listing failed")
        document_titles = None
    if document_titles is None:
        st.caption("Document list is temporarily unavailable.")
    elif document_titles:
        for title in document_titles:
            st.markdown(f"- {title}")
    else:
        st.caption("No documents loaded yet.")

st.title("📚 Papeer — Research Paper Assistant")
st.markdown(
    "Ask questions about your uploaded papers, verify claims against current "
    "literature, or search the web."
)
st.caption(
    "Chat history is kept only for this browser session and is not retained after a restart."
)
st.divider()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your papers, verify a claim, or search the web…"):
    is_btw = prompt.strip().lower().startswith("/btw")
    if is_btw:
        query = prompt.strip()[4:].strip()
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if not query:
                st.markdown("Please add a question after `/btw`.")
            else:
                placeholder = st.empty()
                answer = ""
                try:
                    for chunk in handle_btw(query):
                        answer += chunk
                        placeholder.markdown(answer + "▌")
                    placeholder.markdown(answer)
                except Exception:
                    logger.exception("Side-channel response failed")
                    placeholder.error(
                        "I could not answer that right now. Please try again."
                    )
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                input_state = {
                    "messages": _conversation_messages(),
                    "session_id": session_id,
                    "query": prompt,
                    "route": None,
                    "retrieved_docs": [],
                    "retrieval_attempts": 0,
                    "claim_verdict": None,
                    "claim_source": None,
                    "superseding_papers": [],
                    "answer": None,
                    "is_relevant": None,
                    "rewrite_count": 0,
                    "search_filters": {},
                    "search_queries": [],
                    "web_fallback_attempted": False,
                }
                result = graph.invoke(input_state)
                answer = result.get("answer") or "No response generated."
                placeholder.markdown(answer)
            except Exception:
                logger.exception("Grounded response failed")
                answer = "I could not answer that right now. Please try again."
                placeholder.error(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
