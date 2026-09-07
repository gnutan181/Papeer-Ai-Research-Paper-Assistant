from __future__ import annotations

import threading
from collections.abc import Sequence

TEXT_MODEL = "BAAI/bge-small-en-v1.5"
_model = None
_lock = threading.Lock()


def get_text_embedder():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(TEXT_MODEL)
    return _model


def embed_texts(values: Sequence[str]) -> list[list[float]]:
    if not values:
        return []
    vectors = get_text_embedder().encode(
        list(values), batch_size=16, normalize_embeddings=True, show_progress_bar=False
    )
    return [list(map(float, value)) for value in vectors]


def text_dimension() -> int:
    return len(embed_texts(["dimension probe"])[0])
