from __future__ import annotations

import threading
from collections.abc import Sequence
from io import BytesIO

VISUAL_MODEL = "google/siglip2-base-patch16-224"
_model = _processor = None
_lock = threading.Lock()


def _load():
    global _model, _processor
    with _lock:
        if _model is None:
            from transformers import AutoModel, AutoProcessor

            _processor = AutoProcessor.from_pretrained(VISUAL_MODEL)
            _model = AutoModel.from_pretrained(VISUAL_MODEL)
            _model.eval()
    return _model, _processor


def _normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def embed_visual_texts(values: Sequence[str]) -> list[list[float]]:
    if not values:
        return []
    import torch

    model, processor = _load()
    with torch.inference_mode():
        features = _normalize(
            model.get_text_features(
                **processor(text=list(values), padding=True, return_tensors="pt")
            )
        )
    return features.cpu().tolist()


def embed_images(values: Sequence[bytes]) -> list[list[float]]:
    if not values:
        return []
    import torch
    from PIL import Image

    model, processor = _load()
    images = [Image.open(BytesIO(value)).convert("RGB") for value in values]
    with torch.inference_mode():
        features = _normalize(
            model.get_image_features(**processor(images=images, return_tensors="pt"))
        )
    return features.cpu().tolist()


def visual_dimension() -> int:
    return len(embed_visual_texts(["dimension probe"])[0])
