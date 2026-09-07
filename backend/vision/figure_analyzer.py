from __future__ import annotations

import base64

from backend.config import Settings
from backend.vision.prompts import FIGURE_PROMPT
from backend.vision.schemas import FigureAnalysis


def analyze_figure(
    image: bytes, settings: Settings, client: object | None = None
) -> FigureAnalysis:
    if not settings.enable_vision:
        raise RuntimeError("Vision analysis is disabled.")
    if client is None:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key, timeout=30)
    encoded = base64.b64encode(image).decode("ascii")
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FIGURE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
    )
    return FigureAnalysis.model_validate_json(response.choices[0].message.content)
