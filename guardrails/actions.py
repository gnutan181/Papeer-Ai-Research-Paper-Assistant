"""NeMo Guardrails custom actions for Papeer.

NeMo loads this module automatically from the config folder. Each action
returns one verdict string that rails.co acts on: "allow", "injection",
"unsafe" or "error". The Groq calls themselves (timeouts, one retry, token
windows) live in backend/guardrails.py; the classifier is injected by NeMo as
the registered action parameter ``papeer_classifier``.

Every action fails closed: an exception becomes "error", which rails.co
turns into a block with a "please try again" message.
"""

from __future__ import annotations

import asyncio
import logging

from nemoguardrails.actions import action

logger = logging.getLogger("backend.guardrails")


def _injection_verdict(score: float, threshold: float) -> str:
    return "injection" if score >= threshold else "allow"


@action(name="papeer_check_user_input")
async def papeer_check_user_input(text: str, papeer_classifier=None) -> str:
    """Prompt Guard 2 (injection / jailbreak) and the safety model, in parallel."""
    try:
        score, unsafe = await asyncio.gather(
            asyncio.to_thread(papeer_classifier.max_injection_score, text),
            asyncio.to_thread(papeer_classifier.is_unsafe, text, "user question"),
        )
    except Exception as exc:  # noqa: BLE001 - fail closed on any provider failure
        logger.warning("Input guardrail failed closed (%s).", type(exc).__name__)
        return "error"
    threshold = papeer_classifier.settings.guardrail_injection_threshold
    verdict = _injection_verdict(score, threshold)
    if verdict == "allow" and unsafe:
        verdict = "unsafe"
    logger.info(
        "Input guardrail verdict=%s injection_score=%.3f unsafe=%s",
        verdict,
        score,
        unsafe,
    )
    return verdict


@action(name="papeer_check_document")
async def papeer_check_document(text: str, papeer_classifier=None) -> str:
    """Prompt Guard 2 over every token window of a retrieved chunk or web result."""
    try:
        score = await asyncio.to_thread(papeer_classifier.max_injection_score, text)
    except Exception as exc:  # noqa: BLE001 - fail closed on any provider failure
        logger.warning("Retrieval guardrail failed closed (%s).", type(exc).__name__)
        return "error"
    return _injection_verdict(
        score, papeer_classifier.settings.guardrail_injection_threshold
    )


@action(name="papeer_check_output")
async def papeer_check_output(text: str, papeer_classifier=None) -> str:
    """Safety model over the final answer."""
    try:
        unsafe = await asyncio.to_thread(
            papeer_classifier.is_unsafe, text, "assistant answer"
        )
    except Exception as exc:  # noqa: BLE001 - fail closed on any provider failure
        logger.warning("Output guardrail failed closed (%s).", type(exc).__name__)
        return "error"
    logger.info("Output guardrail verdict=%s", "unsafe" if unsafe else "allow")
    return "unsafe" if unsafe else "allow"
