"""Pure deterministic RRF, de-duplication, diversity, and table expansion."""

from __future__ import annotations

from collections import defaultdict

from backend.retrieval.models import RetrievalHit


def reciprocal_rank_fusion(
    rankings: dict[str, list[RetrievalHit]], k: int = 60
) -> list[RetrievalHit]:
    score: dict[str, float] = defaultdict(float)
    hits: dict[str, RetrievalHit] = {}
    sources: dict[str, list[str]] = defaultdict(list)
    for source in sorted(rankings):
        for rank, hit in enumerate(rankings[source], 1):
            key = hit.element_id
            score[key] += 1 / (k + rank)
            hits[key] = hit
            sources[key].append(source)
    fused = []
    for key in sorted(hits, key=lambda item: (-score[item], item)):
        fused.append(
            hits[key].model_copy(
                update={"score": score[key], "retrieval_sources": sources[key]}
            )
        )
    return fused


def deduplicate_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    seen: set[str] = set()
    result = []
    for hit in hits:
        key = hit.parent_id or hit.element_id
        if key not in seen:
            seen.add(key)
            result.append(hit)
    return result


def modality_diverse(
    hits: list[RetrievalHit], limit: int, per_modality: int = 2
) -> list[RetrievalHit]:
    counts: dict[str, int] = defaultdict(int)
    result = []
    for hit in hits:
        if counts[hit.content_type] >= per_modality:
            continue
        result.append(hit)
        counts[hit.content_type] += 1
        if len(result) == limit:
            break
    return [hit.model_copy(update={"rank": i}) for i, hit in enumerate(result, 1)]


def expand_parent_tables(
    hits: list[RetrievalHit], parents: dict[str, RetrievalHit]
) -> list[RetrievalHit]:
    return [
        parents.get(hit.parent_id, hit)
        if hit.parent_id and hit.content_type == "table"
        else hit
        for hit in hits
    ]
