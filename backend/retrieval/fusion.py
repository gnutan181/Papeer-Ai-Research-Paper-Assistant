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
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    raw_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for source in sorted(rankings):
        for rank, hit in enumerate(rankings[source], 1):
            key = hit.evidence_key()
            score[key] += 1 / (k + rank)
            # Prefer a candidate which retains reconstructable parent context;
            # otherwise keep the first deterministic source ordering.
            current = hits.get(key)
            if current is None or (
                hit.metadata.get("parent_content")
                and not current.metadata.get("parent_content")
            ):
                hits[key] = hit
            sources[key].append(source)
            ranks[key][source] = rank
            raw_scores[key][source] = hit.score
    fused = []
    for key in sorted(hits, key=lambda item: (-score[item], item)):
        fused.append(
            hits[key].model_copy(
                update={
                    "score": score[key],
                    "retrieval_sources": sources[key],
                    "original_ranks": {**hits[key].original_ranks, **ranks[key]},
                    "original_scores": {
                        **hits[key].original_scores,
                        **raw_scores[key],
                    },
                }
            )
        )
    return fused


def deduplicate_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    seen: set[str] = set()
    result = []
    for hit in hits:
        key = hit.evidence_key()
        if key not in seen:
            seen.add(key)
            result.append(hit)
    return result


def modality_diverse(
    hits: list[RetrievalHit], limit: int, per_modality: int = 2
) -> list[RetrievalHit]:
    # Diversity is a soft presentation concern.  It must never suppress useful
    # text merely because no other modality is available.
    counts: dict[str, int] = defaultdict(int)
    result = []
    for hit in hits:
        if counts[hit.content_type] >= per_modality and any(
            other.content_type != hit.content_type for other in hits
        ):
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
