"""Rank fusion strategies for combining dense and sparse retrieval.

The core problem: a dense retriever returns cosine similarities in [-1, 1] and
BM25 returns unbounded positive scores. They cannot be added. Two ways out:

- **Reciprocal Rank Fusion** throws the scores away and uses only rank
  position, which sidesteps the calibration problem entirely. It is the
  default because it needs no tuning per corpus.
- **Weighted score fusion** min-max normalizes each list first, then blends
  with an `alpha` knob. More control, but the normalization makes a result's
  score depend on which other documents came back with it.
"""

from __future__ import annotations

from collections import defaultdict

from .types import Retrieved

DEFAULT_RRF_K = 60


def normalize_scores(results: list[Retrieved]) -> list[Retrieved]:
    """Min-max scale scores into [0, 1], preserving order.

    When every score is identical the retriever has expressed no preference,
    so everything collapses to 0.0 rather than 1.0 — otherwise a retriever
    with no opinion would outrank one that actually discriminated.
    """
    if not results:
        return []

    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    spread = hi - lo

    if spread == 0:
        return [r.with_score(0.0) for r in results]

    return [r.with_score((r.score - lo) / spread) for r in results]


def _normalize_for_blending(results: list[Retrieved]) -> dict[str, Retrieved]:
    """Normalize for weighted fusion, where a lone result must not vanish.

    `normalize_scores` maps a single-document list to 0.0, which is right when
    reporting one retriever's output — there is nothing to be relatively better
    than. But in a blend it silently deletes a legitimate top hit: a document
    that only one retriever found would contribute nothing from either side.
    Here a sole result is treated as maximally confident instead.
    """
    if len(results) == 1:
        return {results[0].id: results[0].with_score(1.0)}
    return {r.id: r for r in normalize_scores(results)}


def reciprocal_rank_fusion(
    rankings: list[list[Retrieved]],
    k: int = DEFAULT_RRF_K,
    weights: list[float] | None = None,
) -> list[Retrieved]:
    """Fuse several rankings by reciprocal rank: score = sum(w / (k + rank)).

    `k` damps the advantage of the top position. A small k makes rank 0
    dominate; a large k flattens the curve so agreement across many lists
    matters more than any single first place.

    Documents missing from a ranking simply contribute nothing for it, so a
    result found by one retriever alone can still surface — just with a
    ceiling on how high it can climb.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError(
            f"weights must have one entry per ranking: "
            f"got {len(weights)} weights for {len(rankings)} rankings"
        )

    effective_weights = weights or [1.0] * len(rankings)

    fused: dict[str, float] = defaultdict(float)
    # Keep the richest copy of each document we see; retrievers may return
    # the same id with different metadata completeness.
    seen: dict[str, Retrieved] = {}

    for ranking, weight in zip(rankings, effective_weights):
        # Ranks are 1-based, per the original RRF paper (Cormack et al. 2009).
        for rank, doc in enumerate(ranking, start=1):
            fused[doc.id] += weight / (k + rank)
            if doc.id not in seen:
                seen[doc.id] = doc

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [seen[doc_id].with_score(score) for doc_id, score in ordered]


def weighted_score_fusion(
    dense: list[Retrieved],
    sparse: list[Retrieved],
    alpha: float = 0.5,
) -> list[Retrieved]:
    """Blend normalized dense and sparse scores: alpha*dense + (1-alpha)*sparse.

    `alpha=1.0` is pure dense, `alpha=0.0` is pure sparse. Unlike RRF this
    keeps the magnitude of each retriever's confidence, which helps when one
    retriever is reliably better on your corpus and you have the eval data to
    prove it. Without that data, prefer RRF.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    dense_norm = _normalize_for_blending(dense)
    sparse_norm = _normalize_for_blending(sparse)

    blended: list[Retrieved] = []
    for doc_id in dense_norm.keys() | sparse_norm.keys():
        d = dense_norm.get(doc_id)
        s = sparse_norm.get(doc_id)
        score = alpha * (d.score if d else 0.0) + (1 - alpha) * (s.score if s else 0.0)
        blended.append((d or s).with_score(score))

    return sorted(blended, key=lambda r: r.score, reverse=True)
