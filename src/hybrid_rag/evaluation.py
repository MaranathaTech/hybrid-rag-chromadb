"""Retrieval evaluation.

A hybrid pipeline that is never measured is a claim, not a result. These are
the standard rank-aware metrics, plus a comparison harness that runs the same
queries through each strategy so the numbers can be put side by side.

Worth saying plainly: hybrid is not free and not always better. It costs a
second index, a second query, and latency. On a corpus of paraphrase-heavy
prose with no rare identifiers, dense alone often matches it. The point of
this module is to find out which case you are in, not to confirm that the
more complicated option won.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .pipeline import HybridConfig, HybridRetriever


@dataclass(frozen=True)
class EvalQuery:
    """A query and the document ids that genuinely answer it."""

    query: str
    relevant_ids: frozenset[str]

    @classmethod
    def of(cls, query: str, *relevant_ids: str) -> EvalQuery:
        return cls(query=query, relevant_ids=frozenset(relevant_ids))


@dataclass
class Metrics:
    """Aggregate retrieval quality over a query set."""

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    n_queries: int

    def as_row(self, label: str) -> str:
        return (
            f"{label:<12} "
            f"recall@{self.k}={self.recall_at_k:.3f}  "
            f"precision@{self.k}={self.precision_at_k:.3f}  "
            f"mrr={self.mrr:.3f}  "
            f"ndcg@{self.k}={self.ndcg_at_k:.3f}"
        )


def recall_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of relevant documents that made the top k."""
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of the top k that are relevant."""
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: frozenset[str]) -> float:
    """1/rank of the first relevant hit; 0 if none.

    Rewards getting one good answer to the top, which is what matters when the
    result feeds an LLM context window rather than a human scanning a page.
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: frozenset[str], k: int) -> float:
    """Normalized discounted cumulative gain, binary relevance.

    Unlike recall, this is sensitive to *where* in the top k a hit landed —
    the metric that actually moves when fusion reorders a shortlist without
    changing its membership.
    """
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    retriever: HybridRetriever,
    queries: list[EvalQuery],
    k: int = 5,
) -> Metrics:
    """Run a query set through a retriever and aggregate the metrics."""
    if not queries:
        raise ValueError("Cannot evaluate against an empty query set.")

    recalls, precisions, rrs, ndcgs = [], [], [], []

    for eval_query in queries:
        result = retriever.search(eval_query.query, limit=k)
        ids = result.ids

        recalls.append(recall_at_k(ids, eval_query.relevant_ids, k))
        precisions.append(precision_at_k(ids, eval_query.relevant_ids, k))
        rrs.append(reciprocal_rank(ids, eval_query.relevant_ids))
        ndcgs.append(ndcg_at_k(ids, eval_query.relevant_ids, k))

    n = len(queries)
    return Metrics(
        recall_at_k=sum(recalls) / n,
        precision_at_k=sum(precisions) / n,
        mrr=sum(rrs) / n,
        ndcg_at_k=sum(ndcgs) / n,
        k=k,
        n_queries=n,
    )


@dataclass
class Comparison:
    """Metrics for each strategy over the same query set."""

    results: dict[str, Metrics] = field(default_factory=dict)

    def best_by(self, metric: str = "ndcg_at_k") -> str:
        return max(self.results, key=lambda s: getattr(self.results[s], metric))

    def report(self) -> str:
        lines = [
            f"Retrieval comparison over {next(iter(self.results.values())).n_queries} queries",
            "-" * 72,
        ]
        lines.extend(
            metrics.as_row(label) for label, metrics in self.results.items()
        )
        lines.append("-" * 72)

        winner = self.best_by()
        hybrid_score = self.results.get("rrf")
        dense_score = self.results.get("dense")

        lines.append(f"Best by ndcg@k: {winner}")

        # State it plainly when the extra machinery did not earn its place.
        if hybrid_score and dense_score:
            delta = hybrid_score.ndcg_at_k - dense_score.ndcg_at_k
            if delta > 0.01:
                lines.append(
                    f"Hybrid beats dense-only by {delta:+.3f} ndcg@k."
                )
            elif delta < -0.01:
                lines.append(
                    f"Hybrid LOSES to dense-only by {delta:.3f} ndcg@k on this "
                    f"corpus. The second index is not paying for itself here."
                )
            else:
                lines.append(
                    f"Hybrid and dense-only are within {abs(delta):.3f} ndcg@k "
                    f"— indistinguishable on this corpus. Prefer the simpler one."
                )

        return "\n".join(lines)


def compare_strategies(
    retriever: HybridRetriever,
    queries: list[EvalQuery],
    k: int = 5,
    strategies: list[str] | None = None,
) -> Comparison:
    """Evaluate the same corpus and queries under each retrieval strategy."""
    to_test = strategies or ["dense", "sparse", "rrf", "weighted"]
    original_config = retriever.config
    comparison = Comparison()

    try:
        for strategy in to_test:
            retriever.config = HybridConfig(
                limit=k,
                fetch_k=max(original_config.fetch_k, k),
                strategy=strategy,
                rrf_k=original_config.rrf_k,
                alpha=original_config.alpha,
                rerank=original_config.rerank,
            )
            comparison.results[strategy] = evaluate(retriever, queries, k=k)
    finally:
        retriever.config = original_config

    return comparison
