"""End-to-end retrieval quality, measured against both corpora.

These are the tests that keep the project's central claim honest. They run real
embeddings through a real Chroma collection — no mocks — and assert the
*relationships* between strategies that the README reports.

The headline result is deliberately not "hybrid wins". It is:

  - On the easy corpus (distinct topics), dense alone is as good as hybrid,
    so the second index buys nothing.
  - On the hard corpus (near-duplicates keyed by exact identifiers), dense
    degrades badly and sparse is perfect — and fusion, by blending dense's
    wrong answers back in, scores WORSE than sparse alone.

If a future change makes hybrid look uniformly best, one of these fails, and
that is the point: a benchmark that can only report wins is not a measurement.

Marked `integration` and excluded from the Docker build, which has no Chroma.
"""

import pytest

from hybrid_rag.corpus import (
    load_eval_queries,
    load_hard_corpus,
    load_hard_eval_queries,
    load_sample_corpus,
)
from hybrid_rag.dense import ChromaDenseRetriever
from hybrid_rag.embeddings import get_embedding_provider
from hybrid_rag.evaluation import compare_strategies, evaluate
from hybrid_rag.pipeline import HybridConfig, HybridRetriever
from hybrid_rag.sparse import BM25Retriever

pytestmark = pytest.mark.integration


def _retriever(corpus, collection: str) -> HybridRetriever:
    """An in-memory Chroma retriever seeded with a corpus."""
    dense = ChromaDenseRetriever(
        embeddings=get_embedding_provider("local"),
        collection_name=collection,
    )
    retriever = HybridRetriever(dense=dense, sparse=BM25Retriever())
    retriever.index(corpus)
    return retriever


@pytest.fixture(scope="module")
def easy():
    return _retriever(load_sample_corpus(), "test_easy")


@pytest.fixture(scope="module")
def hard():
    return _retriever(load_hard_corpus(), "test_hard")


class TestEasyCorpus:
    """Distinct topics: dense alone is already enough."""

    def test_every_strategy_retrieves_well(self, easy):
        comparison = compare_strategies(easy, load_eval_queries(), k=5)

        for strategy, metrics in comparison.results.items():
            assert metrics.recall_at_k >= 0.85, (
                f"{strategy} recall collapsed on an easy corpus: "
                f"{metrics.recall_at_k:.3f}"
            )

    def test_hybrid_does_not_meaningfully_beat_dense(self, easy):
        # The honest negative result. When topics are well separated there is
        # nothing for exact matching to add, and fusion cannot invent signal
        # that is not there. A future change that makes hybrid "win" here is
        # far more likely to be overfitting than an improvement.
        comparison = compare_strategies(easy, load_eval_queries(), k=5)

        dense = comparison.results["dense"].ndcg_at_k
        rrf = comparison.results["rrf"].ndcg_at_k

        assert rrf <= dense + 0.05, (
            f"Hybrid unexpectedly beat dense by {rrf - dense:+.3f} on the easy "
            f"corpus. Verify this is real before believing it."
        )


class TestHardCorpus:
    """Near-duplicates keyed by exact tokens: this is where methods separate."""

    def test_dense_alone_degrades(self, hard):
        # Six release notes differing only by version number look nearly
        # identical to an embedding model.
        metrics = evaluate(
            _configured(hard, "dense"), load_hard_eval_queries(), k=5
        )

        assert metrics.mrr < 0.85, (
            f"Dense retrieval scored mrr={metrics.mrr:.3f} on near-duplicates. "
            f"If this corpus stopped being hard, the comparison below is "
            f"no longer measuring anything."
        )

    def test_sparse_handles_exact_identifiers(self, hard):
        metrics = evaluate(
            _configured(hard, "sparse"), load_hard_eval_queries(), k=5
        )

        assert metrics.mrr > 0.95

    def test_hybrid_beats_dense(self, hard):
        comparison = compare_strategies(hard, load_hard_eval_queries(), k=5)

        dense = comparison.results["dense"].ndcg_at_k
        rrf = comparison.results["rrf"].ndcg_at_k

        assert rrf > dense + 0.10, (
            f"Hybrid ({rrf:.3f}) should clearly beat dense ({dense:.3f}) "
            f"when exact tokens carry the signal."
        )

    def test_hybrid_loses_to_pure_sparse(self, hard):
        # The finding worth pinning. Fusion mixes dense's wrong answers back
        # into a ranking that sparse had exactly right, so RRF scores BELOW
        # sparse alone. Hybrid is a hedge against not knowing your corpus,
        # not a free improvement over knowing it.
        comparison = compare_strategies(hard, load_hard_eval_queries(), k=5)

        sparse = comparison.results["sparse"].ndcg_at_k
        rrf = comparison.results["rrf"].ndcg_at_k

        assert sparse > rrf, (
            f"Expected pure sparse ({sparse:.3f}) to beat fusion ({rrf:.3f}) "
            f"on a corpus where exact matching is sufficient."
        )


class TestCrossCorpusClaim:
    """The claim the README actually makes."""

    def test_the_best_strategy_differs_by_corpus(self, easy, hard):
        # The whole argument in one assertion: no single strategy wins both,
        # so the choice has to be measured per corpus rather than assumed.
        easy_best = compare_strategies(easy, load_eval_queries(), k=5).best_by()
        hard_best = compare_strategies(
            hard, load_hard_eval_queries(), k=5
        ).best_by()

        assert easy_best != hard_best, (
            f"Both corpora now favour {easy_best!r}. The two are supposed to "
            f"demonstrate opposite conclusions; if they agree, one of them "
            f"has stopped being representative."
        )


def _configured(retriever: HybridRetriever, strategy: str) -> HybridRetriever:
    retriever.config = HybridConfig(limit=5, fetch_k=20, strategy=strategy)
    return retriever
