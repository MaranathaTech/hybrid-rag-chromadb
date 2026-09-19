"""Tests for retrieval metrics.

Metrics are the measuring instrument: if they are wrong, every claim the
project makes about retrieval quality is wrong with them. These pin each
formula against hand-computed values rather than against the implementation.
"""

import math

import pytest

from hybrid_rag.evaluation import (
    Comparison,
    EvalQuery,
    Metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


RELEVANT = frozenset({"a", "b"})


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b", "x"], RELEVANT, k=3) == 1.0

    def test_half_retrieved(self):
        assert recall_at_k(["a", "x", "y"], RELEVANT, k=3) == 0.5

    def test_none_retrieved(self):
        assert recall_at_k(["x", "y"], RELEVANT, k=2) == 0.0

    def test_k_truncates_before_counting(self):
        # "b" sits at position 3 and must not count when k=2.
        assert recall_at_k(["a", "x", "b"], RELEVANT, k=2) == 0.5

    def test_empty_relevant_set_is_zero_not_a_crash(self):
        assert recall_at_k(["a"], frozenset(), k=1) == 0.0

    def test_duplicate_ids_do_not_inflate(self):
        # A retriever returning the same document twice must not score 1.0.
        assert recall_at_k(["a", "a"], RELEVANT, k=2) == 0.5


class TestPrecisionAtK:
    def test_all_top_k_relevant(self):
        assert precision_at_k(["a", "b"], RELEVANT, k=2) == 1.0

    def test_half_precision(self):
        assert precision_at_k(["a", "x"], RELEVANT, k=2) == 0.5

    def test_divides_by_k_not_by_result_count(self):
        # Only one result returned but k=5: precision is 1/5, not 1/1.
        # Dividing by the result count would let a retriever score a perfect
        # precision by returning a single lucky hit.
        assert precision_at_k(["a"], RELEVANT, k=5) == pytest.approx(0.2)

    def test_k_zero(self):
        assert precision_at_k(["a"], RELEVANT, k=0) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "x"], RELEVANT) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], RELEVANT) == 0.5

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "b"], RELEVANT) == pytest.approx(1 / 3)

    def test_no_hit(self):
        assert reciprocal_rank(["x", "y"], RELEVANT) == 0.0

    def test_only_the_first_hit_counts(self):
        # Both "a" and "b" are relevant; RR is about the first one only.
        assert reciprocal_rank(["x", "a", "b"], RELEVANT) == 0.5

    def test_empty_results(self):
        assert reciprocal_rank([], RELEVANT) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b", "x"], RELEVANT, k=3) == pytest.approx(1.0)

    def test_no_hits_is_zero(self):
        assert ndcg_at_k(["x", "y"], RELEVANT, k=2) == 0.0

    def test_order_matters_unlike_recall(self):
        # Same documents, different order. This is the property that makes
        # ndcg worth computing alongside recall: fusion reorders shortlists.
        good = ndcg_at_k(["a", "b", "x"], RELEVANT, k=3)
        bad = ndcg_at_k(["x", "a", "b"], RELEVANT, k=3)

        assert good > bad
        assert recall_at_k(["a", "b", "x"], RELEVANT, 3) == recall_at_k(
            ["x", "a", "b"], RELEVANT, 3
        )

    def test_matches_hand_computed_value(self):
        # Hits at ranks 2 and 3: dcg = 1/log2(3) + 1/log2(4)
        # Ideal (ranks 1, 2):     idcg = 1/log2(2) + 1/log2(3)
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)

        assert ndcg_at_k(["x", "a", "b"], RELEVANT, k=3) == pytest.approx(dcg / idcg)

    def test_idcg_accounts_for_k_smaller_than_relevant_set(self):
        # Three relevant docs but k=1: retrieving one of them at rank 1 is the
        # best achievable, so ndcg must be 1.0, not 1/3. An idcg that assumed
        # all three could fit would make a perfect retriever look broken.
        relevant = frozenset({"a", "b", "c"})

        assert ndcg_at_k(["a"], relevant, k=1) == pytest.approx(1.0)

    def test_empty_relevant_set(self):
        assert ndcg_at_k(["a"], frozenset(), k=1) == 0.0


class TestEvalQuery:
    def test_of_builds_a_frozen_relevant_set(self):
        q = EvalQuery.of("what is kubernetes", "d1", "d2")

        assert q.query == "what is kubernetes"
        assert q.relevant_ids == frozenset({"d1", "d2"})

    def test_duplicate_ids_collapse(self):
        assert EvalQuery.of("q", "d1", "d1").relevant_ids == frozenset({"d1"})


class TestComparisonReport:
    def _metrics(self, ndcg: float) -> Metrics:
        return Metrics(
            recall_at_k=ndcg,
            precision_at_k=ndcg,
            mrr=ndcg,
            ndcg_at_k=ndcg,
            k=5,
            n_queries=10,
        )

    def test_reports_a_hybrid_win(self):
        comparison = Comparison(
            results={"dense": self._metrics(0.60), "rrf": self._metrics(0.80)}
        )

        report = comparison.report()

        assert "Hybrid beats dense-only" in report
        assert comparison.best_by() == "rrf"

    def test_reports_a_hybrid_loss_plainly(self):
        # The report must be willing to say the added complexity lost. A
        # harness that can only announce wins is not measuring anything.
        comparison = Comparison(
            results={"dense": self._metrics(0.90), "rrf": self._metrics(0.70)}
        )

        report = comparison.report()

        assert "LOSES" in report
        assert comparison.best_by() == "dense"

    def test_reports_a_tie_as_a_tie(self):
        comparison = Comparison(
            results={"dense": self._metrics(0.800), "rrf": self._metrics(0.805)}
        )

        report = comparison.report()

        assert "indistinguishable" in report
        assert "Prefer the simpler one" in report
