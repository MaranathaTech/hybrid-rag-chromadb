"""Tests for rank fusion.

Fusion is the heart of hybrid search: it decides how a dense ranking and a
sparse ranking combine into one. These tests pin the scoring rules directly,
including the cases where fusion must *not* help — a fusion that improves
every ranking it touches is not measuring anything.
"""

import pytest

from hybrid_rag.fusion import (
    reciprocal_rank_fusion,
    weighted_score_fusion,
    normalize_scores,
)
from hybrid_rag.types import Retrieved


def r(doc_id: str, score: float) -> Retrieved:
    return Retrieved(id=doc_id, score=score, text=f"text of {doc_id}", metadata={})


class TestReciprocalRankFusion:
    def test_document_ranked_first_by_both_wins(self):
        dense = [r("a", 0.9), r("b", 0.8), r("c", 0.7)]
        sparse = [r("a", 12.0), r("c", 8.0), r("b", 3.0)]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        assert fused[0].id == "a"

    def test_appearing_in_both_lists_beats_a_lone_first_place(self):
        # This is the consensus effect RRF actually provides: "b" is second in
        # both rankings and beats "a", which is first in one and absent from
        # the other. Being found twice is what wins.
        dense = [r("a", 0.99), r("b", 0.5)]
        sparse = [r("c", 20.0), r("b", 19.0)]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        assert fused[0].id == "b"

    def test_first_and_third_still_beats_second_twice(self):
        # The consensus effect has a real limit, and it is not a function of k.
        # A document ranked 1st and 3rd always beats one ranked 2nd twice,
        # at EVERY k, because 1/x is convex: 1/(k+1) + 1/(k+3) > 2/(k+2).
        # Worth pinning, because "lower k favours consensus" is a plausible-
        # sounding rule that happens to be false, and tuning k on that belief
        # would be tuning on superstition.
        dense = [r("a", 0.99), r("b", 0.5), r("c", 0.4)]
        sparse = [r("c", 20.0), r("b", 19.0), r("a", 0.1)]

        for k in (1, 2, 10, 60, 600):
            assert reciprocal_rank_fusion([dense, sparse], k=k)[0].id == "a", (
                f"expected 'a' to win at k={k}"
            )

    def test_rrf_score_matches_the_formula(self):
        dense = [r("a", 0.9), r("b", 0.8)]
        sparse = [r("b", 5.0), r("a", 4.0)]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        # Ranks are 1-based: "a" is rank 1 dense, rank 2 sparse.
        expected_a = 1 / 61 + 1 / 62
        by_id = {d.id: d for d in fused}
        assert by_id["a"].score == pytest.approx(expected_a)

    def test_k_controls_how_much_top_ranks_dominate(self):
        # With a small k, rank 0 is worth far more than rank 3.
        # With a large k, the gap between ranks narrows.
        dense = [r("a", 0.9), r("b", 0.8), r("c", 0.7), r("d", 0.6)]

        small_k = reciprocal_rank_fusion([dense], k=1)
        large_k = reciprocal_rank_fusion([dense], k=1000)

        small_ratio = small_k[0].score / small_k[3].score
        large_ratio = large_k[0].score / large_k[3].score

        assert small_ratio > large_ratio

    def test_document_in_only_one_list_still_appears(self):
        dense = [r("a", 0.9)]
        sparse = [r("b", 5.0)]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        assert {d.id for d in fused} == {"a", "b"}

    def test_identical_lists_preserve_the_original_order(self):
        # A fusion that scrambles agreeing rankings is broken.
        ranking = [r("a", 0.9), r("b", 0.8), r("c", 0.7)]

        fused = reciprocal_rank_fusion([ranking, ranking], k=60)

        assert [d.id for d in fused] == ["a", "b", "c"]

    def test_fusion_is_order_independent_across_retrievers(self):
        dense = [r("a", 0.9), r("b", 0.8)]
        sparse = [r("b", 5.0), r("c", 4.0)]

        one = reciprocal_rank_fusion([dense, sparse], k=60)
        two = reciprocal_rank_fusion([sparse, dense], k=60)

        assert [d.id for d in one] == [d.id for d in two]

    def test_weights_shift_the_winner(self):
        dense = [r("a", 0.9), r("b", 0.1)]
        sparse = [r("b", 9.0), r("a", 0.1)]

        dense_heavy = reciprocal_rank_fusion([dense, sparse], k=60, weights=[3.0, 1.0])
        sparse_heavy = reciprocal_rank_fusion([dense, sparse], k=60, weights=[1.0, 3.0])

        assert dense_heavy[0].id == "a"
        assert sparse_heavy[0].id == "b"

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[], []], k=60) == []

    def test_mismatched_weights_length_is_rejected(self):
        dense = [r("a", 0.9)]
        with pytest.raises(ValueError, match="weights"):
            reciprocal_rank_fusion([dense], k=60, weights=[1.0, 2.0])

    def test_text_and_metadata_survive_fusion(self):
        dense = [Retrieved(id="a", score=0.9, text="hello", metadata={"src": "x"})]

        fused = reciprocal_rank_fusion([dense], k=60)

        assert fused[0].text == "hello"
        assert fused[0].metadata == {"src": "x"}


class TestNormalizeScores:
    def test_maps_to_unit_interval(self):
        scores = [r("a", 10.0), r("b", 5.0), r("c", 0.0)]

        normalized = normalize_scores(scores)

        assert [d.score for d in normalized] == pytest.approx([1.0, 0.5, 0.0])

    def test_identical_scores_do_not_divide_by_zero(self):
        # Every score equal means the retriever has no opinion. Collapsing that
        # to 1.0 would silently promote a uniform list over a discriminating one.
        scores = [r("a", 3.0), r("b", 3.0)]

        normalized = normalize_scores(scores)

        assert all(d.score == 0.0 for d in normalized)

    def test_single_document_is_not_forced_to_one(self):
        normalized = normalize_scores([r("a", 7.0)])
        assert normalized[0].score == 0.0

    def test_empty_list(self):
        assert normalize_scores([]) == []

    def test_negative_scores_are_handled(self):
        # Cosine distance and some rerankers emit negatives.
        scores = [r("a", -1.0), r("b", -5.0)]

        normalized = normalize_scores(scores)

        assert [d.score for d in normalized] == pytest.approx([1.0, 0.0])


class TestWeightedScoreFusion:
    def test_alpha_one_is_pure_dense(self):
        dense = [r("a", 1.0), r("b", 0.0)]
        sparse = [r("b", 1.0), r("a", 0.0)]

        fused = weighted_score_fusion(dense, sparse, alpha=1.0)

        assert fused[0].id == "a"

    def test_alpha_zero_is_pure_sparse(self):
        dense = [r("a", 1.0), r("b", 0.0)]
        sparse = [r("b", 1.0), r("a", 0.0)]

        fused = weighted_score_fusion(dense, sparse, alpha=0.0)

        assert fused[0].id == "b"

    def test_missing_from_one_retriever_scores_zero_on_that_side(self):
        dense = [r("a", 1.0)]
        sparse = [r("b", 1.0)]

        fused = weighted_score_fusion(dense, sparse, alpha=0.5)

        by_id = {d.id: d.score for d in fused}
        assert by_id["a"] == pytest.approx(0.5)
        assert by_id["b"] == pytest.approx(0.5)

    def test_alpha_outside_unit_interval_is_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            weighted_score_fusion([], [], alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            weighted_score_fusion([], [], alpha=-0.1)
