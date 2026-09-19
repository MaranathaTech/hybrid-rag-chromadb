"""Tests for the BM25 sparse retriever.

BM25 is scored from scratch here rather than pulled from a library, because the
parameters (k1, b) are the part worth showing: they are what makes sparse
retrieval behave differently from dense, and the difference is the whole
argument for hybrid search.
"""

import math

import pytest

from hybrid_rag.sparse import BM25Retriever, tokenize


CORPUS = [
    ("d1", "the quick brown fox jumps over the lazy dog"),
    ("d2", "kubernetes orchestrates containers across a cluster of nodes"),
    ("d3", "a fox is a small wild animal related to the dog"),
    ("d4", "container orchestration with kubernetes and helm charts"),
]


@pytest.fixture
def bm25():
    retriever = BM25Retriever()
    retriever.index([(doc_id, text) for doc_id, text in CORPUS])
    return retriever


class TestTokenize:
    def test_lowercases_and_splits(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenize("fox, dog; cat!") == ["fox", "dog", "cat"]

    def test_keeps_alphanumerics_together(self):
        assert tokenize("k8s v1.31 node2") == ["k8s", "v1", "31", "node2"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   \n\t ") == []


class TestBM25Scoring:
    def test_exact_term_match_ranks_first(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)

        results = retriever.search("kubernetes", limit=4)

        assert results[0].id in {"d2", "d4"}

    def test_rare_term_outweighs_common_term(self):
        # A term in 1 of 5 docs must dominate one in 4 of 5. Note this needs a
        # corpus where the two terms really do differ in document frequency:
        # in a tiny corpus "the" and a rare noun can share a df and produce
        # identical IDF, which makes the test assert nothing.
        retriever = BM25Retriever()
        retriever.index([
            ("rare", "the kubernetes scheduler"),
            ("common1", "the quick fox"),
            ("common2", "the lazy dog"),
            ("common3", "the small cat"),
            ("common4", "the tall tree"),
        ])

        results = retriever.search("the kubernetes", limit=5)

        assert results[0].id == "rare"

    def test_term_in_every_document_still_scores_above_zero(self):
        # This implementation uses the +1-smoothed IDF, so a term appearing in
        # every document yields a small positive score rather than the negative
        # one the classic formula produces. Negative IDF is incoherent for
        # ranking — it penalises a document for containing a query term — so
        # the smoothing is deliberate. Pinned here so it is a decision, not an
        # accident someone "fixes" later.
        retriever = BM25Retriever()
        retriever.index([
            ("a", "shared term alpha"),
            ("b", "shared term beta"),
        ])

        results = retriever.search("shared", limit=2)

        assert all(0.0 < res.score < 0.5 for res in results)

    def test_rare_term_scores_far_above_a_universal_term(self):
        # The practical consequence of the above: smoothing keeps universal
        # terms positive, but they must still be dwarfed by discriminating ones.
        retriever = BM25Retriever()
        retriever.index([
            ("a", "shared alpha"),
            ("b", "shared beta"),
            ("c", "shared gamma"),
        ])

        universal = retriever.search("shared", limit=1)[0].score
        rare = retriever.search("alpha", limit=1)[0].score

        assert rare > universal * 3

    def test_no_match_returns_no_results(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)

        results = retriever.search("antidisestablishmentarianism", limit=4)

        assert results == []

    def test_shorter_document_wins_on_equal_term_frequency(self):
        # Length normalisation (the b parameter) should favour the document
        # where the matched term is a larger share of the content.
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index([
            ("short", "python"),
            ("long", "python " + " ".join(["filler"] * 50)),
        ])

        results = retriever.search("python", limit=2)

        assert results[0].id == "short"

    def test_b_zero_disables_length_normalisation(self):
        retriever = BM25Retriever(k1=1.5, b=0.0)
        retriever.index([
            ("short", "python"),
            ("long", "python " + " ".join(["filler"] * 50)),
        ])

        results = retriever.search("python", limit=2)

        assert results[0].score == pytest.approx(results[1].score)

    def test_term_frequency_saturates(self):
        # BM25's k1 caps the payoff of repeating a term. Ten occurrences must
        # not score ten times one occurrence, or keyword stuffing wins.
        retriever = BM25Retriever(k1=1.2, b=0.0)
        retriever.index([
            ("once", "python other words here to pad it out a bit"),
            ("many", "python python python python python python python python python python"),
        ])

        results = retriever.search("python", limit=2)
        by_id = {res.id: res.score for res in results}

        assert by_id["many"] < by_id["once"] * 10

    def test_limit_truncates_results(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)

        results = retriever.search("fox dog kubernetes container", limit=2)

        assert len(results) == 2

    def test_results_are_sorted_descending(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)

        results = retriever.search("fox dog", limit=4)
        scores = [res.score for res in results]

        assert scores == sorted(scores, reverse=True)

    def test_multi_term_query_sums_contributions(self):
        retriever = BM25Retriever()
        retriever.index(CORPUS)

        both = retriever.search("fox dog", limit=4)
        just_fox = retriever.search("fox", limit=4)

        both_d3 = next(res.score for res in both if res.id == "d3")
        fox_d3 = next(res.score for res in just_fox if res.id == "d3")

        assert both_d3 > fox_d3


class TestBM25Lifecycle:
    def test_search_before_index_returns_empty(self):
        assert BM25Retriever().search("anything", limit=5) == []

    def test_indexing_empty_corpus(self):
        retriever = BM25Retriever()
        retriever.index([])

        assert retriever.search("anything", limit=5) == []

    def test_reindex_replaces_the_corpus(self):
        retriever = BM25Retriever()
        retriever.index([("old", "kubernetes")])
        retriever.index([("new", "postgres")])

        assert retriever.search("kubernetes", limit=5) == []
        assert retriever.search("postgres", limit=5)[0].id == "new"

    def test_document_with_no_tokens_is_indexed_without_crashing(self):
        retriever = BM25Retriever()
        retriever.index([("empty", "!!!"), ("real", "kubernetes")])

        results = retriever.search("kubernetes", limit=5)

        assert results[0].id == "real"

    def test_text_is_returned_with_results(self):
        retriever = BM25Retriever()
        retriever.index([("d1", "kubernetes cluster")])

        results = retriever.search("kubernetes", limit=1)

        assert results[0].text == "kubernetes cluster"

    def test_idf_uses_the_documented_formula(self):
        # Pinning the formula so a future "optimisation" cannot quietly
        # change ranking behaviour.
        retriever = BM25Retriever(k1=1.2, b=0.0)
        retriever.index([
            ("a", "alpha"),
            ("b", "beta"),
            ("c", "gamma"),
            ("d", "delta"),
        ])

        results = retriever.search("alpha", limit=1)

        n_docs, n_containing = 4, 1
        expected_idf = math.log(1 + (n_docs - n_containing + 0.5) / (n_containing + 0.5))
        # tf=1, doclen == avgdl, b=0 -> tf component is 1*(k1+1)/(1+k1)
        expected = expected_idf * (1 * (1.2 + 1)) / (1 + 1.2)

        assert results[0].score == pytest.approx(expected)
