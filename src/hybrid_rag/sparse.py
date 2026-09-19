"""BM25 sparse retrieval.

Implemented directly rather than pulled from a library, because the parameters
are the interesting part of a hybrid system: `k1` caps how much repeating a
term can help, and `b` controls how hard short documents are favoured. Those
two knobs are why sparse retrieval catches things dense retrieval misses —
exact identifiers, rare proper nouns, error codes — and tuning them is a real
part of the work.

Uses the Robertson/Sparck-Jones IDF with the +1 smoothing variant, which keeps
IDF non-negative for terms appearing in most documents rather than letting them
push scores below zero.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .types import Retrieved

_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """An in-memory BM25 index.

    In-memory is the right call up to roughly a few hundred thousand chunks;
    past that the posting lists belong in something built for it (Elasticsearch,
    Tantivy, or Postgres full-text). The interface here would not change.
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._texts: dict[str, str] = {}
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: Counter[str] = Counter()
        self._n_docs: int = 0

    def index(self, documents: list[tuple[str, str]]) -> None:
        """Build the index from (id, text) pairs, replacing any previous corpus."""
        self._doc_ids = []
        self._texts = {}
        self._term_freqs = []
        self._doc_lengths = []
        self._doc_freqs = Counter()

        for doc_id, text in documents:
            tokens = tokenize(text)
            self._doc_ids.append(doc_id)
            self._texts[doc_id] = text
            self._term_freqs.append(Counter(tokens))
            self._doc_lengths.append(len(tokens))
            for term in set(tokens):
                self._doc_freqs[term] += 1

        self._n_docs = len(self._doc_ids)
        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / self._n_docs if self._n_docs else 0.0

    def _idf(self, term: str) -> float:
        """Inverse document frequency with +1 smoothing."""
        n_containing = self._doc_freqs.get(term, 0)
        if n_containing == 0:
            return 0.0
        return math.log(
            1 + (self._n_docs - n_containing + 0.5) / (n_containing + 0.5)
        )

    def search(self, query: str, limit: int = 10) -> list[Retrieved]:
        """Score every document containing a query term and return the top `limit`."""
        if self._n_docs == 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = {}

        for term in query_terms:
            idf = self._idf(term)
            if idf == 0.0 and self._doc_freqs.get(term, 0) == 0:
                continue

            for doc_idx, term_freq in enumerate(self._term_freqs):
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue

                doc_length = self._doc_lengths[doc_idx]
                length_norm = (
                    1 - self.b + self.b * (doc_length / self._avg_doc_length)
                    if self._avg_doc_length > 0
                    else 1.0
                )
                contribution = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * length_norm)

                doc_id = self._doc_ids[doc_idx]
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [
            Retrieved(id=doc_id, score=score, text=self._texts[doc_id], metadata={})
            for doc_id, score in ranked
        ]
