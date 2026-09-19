"""Shared types for the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Document:
    """A chunk of text to be indexed."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Retrieved:
    """A document returned by a retriever, carrying that retriever's score.

    Scores are only comparable within a single retriever's result list — BM25
    scores are unbounded, cosine similarities sit in [-1, 1]. Anything that
    combines lists across retrievers must normalize or rank first.
    """

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_score(self, score: float) -> Retrieved:
        return replace(self, score=score)


@dataclass(frozen=True)
class SearchResult:
    """The final answer from a hybrid search, with its provenance intact."""

    documents: list[Retrieved]
    dense_hits: int
    sparse_hits: int
    reranked: bool = False

    @property
    def ids(self) -> list[str]:
        return [d.id for d in self.documents]
