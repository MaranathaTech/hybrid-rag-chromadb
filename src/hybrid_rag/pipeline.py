"""The hybrid retrieval pipeline.

Runs dense and sparse retrieval over the same corpus, fuses the rankings, and
optionally reranks the shortlist with a cross-encoder.

Why both retrievers: dense embeddings capture meaning but blur exact tokens —
a query for error code `E1043` or a surname that never appeared in pretraining
can land nowhere near the right chunk. BM25 nails those and is useless at
paraphrase. The failure modes are close to complementary, which is what makes
fusing them worthwhile rather than merely more expensive.

Retrieval depth matters: each retriever fetches `fetch_k` candidates (default
4x the final `limit`) so fusion has a real shortlist to work with. Fusing two
top-5 lists mostly just returns the same 5 documents.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dense import ChromaDenseRetriever
from .fusion import (
    DEFAULT_RRF_K,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from .sparse import BM25Retriever
from .types import Document, Retrieved, SearchResult


@dataclass
class HybridConfig:
    """Tuning knobs for hybrid retrieval."""

    limit: int = 5
    fetch_k: int = 20
    strategy: str = "rrf"  # "rrf" | "weighted" | "dense" | "sparse"
    rrf_k: int = DEFAULT_RRF_K
    alpha: float = 0.5  # weighted strategy only: 1.0 dense, 0.0 sparse
    dense_weight: float = 1.0  # rrf strategy only
    sparse_weight: float = 1.0
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self) -> None:
        valid = {"rrf", "weighted", "dense", "sparse"}
        if self.strategy not in valid:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}. Choose one of: "
                f"{', '.join(sorted(valid))}"
            )
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {self.limit}")
        if self.fetch_k < self.limit:
            raise ValueError(
                f"fetch_k ({self.fetch_k}) must be >= limit ({self.limit}); "
                f"fusion needs a deeper candidate pool than it returns."
            )


class HybridRetriever:
    """Dense + sparse retrieval with rank fusion."""

    def __init__(
        self,
        dense: ChromaDenseRetriever,
        sparse: BM25Retriever | None = None,
        config: HybridConfig | None = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse or BM25Retriever()
        self.config = config or HybridConfig()
        self._reranker = None

    def index(self, documents: list[Document]) -> None:
        """Index into both retrievers.

        Both must see the same corpus — a document missing from one side can
        only ever score on the other, which quietly caps how high it can rank.
        """
        self.dense.index(documents)
        self.sparse.index([(d.id, d.text) for d in documents])

    def search(
        self, query: str, limit: int | None = None
    ) -> SearchResult:
        """Retrieve, fuse, optionally rerank."""
        cfg = self.config
        k = limit or cfg.limit

        dense_hits: list[Retrieved] = []
        sparse_hits: list[Retrieved] = []

        if cfg.strategy != "sparse":
            dense_hits = self.dense.search(query, limit=cfg.fetch_k)
        if cfg.strategy != "dense":
            sparse_hits = self.sparse.search(query, limit=cfg.fetch_k)

        fused = self._fuse(dense_hits, sparse_hits)

        reranked = False
        if cfg.rerank and fused:
            fused = self._rerank(query, fused)
            reranked = True

        return SearchResult(
            documents=fused[:k],
            dense_hits=len(dense_hits),
            sparse_hits=len(sparse_hits),
            reranked=reranked,
        )

    def _fuse(
        self, dense_hits: list[Retrieved], sparse_hits: list[Retrieved]
    ) -> list[Retrieved]:
        cfg = self.config

        if cfg.strategy == "dense":
            return dense_hits
        if cfg.strategy == "sparse":
            return sparse_hits
        if cfg.strategy == "weighted":
            return weighted_score_fusion(dense_hits, sparse_hits, alpha=cfg.alpha)

        return reciprocal_rank_fusion(
            [dense_hits, sparse_hits],
            k=cfg.rrf_k,
            weights=[cfg.dense_weight, cfg.sparse_weight],
        )

    def _rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        """Re-score the shortlist with a cross-encoder.

        A cross-encoder reads the query and document together instead of
        comparing two independently-built vectors, so it is markedly more
        accurate — and far too slow to run over a whole corpus. It only makes
        sense on a shortlist that cheap retrieval has already narrowed, which
        is exactly what fusion just produced.
        """
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.config.rerank_model)

        pairs = [(query, doc.text) for doc in candidates]
        scores = self._reranker.predict(pairs)

        rescored = [
            doc.with_score(float(score)) for doc, score in zip(candidates, scores)
        ]
        return sorted(rescored, key=lambda d: d.score, reverse=True)
