"""FastAPI service exposing the hybrid retriever.

Kept deliberately thin: routing and serialization only. All retrieval logic
lives in `pipeline.py`, so the same code path is exercised by the unit tests,
the CLI, and this service.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .corpus import load_sample_corpus
from .dense import ChromaDenseRetriever
from .embeddings import get_embedding_provider
from .pipeline import HybridConfig, HybridRetriever
from .sparse import BM25Retriever

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Populated at startup. The BM25 index is in-process, so a replica must build
# its own on boot — see the README's note on scaling past one replica.
_state: dict[str, object] = {}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    limit: int = Field(5, ge=1, le=50)
    strategy: str = Field("rrf", description="rrf | weighted | dense | sparse")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="weighted strategy only")
    rerank: bool = Field(False, description="Cross-encoder rerank the shortlist")


class RetrievedModel(BaseModel):
    id: str
    score: float
    text: str
    metadata: dict = {}


class SearchResponse(BaseModel):
    query: str
    strategy: str
    documents: list[RetrievedModel]
    dense_hits: int
    sparse_hits: int
    reranked: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the retrievers and seed the corpus once, at startup."""
    provider_name = os.environ.get("EMBEDDING_PROVIDER", "local")
    logger.info("Initializing embedding provider: %s", provider_name)
    embeddings = get_embedding_provider(provider_name)

    chroma_host = os.environ.get("CHROMA_HOST")
    logger.info(
        "Connecting to Chroma: %s",
        f"{chroma_host}:{os.environ.get('CHROMA_PORT', 8000)}"
        if chroma_host
        else "embedded (in-process)",
    )

    dense = ChromaDenseRetriever(
        embeddings=embeddings,
        collection_name=os.environ.get("CHROMA_COLLECTION", "documents"),
        host=chroma_host,
        port=int(os.environ.get("CHROMA_PORT", "8000")),
    )
    retriever = HybridRetriever(dense=dense, sparse=BM25Retriever())

    documents = load_sample_corpus()
    if dense.count() == 0:
        logger.info("Empty collection — ingesting %d sample docs", len(documents))
        retriever.index(documents)
    else:
        # Chroma persisted the vectors across restarts, but BM25 is in-memory
        # and must always be rebuilt or sparse retrieval silently returns
        # nothing while dense keeps working.
        logger.info(
            "Chroma already holds %d docs — rebuilding BM25 index only",
            dense.count(),
        )
        retriever.sparse.index([(d.id, d.text) for d in documents])

    _state["retriever"] = retriever
    _state["provider"] = embeddings.name
    logger.info("Ready: %d documents indexed", dense.count())

    yield
    _state.clear()


app = FastAPI(
    title="Hybrid RAG",
    description="Dense (ChromaDB) + sparse (BM25) retrieval with rank fusion.",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_retriever() -> HybridRetriever:
    retriever = _state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is still starting up")
    return retriever  # type: ignore[return-value]


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up. Deliberately does not touch Chroma."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the corpus is indexed and queries will succeed."""
    retriever = _state.get("retriever")
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    count = retriever.dense.count()  # type: ignore[union-attr]
    if count == 0:
        raise HTTPException(status_code=503, detail="No documents indexed")

    return {
        "status": "ready",
        "documents": count,
        "embedding_provider": _state.get("provider"),
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Run a hybrid search."""
    retriever = _get_retriever()

    try:
        retriever.config = HybridConfig(
            limit=request.limit,
            fetch_k=max(request.limit * 4, 20),
            strategy=request.strategy,
            alpha=request.alpha,
            rerank=request.rerank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = retriever.search(request.query, limit=request.limit)

    return SearchResponse(
        query=request.query,
        strategy=request.strategy,
        documents=[
            RetrievedModel(
                id=d.id, score=round(d.score, 6), text=d.text, metadata=d.metadata
            )
            for d in result.documents
        ],
        dense_hits=result.dense_hits,
        sparse_hits=result.sparse_hits,
        reranked=result.reranked,
    )


@app.get("/compare")
def compare(query: str, limit: int = 5) -> dict:
    """Run one query through every strategy, side by side.

    The endpoint that makes the tradeoff visible: the same query often returns
    a different top hit under dense, sparse, and fused retrieval.
    """
    retriever = _get_retriever()
    original = retriever.config
    output: dict[str, list[dict]] = {}

    try:
        for strategy in ("dense", "sparse", "rrf", "weighted"):
            retriever.config = HybridConfig(
                limit=limit, fetch_k=max(limit * 4, 20), strategy=strategy
            )
            result = retriever.search(query, limit=limit)
            output[strategy] = [
                {"id": d.id, "score": round(d.score, 6), "text": d.text[:120]}
                for d in result.documents
            ]
    finally:
        retriever.config = original

    return {"query": query, "results": output}
