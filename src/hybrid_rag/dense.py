"""Dense retrieval over ChromaDB.

Chroma is handed pre-computed vectors rather than one of its own embedding
functions, so the embedding provider stays swappable and the same vectors can
feed a reranker or a second index without recomputation.

Chroma returns *distances*, not similarities. Lower is better there, higher is
better everywhere else in this codebase, so the conversion happens here at the
boundary rather than leaking an inverted convention into fusion.
"""

from __future__ import annotations

from .embeddings import EmbeddingProvider
from .types import Document, Retrieved

# Chroma collection metadata key recording which provider built the vectors.
_PROVIDER_KEY = "embedding_provider"


class ChromaDenseRetriever:
    """Vector search over a Chroma collection."""

    def __init__(
        self,
        embeddings: EmbeddingProvider,
        collection_name: str = "documents",
        host: str | None = None,
        port: int = 8000,
        persist_directory: str | None = None,
    ) -> None:
        """Connect to Chroma.

        With `host`, talks to a Chroma server (how it runs in k8s). Without,
        uses an embedded client — persistent if `persist_directory` is given,
        in-memory otherwise, which is what the tests use.
        """
        self.embeddings = embeddings
        self.collection_name = collection_name
        self._client = self._build_client(host, port, persist_directory)
        self._collection = self._ensure_collection()

    def _build_client(
        self, host: str | None, port: int, persist_directory: str | None
    ):
        import chromadb

        if host:
            return chromadb.HttpClient(host=host, port=port)
        if persist_directory:
            return chromadb.PersistentClient(path=persist_directory)
        return chromadb.EphemeralClient()

    def _ensure_collection(self):
        """Fetch or create the collection, refusing a provider mismatch.

        Querying a collection with vectors from a different model returns
        plausible-looking neighbours that are actually meaningless. That is the
        kind of failure that shows up as "retrieval quality is a bit off"
        weeks later, so it fails loudly here instead.
        """
        existing = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                _PROVIDER_KEY: self.embeddings.name,
                # Cosine, not Chroma's L2 default: embeddings are normalized,
                # and cosine keeps scores in a range fusion can reason about.
                "hnsw:space": "cosine",
            },
        )

        recorded = (existing.metadata or {}).get(_PROVIDER_KEY)
        if recorded and recorded != self.embeddings.name:
            raise ValueError(
                f"Collection {self.collection_name!r} was built with "
                f"{recorded!r} but the configured provider is "
                f"{self.embeddings.name!r}. Vectors from different models are "
                f"not comparable — re-ingest, or use a different collection name."
            )
        return existing

    def index(self, documents: list[Document]) -> None:
        """Embed and upsert documents."""
        if not documents:
            return

        vectors = self.embeddings.embed([d.text for d in documents])
        self._collection.upsert(
            ids=[d.id for d in documents],
            embeddings=vectors,
            documents=[d.text for d in documents],
            # Chroma rejects empty metadata dicts, so give it a real key.
            metadatas=[d.metadata or {"_": ""} for d in documents],
        )

    def search(self, query: str, limit: int = 10) -> list[Retrieved]:
        """Return the nearest documents, scored so that higher is better."""
        if self.count() == 0:
            return []

        query_vector = self.embeddings.embed_query(query)
        response = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(limit, self.count()),
        )

        ids = response["ids"][0]
        distances = response["distances"][0]
        texts = response["documents"][0]
        metadatas = response["metadatas"][0] or [{}] * len(ids)

        return [
            Retrieved(
                id=doc_id,
                # Cosine distance in [0, 2] -> similarity in [-1, 1].
                score=1.0 - distance,
                text=text,
                metadata={k: v for k, v in (meta or {}).items() if k != "_"},
            )
            for doc_id, distance, text, meta in zip(ids, distances, texts, metadatas)
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._ensure_collection()
