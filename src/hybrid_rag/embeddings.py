"""Embedding providers.

Local sentence-transformers is the default so the project runs with no API key
and no network egress. Setting `EMBEDDING_PROVIDER=openai` swaps in the API
without touching anything downstream — the retrievers only see `embed()`.

The provider abstraction is not architecture for its own sake: embedding
dimensions differ between providers (384 for all-MiniLM-L6-v2, 1536 for
text-embedding-3-small), and a collection built with one cannot be queried with
another. `dimension` is exposed so the vector store can refuse the mismatch
loudly instead of returning nonsense neighbours.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


class EmbeddingProvider(ABC):
    """Turns text into vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector width. Collections are bound to this."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded alongside a collection, to catch mismatches."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Split from `embed` because asymmetric models (E5, BGE, GTE) require a
        different prefix for queries than for documents, and using the document
        path for a query silently degrades recall.
        """
        return self.embed([text])[0]


class LocalEmbeddings(EmbeddingProvider):
    """sentence-transformers running in-process. No key, no network."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        self._model_name = model_name
        self._model = None  # Loaded lazily; the import is slow and heavy.

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings. Requires OPENAI_API_KEY."""

    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Either export it, or use the "
                "default local provider (EMBEDDING_PROVIDER=local)."
            )
        self._client = None

    def _load(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        if self._model_name not in self._DIMENSIONS:
            raise ValueError(
                f"Unknown embedding model {self._model_name!r}; "
                f"known models: {', '.join(sorted(self._DIMENSIONS))}"
            )
        return self._DIMENSIONS[self._model_name]

    @property
    def name(self) -> str:
        return f"openai:{self._model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._load().embeddings.create(
            model=self._model_name, input=texts
        )
        return [item.embedding for item in response.data]


def get_embedding_provider(
    provider: str | None = None,
    model_name: str | None = None,
) -> EmbeddingProvider:
    """Build a provider from an explicit name or the environment.

    Defaults to local so a fresh clone runs offline.
    """
    resolved = (provider or os.environ.get("EMBEDDING_PROVIDER", "local")).lower()

    if resolved == "local":
        return LocalEmbeddings(model_name or os.environ.get(
            "EMBEDDING_MODEL", DEFAULT_LOCAL_MODEL
        ))
    if resolved == "openai":
        return OpenAIEmbeddings(model_name or os.environ.get(
            "EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL
        ))

    raise ValueError(
        f"Unknown embedding provider {resolved!r}. Supported: local, openai."
    )
