# Completeness

Where this project actually stands. Updated whenever a feature moves between
states.

## Built and verified

| Area | Status | Evidence |
|------|--------|----------|
| BM25 sparse retrieval | Done | 24 unit tests, IDF formula pinned |
| Rank fusion (RRF + weighted) | Done | 22 unit tests incl. convexity behaviour |
| Evaluation metrics | Done | 24 unit tests, each formula hand-computed |
| ChromaDB dense retrieval | Done | exercised by 7 integration tests |
| Embedding providers (local/OpenAI) | Done | local verified end to end |
| Retrieval quality claims | Done | 7 integration tests pin both corpora |
| FastAPI service | Done | `/health`, `/ready`, `/search`, `/compare` |
| CLI | Done | `search`, `compare`, `evaluate`, `serve` |
| Docker build with tests inside | Done | build fails if unit tests fail |
| Local k8s deploy | **Verified on cluster** | Rancher Desktop k3s v1.33.5; pods Ready, smoke test passed on both query types |

## Partially built

| Area | State | What is missing |
|------|-------|-----------------|
| Cross-encoder reranking | Implemented, lightly tested | No integration test measuring its effect on nDCG; the model is not baked into the image, so first use downloads it |
| OpenAI embedding provider | Implemented, untested against the live API | Needs a key; the local path is the default and is covered |

## Not started

- **Persistent BM25 index.** In-process only, so the API is pinned to one
  replica. Moving sparse retrieval to Elasticsearch/Tantivy/Postgres FTS is the
  first real-traffic change.
- **Ingestion of user documents.** The corpora are fixed fixtures; there is no
  upload endpoint or chunking pipeline.
- **Query-type routing.** The measurements show dense and sparse each win on
  different query shapes. Classifying a query and routing it is the obvious
  follow-up and is not implemented.
- **Authentication.** The API is unauthenticated; it is a demo, not a service.
