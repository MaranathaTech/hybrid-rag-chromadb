# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-19

### Added
- Dense retrieval over ChromaDB with a swappable embedding provider (local
  sentence-transformers by default, OpenAI opt-in) and a guard that refuses to
  query a collection built by a different model.
- BM25 sparse retrieval implemented from scratch, with configurable `k1`
  (term-frequency saturation) and `b` (length normalisation).
- Reciprocal Rank Fusion and weighted score fusion, plus dense-only and
  sparse-only strategies for comparison.
- Optional cross-encoder reranking of the fused shortlist.
- Evaluation module: recall@k, precision@k, MRR, nDCG@k, and a strategy
  comparison harness that reports when hybrid retrieval *loses*.
- Two corpora: one of distinct topics where dense alone wins, and one of
  near-duplicates keyed by exact identifiers where sparse alone wins.
- FastAPI service with separate liveness and readiness semantics, and a
  `/compare` endpoint showing one query under every strategy.
- CLI: `search`, `compare`, `evaluate`, `serve`.
- Kubernetes manifests (Chroma StatefulSet + PVC, API Deployment, ConfigMap)
  and `run.sh`, which builds, deploys, waits for rollout, and then queries the
  running service to verify the answers are correct.
- Multi-stage Dockerfile that runs the unit suite during the build and bakes
  the embedding model into the image.

### Notes
- Measured on the included corpora, hybrid retrieval does not uniformly win:
  dense alone is marginally better on distinct topics, and pure BM25 beats
  fusion outright on identifier-keyed near-duplicates. Both results are pinned
  as tests so they cannot silently regress.
