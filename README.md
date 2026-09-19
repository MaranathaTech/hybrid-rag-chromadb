# Hybrid RAG — ChromaDB + BM25

Dense vector search and BM25 sparse retrieval over the same corpus, combined
with rank fusion, deployed to Kubernetes with one command.

The point of this repo is not that hybrid search is better. It is a harness for
finding out **whether it is better on your corpus** — including the cases where
it is not.

```bash
./run.sh          # build image (runs tests), deploy to local k8s, smoke test
./run.sh --down   # tear it down
```

---

## The measured result

Two corpora, same code, same 5-document cutoff, evaluated with recall, MRR and
nDCG. Run it yourself with `python -m hybrid_rag.cli evaluate`.

**Corpus A — 12 documents on distinct topics** (Kubernetes, vectors, RAG, Postgres):

| strategy | recall@5 | MRR | nDCG@5 |
|----------|---------:|----:|-------:|
| dense    | 1.000 | **0.964** | **0.974** |
| sparse   | 0.929 | 0.893 | 0.902 |
| rrf      | 1.000 | 0.952 | 0.964 |
| weighted | 1.000 | 0.952 | 0.964 |

**Dense alone wins.** Hybrid is very slightly worse, and costs a second index,
a second query, and latency to get there. On this corpus the added complexity
is not justified.

**Corpus B — 15 documents that are near-duplicates**, differing only by version
number, error code, or config key (`release-2.2.1` vs `release-2.3.0`, `E1103`
vs `E1104`, `max_open_files` vs `max_batch_size`):

| strategy | recall@5 | MRR | nDCG@5 |
|----------|---------:|----:|-------:|
| dense    | 0.933 | 0.632 | 0.706 |
| sparse   | 1.000 | **1.000** | **1.000** |
| rrf      | 1.000 | 0.872 | 0.904 |
| weighted | 1.000 | 1.000 | 1.000 |

Two things happen here, and the second is the one usually left out of RAG
write-ups:

1. **Dense retrieval degrades badly** (nDCG 0.974 → 0.706). Six release notes
   that differ by a version number are nearly identical in embedding space.
2. **Pure sparse beats fusion.** BM25 alone is perfect; RRF scores *lower*
   (0.904) because it mixes dense's wrong answers back into a ranking that was
   already exactly right.

### What this actually means

> Hybrid search is a hedge against not knowing your corpus. It is rarely the
> best strategy for a corpus you *do* know — it is the strategy that is never
> catastrophically wrong.

If your queries are paraphrases of your documents, dense alone is likely
enough. If they are identifiers, part numbers, error codes or names, sparse
alone may beat everything. Fusion earns its cost when you have both kinds of
traffic and cannot tell them apart ahead of time.

Both results are pinned as tests in
[`tests/integration/test_retrieval_quality.py`](tests/integration/test_retrieval_quality.py),
including `test_hybrid_loses_to_pure_sparse` and
`test_the_best_strategy_differs_by_corpus`. A benchmark that can only report
wins is not measuring anything.

---

## How it works

```
                 query
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   ChromaDB               BM25 index
   (HNSW, cosine)         (in-process)
        │                     │
   top fetch_k           top fetch_k
        └──────────┬──────────┘
                   ▼
            rank fusion
      (RRF │ weighted │ dense │ sparse)
                   │
          optional cross-encoder rerank
                   │
                   ▼
               top k results
```

### Rank fusion

Dense returns cosine similarities in `[-1, 1]`; BM25 returns unbounded positive
scores. They cannot simply be added.

- **Reciprocal Rank Fusion** (default) discards the scores and uses only rank
  position: `score = Σ weight / (k + rank)`. No per-corpus calibration needed.
- **Weighted score fusion** min-max normalizes each list, then blends with
  `alpha`. More control, but a document's score then depends on which other
  documents came back with it.

Two RRF behaviours worth knowing, both pinned in `tests/unit/test_fusion.py`:

- A document found by **both** retrievers beats one ranked first by only one.
  That consensus effect is the main reason RRF works.
- A document ranked 1st and 3rd **always** beats one ranked 2nd twice, at every
  `k`, because `1/x` is convex. "Lower `k` favours consensus" sounds plausible
  and is false — tuning `k` on that belief is tuning on superstition.

### BM25

Implemented directly rather than pulled from a library, because the parameters
are the interesting part: `k1` caps how much repeating a term helps (so keyword
stuffing cannot win) and `b` controls how hard short documents are favoured.

Uses the `+1`-smoothed IDF, so a term appearing in every document scores a small
positive value rather than the negative one the classic formula gives. Negative
IDF is incoherent for ranking — it penalises a document for *containing* a query
term.

### Embeddings

Local `sentence-transformers/all-MiniLM-L6-v2` by default: **no API key, no
network egress**, baked into the Docker image so a cold pod does not download it.

```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

A collection records which provider built it and **refuses** to be queried by a
different one. Querying 384-dimensional vectors with a 1536-dimensional model
returns plausible-looking neighbours that are meaningless — the kind of failure
that surfaces as "retrieval quality seems a bit off" three weeks later.

---

## Running it

### Local

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pip install -e .

python -m hybrid_rag.cli evaluate                    # the table above
python -m hybrid_rag.cli compare "E1043"             # one query, every strategy
python -m hybrid_rag.cli search "how do I survive a node dying"
python -m hybrid_rag.cli serve                       # API on :8080
```

### Kubernetes

`./run.sh` builds the image, applies the manifests, waits for rollout, and then
**queries the running service to check the answers are correct** — deploying
successfully is not the same as working.

Verified on Rancher Desktop (k3s v1.33.5). Also handles `kind` and `minikube`,
loading the image into the cluster where required.

```
NAME                   READY   STATUS    RESTARTS   AGE
api-67cf4b76f5-69bj2   1/1     Running   0          39s
chroma-0               1/1     Running   0          61s
```

```bash
kubectl port-forward -n hybrid-rag svc/api 8080:8080
curl 'http://localhost:8080/compare?query=E1043'
```

| endpoint | purpose |
|----------|---------|
| `GET /health` | liveness — process is up, does not touch Chroma |
| `GET /ready` | readiness — corpus indexed and queryable |
| `POST /search` | hybrid search (`query`, `limit`, `strategy`, `alpha`, `rerank`) |
| `GET /compare` | one query under all four strategies |

Liveness and readiness are deliberately different: readiness waits for the
embedding model to load and ingestion to finish, so traffic never arrives early.

---

## Tests

```bash
.venv/bin/python -m pytest tests/unit         # 70 tests, no services needed
.venv/bin/python -m pytest tests/integration  # 7 tests, real embeddings + Chroma
```

Unit tests run **inside the Docker build**, so a failing test fails the image
and broken retrieval logic cannot reach the cluster. Integration tests are
excluded from the build because they need a live Chroma that is not reachable at
build time; `./run.sh` runs its smoke test against the deployed stack instead.

---

## Scaling past one replica

The API runs a single replica on purpose. The BM25 index lives in the process,
so each replica builds and holds its own copy, and memory scales with replicas
rather than being shared. The first thing to change for real traffic is moving
sparse retrieval into a service that owns it — Elasticsearch, Tantivy, or
Postgres full-text. The `BM25Retriever` interface would not change.

Chroma is a single StatefulSet with a PVC. Fine to a few hundred thousand
chunks; past that, a clustered vector database earns its keep.

---

## Layout

```
src/hybrid_rag/
  types.py        Document, Retrieved, SearchResult
  embeddings.py   provider abstraction (local | openai)
  dense.py        ChromaDB retriever, distance → similarity at the boundary
  sparse.py       BM25 from scratch
  fusion.py       RRF and weighted score fusion
  pipeline.py     orchestration + optional cross-encoder rerank
  evaluation.py   recall / precision / MRR / nDCG + strategy comparison
  corpus.py       the two corpora and their labelled queries
  api.py          FastAPI service
  cli.py          command line
k8s/              namespace, configmap, Chroma StatefulSet, API Deployment
```

## License

MIT
