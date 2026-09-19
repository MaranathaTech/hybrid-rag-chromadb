# CLAUDE.md

Guidance for working on this codebase.

## What this project is

A harness for measuring whether hybrid retrieval beats single-strategy
retrieval **on a given corpus**. It is not an argument that hybrid search is
better. The repo's value is in the measurements, including the ones where the
more complicated approach loses.

## The rule that matters most

**Never tune until hybrid wins.** Two results in this repo are deliberate and
must not be "fixed":

1. On the easy corpus, dense alone beats fusion (nDCG 0.974 vs 0.964).
2. On the hard corpus, pure BM25 beats fusion (1.000 vs 0.904).

Both are pinned by tests (`test_hybrid_does_not_meaningfully_beat_dense`,
`test_hybrid_loses_to_pure_sparse`). If a change makes hybrid win everywhere,
the change is almost certainly overfitting the fixtures, not an improvement.
`test_the_best_strategy_differs_by_corpus` fails loudly if both corpora ever
agree — at that point one of them has stopped being representative.

## Architecture

```
api.py / cli.py      entry points — routing and serialization only
   └── pipeline.py   orchestration: fetch both, fuse, optionally rerank
          ├── dense.py    ChromaDB; converts distance → similarity here
          ├── sparse.py   BM25, self-contained, in-process
          └── fusion.py   pure functions over Retrieved lists
evaluation.py        metrics + strategy comparison
corpus.py            fixtures: the two corpora and labelled queries
types.py             Document, Retrieved, SearchResult (frozen dataclasses)
```

Score convention: **higher is always better**, everywhere. Chroma returns
cosine *distance*, and that inversion is handled at the boundary in `dense.py`
so the convention never leaks into fusion.

`Retrieved` is frozen; use `.with_score()` rather than mutating.

## Testing

- `tests/unit/` — no services, no network. Runs inside the Docker build, so a
  failure fails the image.
- `tests/integration/` — real embeddings and a real (in-memory) Chroma.
  Excluded from the build because Chroma is not reachable at build time.

TDD is the default here. Fusion, BM25 scoring, and the metrics are all cases
where the expected behaviour can be stated before the code exists — write the
failing test from the formula first.

When writing a test that asserts a *relationship* (A beats B), verify it can
actually fail. Three tests in this repo originally passed while measuring
nothing because the fixture made every option score identically. A control that
cannot fail is decoration.

## Gotchas

- **Embedding dimensions bind a collection.** `dense.py` records the provider
  in the collection metadata and refuses a mismatch. Do not remove that check
  to make a test pass — re-ingest or use a different collection name.
- **BM25 must be rebuilt on every boot.** Chroma persists across restarts, the
  in-process BM25 index does not. `api.py` rebuilds it even when Chroma already
  has documents; skipping that leaves sparse retrieval silently returning
  nothing while dense keeps working.
- **`fetch_k` must exceed `limit`.** Fusing two top-5 lists mostly returns the
  same 5 documents. The default fetches 4x.
- **Single API replica is intentional.** The BM25 index is in-process. See
  "Scaling past one replica" in the README before raising it.
- **Smoothed IDF is deliberate.** A term in every document scores slightly
  positive, not negative. Negative IDF penalises a document for containing a
  query term, which is incoherent.

## Deploying

`./run.sh` builds, deploys, waits for rollout, and then queries the running
service to confirm the answers are correct. A green rollout is not evidence the
thing works — keep the smoke test honest, and make it assert on document IDs
rather than on HTTP status.

Verified against Rancher Desktop (k3s v1.33.5). If `kubectl` cannot reach a
cluster, check `DiskPressure` on the node and `docker system df` — build cache
filling the VM disk will evict pods with confusing symptoms.

## Conventions

- Comments explain *why*, especially where a choice looks arbitrary (the `k` in
  RRF, the `b` in BM25, the 4x fetch multiplier).
- Prefer a measurement over an assertion in both code and prose. If the README
  claims a number, `python -m hybrid_rag.cli evaluate` must produce it.
