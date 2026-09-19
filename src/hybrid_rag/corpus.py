"""Sample corpus and evaluation set.

Chosen so the two retrievers actually disagree. A corpus where dense retrieval
already wins everything would make the hybrid machinery look good while
demonstrating nothing.

Three query types are represented:

1. **Paraphrase** — "how do I make my app survive a node dying" vs. a document
   about replicas and pod anti-affinity. No shared terms; dense finds it, BM25
   cannot.
2. **Rare identifier** — error codes like `CrashLoopBackOff` or `E1043`.
   Embeddings smear these into a general "error" region; BM25 matches exactly.
3. **Both** — ordinary queries where the two agree, which is most real traffic.
"""

from __future__ import annotations

from .evaluation import EvalQuery
from .types import Document

_DOCS: list[tuple[str, str, dict]] = [
    (
        "k8s-replicas",
        "Running more than one replica of a Deployment keeps an application "
        "available when a single machine goes away unexpectedly. Spread the "
        "copies across failure domains with pod anti-affinity so that losing "
        "one host never takes down every copy at once.",
        {"topic": "kubernetes", "kind": "concept"},
    ),
    (
        "k8s-crashloop",
        "CrashLoopBackOff means the kubelet started your container, the "
        "container exited, and the kubelet is now waiting longer between each "
        "restart attempt. Read the previous container's logs with "
        "kubectl logs --previous to see why it exited.",
        {"topic": "kubernetes", "kind": "troubleshooting"},
    ),
    (
        "k8s-probes",
        "A liveness probe restarts a container that has stopped making "
        "progress. A readiness probe removes a pod from Service endpoints "
        "while it cannot serve traffic. Using a liveness probe where a "
        "readiness probe belongs causes restart storms under load.",
        {"topic": "kubernetes", "kind": "concept"},
    ),
    (
        "k8s-oomkilled",
        "Exit code 137 with reason OOMKilled means the kernel killed the "
        "process because the container exceeded its memory limit. Raise "
        "resources.limits.memory or reduce the workload's footprint.",
        {"topic": "kubernetes", "kind": "troubleshooting"},
    ),
    (
        "vector-hnsw",
        "HNSW builds a navigable small-world graph in layers. Searching walks "
        "the sparse top layer to get close to the target quickly, then "
        "descends into denser layers to refine. The ef_search parameter trades "
        "recall against latency at query time.",
        {"topic": "vectors", "kind": "concept"},
    ),
    (
        "vector-cosine",
        "Cosine similarity compares the direction of two vectors and ignores "
        "their magnitude, which suits text embeddings because document length "
        "should not dominate the comparison. Normalizing vectors first makes "
        "the dot product equivalent to cosine.",
        {"topic": "vectors", "kind": "concept"},
    ),
    (
        "vector-dimension-mismatch",
        "Error E1043: collection expects vectors of width 384 but received "
        "1536. This happens when a collection built with one embedding model "
        "is queried with another. The vectors are not comparable and the "
        "nearest neighbours returned would be meaningless.",
        {"topic": "vectors", "kind": "troubleshooting"},
    ),
    (
        "rag-chunking",
        "Splitting source material into overlapping windows keeps a sentence "
        "near the sentences around it, so an answer is not cut in half at a "
        "boundary. Overlap costs storage and adds near-duplicate results that "
        "a deduplication pass should collapse.",
        {"topic": "rag", "kind": "concept"},
    ),
    (
        "rag-reranking",
        "A cross-encoder reads the question and the candidate passage together "
        "rather than comparing two independently computed vectors, which makes "
        "it considerably more accurate and far too slow to run over an entire "
        "corpus. Apply it only to a shortlist.",
        {"topic": "rag", "kind": "concept"},
    ),
    (
        "rag-hallucination",
        "When retrieval returns nothing relevant, a language model will often "
        "answer from its own parameters anyway and sound equally confident "
        "doing it. Returning an explicit no-results signal beats passing an "
        "empty context and hoping the model declines to answer.",
        {"topic": "rag", "kind": "concept"},
    ),
    (
        "bm25-scoring",
        "BM25 weights a term by how rare it is across the corpus, saturates "
        "the benefit of repeating it within one document, and penalises longer "
        "documents. The k1 parameter controls saturation and b controls length "
        "normalisation.",
        {"topic": "search", "kind": "concept"},
    ),
    (
        "postgres-vacuum",
        "An UPDATE in Postgres writes a new row version and leaves the old one "
        "behind. Autovacuum reclaims that space. When it cannot keep up, "
        "tables grow far beyond their live data and sequential scans slow down "
        "accordingly.",
        {"topic": "postgres", "kind": "concept"},
    ),
]


def load_sample_corpus() -> list[Document]:
    """The demo corpus."""
    return [
        Document(id=doc_id, text=text, metadata=metadata)
        for doc_id, text, metadata in _DOCS
    ]


def load_eval_queries() -> list[EvalQuery]:
    """Queries with hand-labelled relevant documents.

    Labelled by hand against the corpus above. Small enough that every label
    can be checked by reading, which matters more than volume for a demo —
    a large auto-labelled set would mostly measure the labeller.
    """
    return [
        # --- Paraphrase: no term overlap, dense should win -----------------
        EvalQuery.of(
            "how do I keep my service up when a machine dies", "k8s-replicas"
        ),
        EvalQuery.of(
            "why would a container get shut down for using too much RAM",
            "k8s-oomkilled",
        ),
        EvalQuery.of(
            "the model made something up because nothing useful came back",
            "rag-hallucination",
        ),
        EvalQuery.of(
            "comparing direction instead of length when matching text",
            "vector-cosine",
        ),
        # --- Rare identifiers: exact tokens, sparse should win -------------
        EvalQuery.of("CrashLoopBackOff", "k8s-crashloop"),
        EvalQuery.of("E1043", "vector-dimension-mismatch"),
        EvalQuery.of("exit code 137 OOMKilled", "k8s-oomkilled"),
        EvalQuery.of("ef_search parameter", "vector-hnsw"),
        # --- Ordinary queries: both should manage ---------------------------
        EvalQuery.of("liveness and readiness probes", "k8s-probes"),
        EvalQuery.of("cross-encoder reranking", "rag-reranking"),
        EvalQuery.of("chunk overlap when splitting documents", "rag-chunking"),
        EvalQuery.of("BM25 k1 and b parameters", "bm25-scoring"),
        EvalQuery.of("autovacuum and dead tuples", "postgres-vacuum"),
        EvalQuery.of("HNSW graph layers", "vector-hnsw"),
    ]


# ─── Hard corpus ──────────────────────────────────────────────────────────────
#
# The 12-document corpus above is too easy to tell retrievers apart: its topics
# are distinct enough that dense retrieval separates them without help, and
# every strategy scores ~0.97 ndcg. A benchmark where everything wins measures
# nothing.
#
# This corpus is built the way real ones fail. Many documents cover the SAME
# topic and differ only by a specific version, error code, or identifier. The
# right answer is distinguished by an exact token, and its near-misses are
# strong semantic matches — which is precisely where dense retrieval alone
# degrades and exact matching earns its place.

_HARD_DOCS: list[tuple[str, str, dict]] = []
_HARD_QUERIES: list[tuple[str, str]] = []

# Near-identical release notes: same prose, different version and fix.
_RELEASES = [
    ("2.1.4", "connection pool exhaustion under sustained write load"),
    ("2.2.0", "a deadlock between the compaction thread and the WAL writer"),
    ("2.2.1", "incorrect checksum validation on restored snapshots"),
    ("2.3.0", "memory growth in the query planner on deeply nested joins"),
    ("2.3.1", "a race condition when two clients create the same index"),
    ("3.0.0", "silent truncation of values larger than the page size"),
]
for version, fix in _RELEASES:
    _HARD_DOCS.append((
        f"release-{version}",
        f"Release {version} of the storage engine. This version fixes {fix}. "
        f"Upgrading from an earlier release requires no migration and the "
        f"on-disk format is unchanged. Restart each node in turn.",
        {"topic": "releases", "version": version},
    ))
    _HARD_QUERIES.append((f"what did release {version} fix", f"release-{version}"))

# Near-identical error pages: same shape, different code.
_ERRORS = [
    ("E1102", "the request body exceeded the configured maximum size"),
    ("E1103", "the request was rejected because the content type was unsupported"),
    ("E1104", "the upstream service closed the connection before responding"),
    ("E1105", "the authentication token was well-formed but had expired"),
    ("E1106", "the requested resource exists but is not readable by this role"),
]
for code, meaning in _ERRORS:
    _HARD_DOCS.append((
        f"error-{code}",
        f"Error {code}. This status is returned when {meaning}. Check the "
        f"gateway logs for the corresponding request id and retry once the "
        f"underlying condition is resolved.",
        {"topic": "errors", "code": code},
    ))
    _HARD_QUERIES.append((f"{code}", f"error-{code}"))

# Near-identical config keys: same phrasing, different knob.
_SETTINGS = [
    ("max_connections", "how many client sockets the server accepts at once"),
    ("max_open_files", "how many file descriptors the storage layer may hold"),
    ("max_batch_size", "how many rows are grouped into a single write"),
    ("max_idle_time", "how long an unused connection is kept before closing"),
]
for key, meaning in _SETTINGS:
    _HARD_DOCS.append((
        f"config-{key}",
        f"The {key} setting controls {meaning}. Raising it increases "
        f"throughput at the cost of memory. The default suits most "
        f"deployments and should only be changed with measurements in hand.",
        {"topic": "config", "key": key},
    ))
    _HARD_QUERIES.append((f"{key} setting", f"config-{key}"))


def load_hard_corpus() -> list[Document]:
    """A corpus where near-duplicates make exact matching matter."""
    return [
        Document(id=doc_id, text=text, metadata=metadata)
        for doc_id, text, metadata in _HARD_DOCS
    ]


def load_hard_eval_queries() -> list[EvalQuery]:
    """Queries whose answers differ from their neighbours by one token."""
    return [EvalQuery.of(query, doc_id) for query, doc_id in _HARD_QUERIES]
