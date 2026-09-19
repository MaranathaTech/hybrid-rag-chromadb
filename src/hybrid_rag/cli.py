"""Command line interface.

`python -m hybrid_rag.cli evaluate` is the command that matters: it runs the
labelled query set through every strategy and prints the comparison, including
when hybrid loses.
"""

from __future__ import annotations

import argparse
import sys

from .corpus import load_eval_queries, load_sample_corpus
from .dense import ChromaDenseRetriever
from .embeddings import get_embedding_provider
from .evaluation import compare_strategies
from .pipeline import HybridConfig, HybridRetriever
from .sparse import BM25Retriever


def _build_retriever(args: argparse.Namespace) -> HybridRetriever:
    embeddings = get_embedding_provider(args.provider)
    dense = ChromaDenseRetriever(
        embeddings=embeddings,
        collection_name=args.collection,
        host=args.chroma_host,
        port=args.chroma_port,
    )
    retriever = HybridRetriever(dense=dense, sparse=BM25Retriever())

    documents = load_sample_corpus()
    if dense.count() == 0:
        retriever.index(documents)
    else:
        retriever.sparse.index([(d.id, d.text) for d in documents])

    return retriever


def cmd_search(args: argparse.Namespace) -> int:
    retriever = _build_retriever(args)
    retriever.config = HybridConfig(
        limit=args.limit,
        fetch_k=max(args.limit * 4, 20),
        strategy=args.strategy,
        rerank=args.rerank,
    )

    result = retriever.search(args.query, limit=args.limit)

    print(f"\nQuery: {args.query!r}   strategy={args.strategy}")
    print(
        f"(dense returned {result.dense_hits}, sparse returned "
        f"{result.sparse_hits}, reranked={result.reranked})\n"
    )
    for rank, doc in enumerate(result.documents, start=1):
        print(f"{rank}. [{doc.score:+.4f}] {doc.id}")
        print(f"   {doc.text[:150]}...")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Show one query under every strategy."""
    retriever = _build_retriever(args)
    original = retriever.config

    print(f"\nQuery: {args.query!r}\n" + "=" * 72)
    for strategy in ("dense", "sparse", "rrf", "weighted"):
        retriever.config = HybridConfig(
            limit=args.limit, fetch_k=max(args.limit * 4, 20), strategy=strategy
        )
        result = retriever.search(args.query, limit=args.limit)
        top = result.documents[0] if result.documents else None
        print(
            f"{strategy:<10} top hit: "
            + (f"{top.id} ({top.score:+.4f})" if top else "(nothing)")
        )
    retriever.config = original
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the labelled query set through every strategy."""
    retriever = _build_retriever(args)
    queries = load_eval_queries()

    comparison = compare_strategies(retriever, queries, k=args.k)
    print()
    print(comparison.report())
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "hybrid_rag.api:app", host=args.host, port=args.port, log_level="info"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hybrid_rag", description="Hybrid dense + sparse retrieval."
    )
    parser.add_argument("--provider", default=None, help="local | openai")
    parser.add_argument("--collection", default="documents")
    parser.add_argument("--chroma-host", default=None)
    parser.add_argument("--chroma-port", type=int, default=8000)

    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Run a single query")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument(
        "--strategy", default="rrf", choices=["rrf", "weighted", "dense", "sparse"]
    )
    p_search.add_argument("--rerank", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_compare = sub.add_parser("compare", help="One query, every strategy")
    p_compare.add_argument("query")
    p_compare.add_argument("--limit", type=int, default=3)
    p_compare.set_defaults(func=cmd_compare)

    p_eval = sub.add_parser("evaluate", help="Metrics over the labelled query set")
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.set_defaults(func=cmd_evaluate)

    p_serve = sub.add_parser("serve", help="Run the HTTP API")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
