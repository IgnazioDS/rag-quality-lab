"""Run the full benchmark over selected chunking strategies and retrievers.

Usage:
    python scripts/run_benchmark.py [--corpus squad-200] [--all-strategies]
        [--strategies fixed recursive semantic] [--all-retrievers]
        [--retrievers dense sparse hybrid] [--k 1 3 5 10] [--llm-judge]
"""
import argparse
import logging
import sys
from pathlib import Path

import psycopg2
from rich.console import Console
from rich.table import Table

sys.path.insert(0, ".")
from rag_quality_lab.chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker
from rag_quality_lab.config import config
from rag_quality_lab.embedders.openai import OpenAIEmbedder
from rag_quality_lab.pipeline import BenchmarkPipeline
from rag_quality_lab.retrievers.dense import DenseRetriever
from rag_quality_lab.retrievers.hybrid import HybridRetriever
from rag_quality_lab.retrievers.sparse import SparseRetriever

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_chunkers(strategies: list[str], embedder: object) -> list[object]:
    chunkers = []
    for strategy in strategies:
        if strategy == "fixed":
            chunkers.append(FixedSizeChunker())
        elif strategy == "recursive":
            chunkers.append(RecursiveChunker())
        elif strategy == "semantic":
            chunkers.append(SemanticChunker(embedder=embedder))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return chunkers


def build_retriever_configs(
    retriever_names: list[str],
    conn: object,
    embedder: object,
    corpus_id: str,
    corpus_chunks: list,
) -> list[dict]:
    configs = []

    dense = None
    sparse = None

    if "dense" in retriever_names or "hybrid" in retriever_names:
        dense = DenseRetriever(conn=conn, embedder=embedder, corpus_id=corpus_id)

    if "sparse" in retriever_names or "hybrid" in retriever_names:
        sparse = SparseRetriever(corpus_chunks=corpus_chunks)

    if "dense" in retriever_names and dense is not None:
        configs.append({"name": "dense", "retriever": dense, "params": {}})

    if "sparse" in retriever_names and sparse is not None:
        configs.append({"name": "sparse", "retriever": sparse, "params": {}})

    if "hybrid" in retriever_names and dense is not None and sparse is not None:
        hybrid = HybridRetriever(dense=dense, sparse=sparse)
        configs.append({"name": "hybrid", "retriever": hybrid, "params": {"rrf_k": 60}})

    return configs


def print_summary(runs: list) -> None:
    table = Table(title="Benchmark Summary", show_header=True, header_style="bold magenta")
    table.add_column("Chunker", style="cyan")
    table.add_column("Retriever", style="green")
    table.add_column("Hit Rate @5", justify="right")
    table.add_column("MRR @5", justify="right")
    table.add_column("NDCG @5", justify="right")
    table.add_column("P50 (ms)", justify="right")

    for run in runs:
        agg = run.aggregated_metrics
        timing = run.timing
        table.add_row(
            run.chunker_name,
            run.retriever_name,
            f"{agg.get('hit_rate@5', 0.0):.4f}",
            f"{agg.get('mrr@5', 0.0):.4f}",
            f"{agg.get('ndcg@5', 0.0):.4f}",
            f"{timing.get('retrieval_p50_ms', 0.0):.1f}",
        )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG quality benchmark")
    parser.add_argument("--corpus", default="squad-200", help="Corpus name (default: squad-200)")
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run all chunking strategies",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["fixed", "recursive", "semantic"],
        help="Specific strategies to run",
    )
    parser.add_argument(
        "--all-retrievers",
        action="store_true",
        help="Run all retrievers",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        choices=["dense", "sparse", "hybrid"],
        help="Specific retrievers to run",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for evaluation (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Enable LLM-as-judge scoring (costs API credits)",
    )
    args = parser.parse_args()

    strategies = ["fixed", "recursive", "semantic"] if args.all_strategies else (args.strategies or ["fixed"])
    retrievers = ["dense", "sparse", "hybrid"] if args.all_retrievers else (args.retrievers or ["dense"])

    corpus_dir = Path("data/corpora") / args.corpus
    queries_path = Path("data/queries") / f"{args.corpus}.jsonl"

    if not corpus_dir.exists():
        console.print(f"[red]Corpus directory not found: {corpus_dir}[/red]")
        console.print("[yellow]Run `make ingest` first.[/yellow]")
        sys.exit(1)

    embedder = OpenAIEmbedder()
    conn = psycopg2.connect(config.database_url)

    all_runs = []

    for strategy in strategies:
        console.rule(f"[bold]Strategy: {strategy}[/bold]")

        chunkers = build_chunkers([strategy], embedder)
        chunker = chunkers[0]

        # Pre-chunk corpus for sparse retriever (needs all chunks in memory)
        doc_files = sorted(corpus_dir.glob("*.txt"))
        all_chunks = []
        for doc_file in doc_files:
            text = doc_file.read_text(encoding="utf-8")
            all_chunks.extend(chunker.chunk(text, doc_file.stem))

        # Build corpus_id to match what the pipeline will generate
        chunker_params = {}
        corpus_id = (
            f"{args.corpus}_{chunker.strategy_name}_"
            f"{abs(hash(str(chunker_params))):08x}"
        )

        retriever_configs = build_retriever_configs(
            retrievers, conn, embedder, corpus_id, all_chunks
        )

        pipeline = BenchmarkPipeline(conn=conn, embedder=embedder)
        runs = pipeline.run(
            corpus_dir=corpus_dir,
            queries_path=queries_path,
            corpus_name=args.corpus,
            chunker=chunker,
            retriever_configs=retriever_configs,
            k_values=args.k,
            use_llm_judge=args.llm_judge,
        )
        all_runs.extend(runs)

    conn.close()
    console.rule("[bold]Results[/bold]")
    print_summary(all_runs)


if __name__ == "__main__":
    main()
