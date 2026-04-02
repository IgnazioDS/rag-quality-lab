from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras

from rag_quality_lab.chunkers.base import BaseChunker
from rag_quality_lab.config import config
from rag_quality_lab.embedders.base import BaseEmbedder
from rag_quality_lab.evaluators.llm_judge import LLMJudge
from rag_quality_lab.evaluators.metrics import compute_all_metrics
from rag_quality_lab.models import BenchmarkRun, Chunk, QueryResult, RetrievalResult
from rag_quality_lab.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


def _load_queries(queries_path: Path) -> list[dict[str, Any]]:
    queries = []
    with queries_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _upsert_chunks(
    conn: Any,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    corpus_id: str,
) -> None:
    sql = """
        INSERT INTO chunks (
            chunk_id, corpus_id, source_doc_id, chunk_index, chunk_text,
            char_start, char_end, token_count, strategy_name, strategy_params, embedding
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            chunk_text = EXCLUDED.chunk_text
    """
    rows = [
        (
            chunk.chunk_id,
            corpus_id,
            chunk.metadata.source_doc_id,
            chunk.metadata.chunk_index,
            chunk.text,
            chunk.metadata.char_start,
            chunk.metadata.char_end,
            chunk.metadata.token_count,
            chunk.metadata.strategy_name,
            json.dumps(chunk.metadata.strategy_params),
            "[" + ",".join(str(v) for v in emb) + "]",
        )
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    conn.commit()


def _upsert_corpus(
    conn: Any,
    corpus_id: str,
    corpus_name: str,
    doc_count: int,
    chunk_count: int,
    strategy_name: str,
    strategy_params: dict[str, Any],
) -> None:
    sql = """
        INSERT INTO corpora
            (corpus_id, name, doc_count, chunk_count, strategy_name, strategy_params)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (corpus_id) DO UPDATE SET
            chunk_count = EXCLUDED.chunk_count
    """
    with conn.cursor() as cur:
        cur.execute(
            sql,
            (
                corpus_id,
                corpus_name,
                doc_count,
                chunk_count,
                strategy_name,
                json.dumps(strategy_params),
            ),
        )
    conn.commit()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, p))


class BenchmarkPipeline:
    """Orchestrates chunking, embedding, ingestion, retrieval, and evaluation."""

    def __init__(
        self,
        conn: Any,
        embedder: BaseEmbedder,
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._judge: LLMJudge | None = None

    def run(
        self,
        corpus_dir: Path,
        queries_path: Path,
        corpus_name: str,
        chunker: BaseChunker,
        retriever_configs: list[dict[str, Any]],
        k_values: list[int] | None = None,
        use_llm_judge: bool = False,
    ) -> list[BenchmarkRun]:
        if k_values is None:
            k_values = [1, 3, 5, 10]

        if use_llm_judge and self._judge is None:
            self._judge = LLMJudge()

        queries = _load_queries(queries_path)
        doc_files = sorted(corpus_dir.glob("*.txt"))

        chunker_params = getattr(chunker, "_" + "strategy_params", {})
        # Build corpus_id from chunker identity
        corpus_id = (
            f"{corpus_name}_{chunker.strategy_name}_"
            f"{abs(hash(str(chunker_params))):08x}"
        )

        # --- Ingestion phase ---
        ingest_start = time.perf_counter()
        all_chunks: list[Chunk] = []
        doc_count = 0

        for doc_file in doc_files:
            doc_text = doc_file.read_text(encoding="utf-8")
            doc_id = doc_file.stem
            chunks = chunker.chunk(doc_text, doc_id)
            all_chunks.extend(chunks)
            doc_count += 1
            logger.info("Chunked %s → %d chunks", doc_file.name, len(chunks))

        logger.info(
            "Ingesting %d total chunks for corpus_id=%s", len(all_chunks), corpus_id
        )
        chunk_texts = [c.text for c in all_chunks]
        embeddings = self._embedder.embed(chunk_texts)
        _upsert_chunks(self._conn, all_chunks, embeddings, corpus_id)
        _upsert_corpus(
            self._conn,
            corpus_id,
            corpus_name,
            doc_count,
            len(all_chunks),
            chunker.strategy_name,
            chunker_params,
        )

        ingest_end = time.perf_counter()
        ingestion_s = ingest_end - ingest_start

        benchmark_runs: list[BenchmarkRun] = []

        for rc in retriever_configs:
            retriever_name: str = rc["name"]
            retriever: BaseRetriever = rc["retriever"]
            retriever_params: dict[str, Any] = rc.get("params", {})

            query_results: list[QueryResult] = []
            per_query: list[dict[str, Any]] = []
            retrieval_latencies_ms: list[float] = []

            max_k = max(k_values)

            for query_record in queries:
                query_id = query_record["query_id"]
                query_text = query_record["query_text"]
                relevant_ids: set[str] = set(query_record.get("relevant_chunk_ids", []))
                reference_answer = query_record.get("reference_answer", "")

                t0 = time.perf_counter()
                retrieved: list[RetrievalResult] = retriever.retrieve(query_text, max_k)
                t1 = time.perf_counter()
                retrieval_latencies_ms.append((t1 - t0) * 1000)

                query_metrics = compute_all_metrics(retrieved, relevant_ids, k_values)

                judge_scores: dict[str, float] = {}
                if use_llm_judge and self._judge is not None and retrieved:
                    top_chunk = retrieved[0]
                    js = self._judge.score(query_text, top_chunk.chunk_text, reference_answer)
                    judge_scores = {
                        "judge_faithfulness": js.faithfulness,
                        "judge_relevance": js.relevance,
                        "judge_answer_correctness": js.answer_correctness,
                    }

                query_results.append(
                    QueryResult(
                        query_id=query_id,
                        query_text=query_text,
                        retrieved=retrieved,
                        relevant_ids=relevant_ids,
                        reference_answer=reference_answer,
                    )
                )

                per_query.append(
                    {
                        "query_id": query_id,
                        "metrics": {**query_metrics, **judge_scores},
                    }
                )

            # Aggregate metrics across queries
            all_metric_keys = per_query[0]["metrics"].keys() if per_query else []
            aggregated: dict[str, float] = {}
            for key in all_metric_keys:
                values = [pq["metrics"][key] for pq in per_query]
                aggregated[key] = float(np.mean(values))

            timing = {
                "ingestion_s": ingestion_s,
                "retrieval_p50_ms": _percentile(retrieval_latencies_ms, 50),
                "retrieval_p95_ms": _percentile(retrieval_latencies_ms, 95),
                "retrieval_p99_ms": _percentile(retrieval_latencies_ms, 99),
            }

            run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
            run = BenchmarkRun(
                run_id=run_id,
                corpus_name=corpus_name,
                chunker_name=chunker.strategy_name,
                chunker_params=chunker_params,
                retriever_name=retriever_name,
                retriever_params=retriever_params,
                embedding_model=(
                self._embedder._model  # type: ignore[attr-defined]
                if hasattr(self._embedder, "_model")
                else "unknown"
            ),
                query_results=query_results,
                aggregated_metrics=aggregated,
                timing=timing,
                per_query=per_query,
            )

            self._write_result(run)
            benchmark_runs.append(run)
            logger.info(
                "Run %s complete: chunker=%s retriever=%s hit_rate@5=%.3f",
                run_id,
                chunker.strategy_name,
                retriever_name,
                aggregated.get("hit_rate@5", 0.0),
            )

        return benchmark_runs

    def _write_result(self, run: BenchmarkRun) -> None:
        results_dir = config.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        artifact = {
            "run_id": run.run_id,
            "config": {
                "corpus_name": run.corpus_name,
                "chunker_name": run.chunker_name,
                "chunker_params": run.chunker_params,
                "retriever_name": run.retriever_name,
                "retriever_params": run.retriever_params,
                "embedding_model": run.embedding_model,
            },
            "per_query": run.per_query,
            "aggregated": run.aggregated_metrics,
            "timing": run.timing,
        }

        path = results_dir / f"{run.run_id}_{run.chunker_name}_{run.retriever_name}.json"
        path.write_text(json.dumps(artifact, indent=2))
        logger.info("Result written to %s", path)
