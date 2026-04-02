"""Integration tests for the full BenchmarkPipeline.

Requires a running pgvector instance:
    docker compose up -d
    python scripts/init_db.py

Mark: @pytest.mark.integration
Skip in CI without Docker by setting SKIP_INTEGRATION=1 in the environment.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_quality_lab.chunkers.fixed_size import FixedSizeChunker
from rag_quality_lab.models import RetrievalResult

# Skip integration tests when Docker is not available
_SKIP_INTEGRATION = os.environ.get("SKIP_INTEGRATION", "0") == "1"

pytestmark = pytest.mark.integration


@pytest.fixture
def temp_corpus(tmp_path: Path) -> Path:
    """Create 5 temporary document files."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(5):
        doc = corpus_dir / f"doc_{i:04d}.txt"
        # ~100 words each
        doc.write_text(
            " ".join([f"word{j}" for j in range(100)]) + ".",
            encoding="utf-8",
        )
    return corpus_dir


@pytest.fixture
def temp_queries(tmp_path: Path) -> Path:
    """Create 3 queries with known relevant chunk IDs."""
    queries_path = tmp_path / "queries.jsonl"
    queries = [
        {
            "query_id": "q0000",
            "query_text": "what is word0 word1",
            "relevant_chunk_ids": ["doc_0000_chunk_0"],
            "reference_answer": "word0 word1",
        },
        {
            "query_id": "q0001",
            "query_text": "word50 word51 word52",
            "relevant_chunk_ids": ["doc_0001_chunk_0"],
            "reference_answer": "word50 word51",
        },
        {
            "query_id": "q0002",
            "query_text": "word90 word91",
            "relevant_chunk_ids": ["doc_0002_chunk_0"],
            "reference_answer": "word90 word91",
        },
    ]
    queries_path.write_text(
        "\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8"
    )
    return queries_path


@pytest.mark.skipif(_SKIP_INTEGRATION, reason="Requires Docker compose up")
def test_pipeline_writes_result_json(
    temp_corpus: Path, temp_queries: Path, tmp_path: Path
) -> None:
    """Run pipeline end-to-end; assert result JSON is written with correct schema."""
    import psycopg2

    from rag_quality_lab.config import config
    from rag_quality_lab.embedders.base import BaseEmbedder
    from rag_quality_lab.pipeline import BenchmarkPipeline
    from rag_quality_lab.retrievers.dense import DenseRetriever

    # Mock embedder to avoid real API calls in integration test
    mock_embedder = MagicMock(spec=BaseEmbedder)
    # Return deterministic 1536-dim vectors
    mock_embedder.embed.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    mock_embedder.embed_one.side_effect = lambda text: [0.1] * 1536
    mock_embedder._model = "mock-model"

    conn = psycopg2.connect(config.database_url)
    chunker = FixedSizeChunker(chunk_size=512, overlap=0)

    corpus_id = f"test_fixed_size_{abs(hash(str({}))):08x}"
    dense = DenseRetriever(conn=conn, embedder=mock_embedder, corpus_id=corpus_id)

    retriever_configs = [{"name": "dense", "retriever": dense, "params": {}}]

    with patch.object(config, "results_dir", tmp_path / "results"):
        pipeline = BenchmarkPipeline(conn=conn, embedder=mock_embedder)
        runs = pipeline.run(
            corpus_dir=temp_corpus,
            queries_path=temp_queries,
            corpus_name="test",
            chunker=chunker,
            retriever_configs=retriever_configs,
            k_values=[1, 3, 5],
        )

    conn.close()

    assert len(runs) == 1
    run = runs[0]

    # Verify result file was written
    results_dir = tmp_path / "results"
    result_files = list(results_dir.glob("*.json"))
    assert len(result_files) == 1

    artifact = json.loads(result_files[0].read_text())

    # Schema validation
    assert "run_id" in artifact
    assert "config" in artifact
    assert "per_query" in artifact
    assert "aggregated" in artifact
    assert "timing" in artifact

    cfg = artifact["config"]
    assert cfg["corpus_name"] == "test"
    assert cfg["chunker_name"] == "fixed_size"
    assert cfg["retriever_name"] == "dense"

    # Metrics are floats in [0, 1]
    agg = artifact["aggregated"]
    assert isinstance(agg.get("hit_rate@5"), float)
    assert 0.0 <= agg["hit_rate@5"] <= 1.0

    # Timing fields are positive
    timing = artifact["timing"]
    assert timing["retrieval_p50_ms"] >= 0.0
    assert timing["ingestion_s"] > 0.0

    # Per-query entries exist
    assert len(artifact["per_query"]) == 3
    for pq in artifact["per_query"]:
        assert "query_id" in pq
        assert "metrics" in pq
