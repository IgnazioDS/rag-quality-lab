from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkMetadata:
    source_doc_id: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    strategy_name: str
    strategy_params: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str  # f"{source_doc_id}_chunk_{chunk_index}"
    text: str
    metadata: ChunkMetadata


@dataclass
class RetrievalResult:
    chunk_id: str
    chunk_text: str
    score: float
    rank: int
    retriever_name: str


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    retrieved: list[RetrievalResult]
    relevant_ids: set[str]
    reference_answer: str


@dataclass
class BenchmarkRun:
    run_id: str  # ISO timestamp: 20260402T143022
    corpus_name: str
    chunker_name: str
    chunker_params: dict[str, Any]
    retriever_name: str
    retriever_params: dict[str, Any]
    embedding_model: str
    query_results: list[QueryResult]
    aggregated_metrics: dict[str, float]
    timing: dict[str, float]
    per_query: list[dict[str, Any]] = field(default_factory=list)
