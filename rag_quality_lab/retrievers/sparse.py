import logging

from rank_bm25 import BM25Okapi

from rag_quality_lab.models import Chunk, RetrievalResult
from rag_quality_lab.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


class SparseRetriever(BaseRetriever):
    """BM25 retrieval using rank_bm25. Index is built at construction time."""

    def __init__(self, corpus_chunks: list[Chunk]) -> None:
        self._chunks = corpus_chunks
        tokenized = [chunk.text.split() for chunk in corpus_chunks]
        self._index = BM25Okapi(tokenized)
        logger.debug("SparseRetriever built BM25 index over %d chunks", len(corpus_chunks))

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_tokens = query.split()
        scores: list[float] = self._index.get_scores(query_tokens).tolist()

        max_score = max(scores) if scores else 0.0

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (chunk_idx, raw_score) in enumerate(indexed, start=1):
            normalized_score = raw_score / max_score if max_score > 0.0 else 0.0
            chunk = self._chunks[chunk_idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.text,
                    score=normalized_score,
                    rank=rank,
                    retriever_name="sparse",
                )
            )

        logger.debug("SparseRetriever returned %d results for query=%r", len(results), query[:50])
        return results
