import logging

from rag_quality_lab.models import RetrievalResult
from rag_quality_lab.retrievers.base import BaseRetriever
from rag_quality_lab.retrievers.dense import DenseRetriever
from rag_quality_lab.retrievers.sparse import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Reciprocal Rank Fusion over dense and sparse retrievers."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        candidate_count = top_k * 2
        missing_rank = candidate_count + 1

        dense_results = self._dense.retrieve(query, candidate_count)
        sparse_results = self._sparse.retrieve(query, candidate_count)

        # Map chunk_id → rank for each signal
        dense_ranks: dict[str, int] = {r.chunk_id: r.rank for r in dense_results}
        sparse_ranks: dict[str, int] = {r.chunk_id: r.rank for r in sparse_results}
        chunk_texts: dict[str, str] = {
            r.chunk_id: r.chunk_text for r in (*dense_results, *sparse_results)
        }

        all_chunk_ids = set(dense_ranks) | set(sparse_ranks)
        rrf_scores: dict[str, float] = {}

        for chunk_id in all_chunk_ids:
            d_rank = dense_ranks.get(chunk_id, missing_rank)
            s_rank = sparse_ranks.get(chunk_id, missing_rank)
            rrf_scores[chunk_id] = (
                self._dense_weight / (self._rrf_k + d_rank)
                + self._sparse_weight / (self._rrf_k + s_rank)
            )

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = [
            RetrievalResult(
                chunk_id=chunk_id,
                chunk_text=chunk_texts[chunk_id],
                score=score,
                rank=rank,
                retriever_name="hybrid",
            )
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]

        logger.debug("HybridRetriever returned %d results for query=%r", len(results), query[:50])
        return results
