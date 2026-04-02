import logging
from typing import Any

import psycopg2

from rag_quality_lab.embedders.base import BaseEmbedder
from rag_quality_lab.models import RetrievalResult
from rag_quality_lab.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """pgvector cosine similarity retrieval using raw psycopg2."""

    def __init__(
        self,
        conn: Any,  # psycopg2.connection — typed as Any to avoid runtime import issues
        embedder: BaseEmbedder,
        corpus_id: str,
        table: str = "chunks",
    ) -> None:
        self._conn = conn
        self._embedder = embedder
        self._corpus_id = corpus_id
        self._table = table

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = self._embedder.embed_one(query)
        vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"

        sql = f"""
            SELECT chunk_id, chunk_text, 1 - (embedding <=> %s::vector) AS score
            FROM {self._table}
            WHERE corpus_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """  # noqa: S608 — table name is controlled internally, not user input

        with self._conn.cursor() as cur:
            cur.execute(sql, (vector_str, self._corpus_id, vector_str, top_k))
            rows = cur.fetchall()

        results = [
            RetrievalResult(
                chunk_id=row[0],
                chunk_text=row[1],
                score=float(row[2]),
                rank=rank,
                retriever_name="dense",
            )
            for rank, row in enumerate(rows, start=1)
        ]
        logger.debug("DenseRetriever returned %d results for query=%r", len(results), query[:50])
        return results
