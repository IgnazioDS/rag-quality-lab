from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np
import tiktoken

from rag_quality_lab.chunkers.base import BaseChunker
from rag_quality_lab.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_get_encoder().encode(text))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class SemanticChunker(BaseChunker):
    """Group consecutive sentences by embedding similarity, bounded by token count.

    One batched embedding call is made per document — all sentences are embedded
    together before the grouping pass begins.
    """

    def __init__(
        self,
        embedder: object,
        similarity_threshold: float = 0.8,
        max_tokens: int = 400,
    ) -> None:
        # embedder must implement embed(texts: list[str]) -> list[list[float]]
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold
        self._max_tokens = max_tokens

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
        if not sentences:
            return []

        # Single batched embedding call — not N individual calls
        embeddings: list[list[float]] = self._embedder.embed(sentences)  # type: ignore[attr-defined]

        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_embeddings: list[list[float]] = []
        current_tokens = 0
        chunk_index = 0
        char_cursor = 0

        def _flush(sentence_group: list[str], start_cursor: int) -> tuple[Chunk, int]:
            joined = " ".join(sentence_group)
            end_cursor = start_cursor + len(joined)
            token_count = _token_count(joined)
            metadata = ChunkMetadata(
                source_doc_id=doc_id,
                chunk_index=chunk_index,
                char_start=start_cursor,
                char_end=end_cursor,
                token_count=token_count,
                strategy_name=self.strategy_name,
                strategy_params={
                    "similarity_threshold": self._similarity_threshold,
                    "max_tokens": self._max_tokens,
                },
            )
            return (
                Chunk(chunk_id=f"{doc_id}_chunk_{chunk_index}", text=joined, metadata=metadata),
                end_cursor,
            )

        for sentence, embedding in zip(sentences, embeddings):
            sentence_tokens = _token_count(sentence)

            if current_sentences:
                centroid = list(np.mean(np.array(current_embeddings), axis=0))
                sim = _cosine_similarity(embedding, centroid)
                would_exceed = (current_tokens + sentence_tokens) > self._max_tokens

                if sim < self._similarity_threshold or would_exceed:
                    chunk, char_cursor = _flush(current_sentences, char_cursor)
                    chunks.append(chunk)
                    chunk_index += 1
                    # Advance cursor past any whitespace between chunks
                    if char_cursor < len(text) and text[char_cursor] == " ":
                        char_cursor += 1
                    current_sentences = []
                    current_embeddings = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_embeddings.append(embedding)
            current_tokens += sentence_tokens

        if current_sentences:
            chunk, _ = _flush(current_sentences, char_cursor)
            chunks.append(chunk)

        logger.debug("SemanticChunker produced %d chunks for doc_id=%s", len(chunks), doc_id)
        return chunks
