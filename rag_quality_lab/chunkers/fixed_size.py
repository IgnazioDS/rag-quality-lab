import logging
from functools import lru_cache

import tiktoken

from rag_quality_lab.chunkers.base import BaseChunker
from rag_quality_lab.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


class FixedSizeChunker(BaseChunker):
    """Split text by character count with optional overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def strategy_name(self) -> str:
        return "fixed_size"

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self._chunk_size - self._overlap
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            slice_text = text[start:end]
            token_count = len(_get_encoder().encode(slice_text))

            metadata = ChunkMetadata(
                source_doc_id=doc_id,
                chunk_index=index,
                char_start=start,
                char_end=end,
                token_count=token_count,
                strategy_name=self.strategy_name,
                strategy_params={"chunk_size": self._chunk_size, "overlap": self._overlap},
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{index}",
                    text=slice_text,
                    metadata=metadata,
                )
            )
            index += 1
            start += step

        logger.debug("FixedSizeChunker produced %d chunks for doc_id=%s", len(chunks), doc_id)
        return chunks
