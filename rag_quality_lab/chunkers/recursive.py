import logging
from functools import lru_cache
from typing import Final

import tiktoken

from rag_quality_lab.chunkers.base import BaseChunker
from rag_quality_lab.models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

_DEFAULT_SEPARATORS: Final[list[str]] = ["\n\n", "\n", ". ", " ", ""]


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_get_encoder().encode(text))


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _recursive_split(text: str, separators: list[str], max_tokens: int) -> list[str]:
    """Recursively split text using the separator hierarchy until pieces fit within max_tokens."""
    if _token_count(text) <= max_tokens:
        return [text]

    if not separators:
        # Character-by-character fallback — group chars into max_tokens sized pieces
        pieces: list[str] = []
        current = ""
        for char in text:
            if _token_count(current + char) > max_tokens:
                if current:
                    pieces.append(current)
                current = char
            else:
                current += char
        if current:
            pieces.append(current)
        return pieces

    separator = separators[0]
    remaining = separators[1:]
    raw_pieces = _split_by_separator(text, separator)

    result: list[str] = []
    for piece in raw_pieces:
        if not piece:
            continue
        if _token_count(piece) <= max_tokens:
            result.append(piece)
        else:
            result.extend(_recursive_split(piece, remaining, max_tokens))

    return result


class RecursiveChunker(BaseChunker):
    """Split by separator hierarchy, bounded by token count."""

    def __init__(
        self,
        max_tokens: int = 400,
        separators: list[str] | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._separators = separators if separators is not None else _DEFAULT_SEPARATORS

    @property
    def strategy_name(self) -> str:
        return "recursive"

    def chunk(self, text: str, doc_id: str) -> list[Chunk]:
        raw_pieces = _recursive_split(text, self._separators, self._max_tokens)

        chunks: list[Chunk] = []
        cursor = 0

        for index, piece in enumerate(raw_pieces):
            # Find the actual position of this piece in the original text starting from cursor.
            # This ensures char_start of chunk N equals char_end of chunk N-1.
            char_start = cursor
            char_end = cursor + len(piece)
            token_count = _token_count(piece)

            metadata = ChunkMetadata(
                source_doc_id=doc_id,
                chunk_index=index,
                char_start=char_start,
                char_end=char_end,
                token_count=token_count,
                strategy_name=self.strategy_name,
                strategy_params={"max_tokens": self._max_tokens, "separators": self._separators},
            )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_chunk_{index}",
                    text=piece,
                    metadata=metadata,
                )
            )
            cursor = char_end

        logger.debug("RecursiveChunker produced %d chunks for doc_id=%s", len(chunks), doc_id)
        return chunks
