"""Deterministic tests for all chunking strategies."""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from rag_quality_lab.chunkers.fixed_size import FixedSizeChunker
from rag_quality_lab.chunkers.recursive import RecursiveChunker, _get_encoder
from rag_quality_lab.chunkers.semantic import SemanticChunker


def _token_count(text: str) -> int:
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------


class TestFixedSizeChunker:
    def test_three_even_chunks_no_overlap(self) -> None:
        text = "a" * 1200
        chunker = FixedSizeChunker(chunk_size=400, overlap=0)
        chunks = chunker.chunk(text, "doc_even")

        assert len(chunks) == 3
        assert chunks[0].metadata.char_start == 0
        assert chunks[1].metadata.char_start == 400
        assert chunks[2].metadata.char_start == 800
        assert all(len(c.text) == 400 for c in chunks)

    def test_two_chunks_with_overlap(self) -> None:
        text = "b" * 500
        chunker = FixedSizeChunker(chunk_size=400, overlap=50)
        chunks = chunker.chunk(text, "doc_overlap")

        assert len(chunks) == 2
        assert chunks[0].metadata.char_start == 0
        assert chunks[1].metadata.char_start == 350

    def test_all_chunks_have_positive_token_count(self) -> None:
        text = "Hello world. " * 100
        chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(text, "doc_tokens")

        assert all(c.metadata.token_count > 0 for c in chunks)

    def test_chunk_id_format(self) -> None:
        text = "x" * 600
        chunker = FixedSizeChunker(chunk_size=300, overlap=0)
        chunks = chunker.chunk(text, "mydoc")

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"mydoc_chunk_{i}"

    def test_overlap_must_be_less_than_chunk_size(self) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------


class TestRecursiveChunker:
    def _make_paragraph_text(self, num_paragraphs: int, tokens_each: int) -> str:
        """Build a text with paragraphs separated by double newlines."""
        # Use a word that is approximately 1 token
        word = "word "
        paragraph = word * tokens_each
        return "\n\n".join(paragraph.strip() for _ in range(num_paragraphs))

    def test_paragraph_split_respects_boundaries(self) -> None:
        # 5 paragraphs × ~50 tokens each; max_tokens=250 fits each paragraph individually
        text = self._make_paragraph_text(5, 50)
        chunker = RecursiveChunker(max_tokens=250)
        chunks = chunker.chunk(text, "doc_para")

        assert len(chunks) == 5
        for chunk in chunks:
            assert _token_count(chunk.text) <= 250

    def test_no_chunk_exceeds_max_tokens(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 200
        chunker = RecursiveChunker(max_tokens=50)
        chunks = chunker.chunk(text, "doc_long")

        for chunk in chunks:
            actual = _token_count(chunk.text)
            assert actual <= 50, f"Chunk exceeds max_tokens: {actual} > 50"

    def test_no_gaps_between_chunks(self) -> None:
        text = "Hello world this is a sentence. " * 30
        chunker = RecursiveChunker(max_tokens=30)
        chunks = chunker.chunk(text, "doc_gaps")

        assert len(chunks) > 1
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1].metadata.char_end
            curr_start = chunks[i].metadata.char_start
            assert curr_start == prev_end, (
                f"Gap between chunk {i-1} (end={prev_end}) and chunk {i} (start={curr_start})"
            )

    def test_chunk_id_format(self) -> None:
        text = "word " * 100
        chunker = RecursiveChunker(max_tokens=20)
        chunks = chunker.chunk(text, "rdoc")

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"rdoc_chunk_{i}"


# ---------------------------------------------------------------------------
# SemanticChunker (unit test — mock embedder)
# ---------------------------------------------------------------------------


def _unit_vector(v: list[float]) -> list[float]:
    arr = np.array(v, dtype=np.float64)
    return (arr / np.linalg.norm(arr)).tolist()


class TestSemanticChunker:
    def _make_mock_embedder(self, embeddings: list[list[float]]) -> MagicMock:
        embedder = MagicMock()
        embedder.embed.return_value = embeddings
        return embedder

    def test_splits_on_dissimilar_sentence(self) -> None:
        # 3 similar sentences + 1 orthogonal sentence
        similar = _unit_vector([1.0, 0.0, 0.0])
        orthogonal = _unit_vector([0.0, 1.0, 0.0])

        sentence_a = "The cat sat on the mat."
        sentence_b = "The cat loves the mat."
        sentence_c = "Cats often sit on mats."
        sentence_d = "Quantum mechanics describes subatomic particles."

        text = f"{sentence_a} {sentence_b} {sentence_c} {sentence_d}"
        embeddings = [similar, similar, similar, orthogonal]
        embedder = self._make_mock_embedder(embeddings)

        chunker = SemanticChunker(
            embedder=embedder,
            similarity_threshold=0.8,
            max_tokens=400,
        )
        chunks = chunker.chunk(text, "sdoc")

        assert len(chunks) == 2

    def test_first_chunk_contains_similar_sentences(self) -> None:
        similar = _unit_vector([1.0, 0.0, 0.0])
        orthogonal = _unit_vector([0.0, 1.0, 0.0])

        sentence_a = "The cat sat on the mat."
        sentence_b = "The cat loves the mat."
        sentence_c = "Cats often sit on mats."
        sentence_d = "Quantum mechanics describes subatomic particles."

        text = f"{sentence_a} {sentence_b} {sentence_c} {sentence_d}"
        embeddings = [similar, similar, similar, orthogonal]
        embedder = self._make_mock_embedder(embeddings)

        chunker = SemanticChunker(
            embedder=embedder,
            similarity_threshold=0.8,
            max_tokens=400,
        )
        chunks = chunker.chunk(text, "sdoc2")

        assert sentence_a in chunks[0].text
        assert sentence_b in chunks[0].text
        assert sentence_c in chunks[0].text

    def test_single_batch_embed_call(self) -> None:
        similar = _unit_vector([1.0, 0.0, 0.0])
        text = "Hello world. This is a test. Another sentence here."
        embeddings = [similar, similar, similar]
        embedder = self._make_mock_embedder(embeddings)

        chunker = SemanticChunker(embedder=embedder, similarity_threshold=0.8)
        chunker.chunk(text, "sdoc3")

        # Must be exactly one embed() call regardless of sentence count
        assert embedder.embed.call_count == 1

    def test_max_tokens_triggers_split(self) -> None:
        similar = _unit_vector([1.0, 0.0, 0.0])
        # All sentences are similar — only token limit should trigger splits
        sentences = ["word " * 20 + "." for _ in range(5)]
        text = " ".join(sentences)
        embeddings = [similar] * 5
        embedder = self._make_mock_embedder(embeddings)

        max_tokens = 30
        chunker = SemanticChunker(
            embedder=embedder,
            similarity_threshold=0.99,  # very high threshold — similarity won't trigger splits
            max_tokens=max_tokens,
        )
        chunks = chunker.chunk(text, "sdoc4")

        # Should have split due to token limit
        assert len(chunks) > 1
