"""Exact-value tests for all retrieval metrics.

All expected values are hand-calculated and verified against formulas.
Use pytest.approx(abs=1e-4) for float comparison.
"""
import math

import pytest

from rag_quality_lab.evaluators.metrics import (
    compute_all_metrics,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag_quality_lab.models import RetrievalResult


def _make_results(chunk_ids: list[str]) -> list[RetrievalResult]:
    """Build a result list with rank = position + 1 and descending scores."""
    return [
        RetrievalResult(
            chunk_id=cid,
            chunk_text="",
            score=1.0 - i * 0.1,
            rank=i + 1,
            retriever_name="test",
        )
        for i, cid in enumerate(chunk_ids)
    ]


# Setup A: relevant item is at rank 1
results_rank1 = _make_results(["doc1", "doc2", "doc3", "doc4", "doc5"])
relevant_a = {"doc1"}

# Setup B: relevant item is at rank 3
results_rank3 = _make_results(["doc2", "doc4", "doc3", "doc5", "doc6"])
relevant_b = {"doc3"}


# ---------------------------------------------------------------------------
# hit_rate_at_k
# ---------------------------------------------------------------------------


class TestHitRate:
    def test_relevant_at_rank1_k1(self) -> None:
        assert hit_rate_at_k(results_rank1, relevant_a, k=1) == pytest.approx(1.0)

    def test_relevant_at_rank1_k5(self) -> None:
        assert hit_rate_at_k(results_rank1, relevant_a, k=5) == pytest.approx(1.0)

    def test_relevant_at_rank3_k2_miss(self) -> None:
        assert hit_rate_at_k(results_rank3, relevant_b, k=2) == pytest.approx(0.0)

    def test_relevant_at_rank3_k3_hit(self) -> None:
        assert hit_rate_at_k(results_rank3, relevant_b, k=3) == pytest.approx(1.0)

    def test_nonexistent_relevant(self) -> None:
        assert hit_rate_at_k(results_rank1, {"nonexistent"}, k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------


class TestMRR:
    def test_relevant_at_rank1(self) -> None:
        # MRR = 1/1 = 1.0
        assert mrr_at_k(results_rank1, relevant_a, k=5) == pytest.approx(1.0)

    def test_relevant_at_rank3(self) -> None:
        # MRR = 1/3
        assert mrr_at_k(results_rank3, relevant_b, k=5) == pytest.approx(1 / 3, abs=1e-4)

    def test_nonexistent_returns_zero(self) -> None:
        assert mrr_at_k(results_rank1, {"nonexistent"}, k=5) == pytest.approx(0.0)

    def test_k_cutoff_excludes_relevant(self) -> None:
        # relevant is at rank 3 but k=2 — should return 0
        assert mrr_at_k(results_rank3, relevant_b, k=2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_relevant_at_rank1_k5(self) -> None:
        # 1 relevant in top-5 → 1/5 = 0.2
        assert precision_at_k(results_rank1, relevant_a, k=5) == pytest.approx(0.2, abs=1e-4)

    def test_relevant_at_rank3_k3(self) -> None:
        # 1 relevant in top-3 → 1/3
        assert precision_at_k(results_rank3, relevant_b, k=3) == pytest.approx(1 / 3, abs=1e-4)

    def test_relevant_at_rank1_k1(self) -> None:
        assert precision_at_k(results_rank1, relevant_a, k=1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecall:
    def test_relevant_at_rank1_k5(self) -> None:
        # 1 relevant total, found in top-5 → 1.0
        assert recall_at_k(results_rank1, relevant_a, k=5) == pytest.approx(1.0)

    def test_not_found_returns_zero(self) -> None:
        assert recall_at_k(results_rank1, {"nonexistent"}, k=5) == pytest.approx(0.0)

    def test_empty_relevant_set(self) -> None:
        assert recall_at_k(results_rank1, set(), k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_relevant_at_rank1_is_perfect(self) -> None:
        # Only one relevant item, it's at rank 1 → DCG = 1/log2(2) = 1.0, IDCG = 1.0
        assert ndcg_at_k(results_rank1, relevant_a, k=5) == pytest.approx(1.0)

    def test_relevant_at_rank3(self) -> None:
        # DCG = 1/log2(4) = 1/2, IDCG = 1/log2(2) = 1.0 → NDCG = 0.5
        expected = 1.0 / math.log2(4)  # rank=3 → log2(3+1)
        assert ndcg_at_k(results_rank3, relevant_b, k=5) == pytest.approx(expected, abs=1e-4)

    def test_nonexistent_returns_zero(self) -> None:
        assert ndcg_at_k(results_rank1, {"nonexistent"}, k=5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_returns_all_expected_keys(self) -> None:
        metrics = compute_all_metrics(results_rank1, relevant_a, k_values=[1, 5])
        expected_keys = {
            "hit_rate@1", "hit_rate@5",
            "mrr@1", "mrr@5",
            "ndcg@1", "ndcg@5",
            "precision@1", "precision@5",
            "recall@1", "recall@5",
        }
        assert set(metrics.keys()) == expected_keys

    def test_values_are_consistent_with_individual_functions(self) -> None:
        metrics = compute_all_metrics(results_rank3, relevant_b, k_values=[3, 5])
        assert metrics["hit_rate@3"] == pytest.approx(hit_rate_at_k(results_rank3, relevant_b, 3))
        assert metrics["mrr@5"] == pytest.approx(mrr_at_k(results_rank3, relevant_b, 5))
        assert metrics["ndcg@5"] == pytest.approx(ndcg_at_k(results_rank3, relevant_b, 5))

    def test_default_k_values(self) -> None:
        metrics = compute_all_metrics(results_rank1, relevant_a)
        for k in [1, 3, 5, 10]:
            assert f"hit_rate@{k}" in metrics
