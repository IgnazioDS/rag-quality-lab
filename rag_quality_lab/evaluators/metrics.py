import math

from rag_quality_lab.models import RetrievalResult


def hit_rate_at_k(
    results: list[RetrievalResult], relevant_ids: set[str], k: int
) -> float:
    """Fraction of queries for which at least one relevant result appears in the top-K.

    For a single query this is binary: 1.0 if any of the top-K results is relevant, else 0.0.
    """
    top_k = results[:k]
    return 1.0 if any(r.chunk_id in relevant_ids for r in top_k) else 0.0


def mrr_at_k(
    results: list[RetrievalResult], relevant_ids: set[str], k: int
) -> float:
    """Reciprocal rank of the first relevant result in the top-K."""
    for result in results[:k]:
        if result.chunk_id in relevant_ids:
            return 1.0 / result.rank
    return 0.0


def ndcg_at_k(
    results: list[RetrievalResult], relevant_ids: set[str], k: int
) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    top_k = results[:k]

    dcg = sum(
        1.0 / math.log2(result.rank + 1)
        for result in top_k
        if result.chunk_id in relevant_ids
    )

    # Ideal DCG: all relevant results at the top ranks
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(
    results: list[RetrievalResult], relevant_ids: set[str], k: int
) -> float:
    """Fraction of the top-K results that are relevant."""
    if k == 0:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.chunk_id in relevant_ids)
    return hits / k


def recall_at_k(
    results: list[RetrievalResult], relevant_ids: set[str], k: int
) -> float:
    """Fraction of all relevant chunks that appear in the top-K results."""
    if not relevant_ids:
        return 0.0
    top_k = results[:k]
    hits = sum(1 for r in top_k if r.chunk_id in relevant_ids)
    return hits / len(relevant_ids)


def compute_all_metrics(
    results: list[RetrievalResult],
    relevant_ids: set[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all metrics at each K value.

    Returns keys like 'hit_rate@5', 'mrr@3', 'ndcg@10', etc.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(results, relevant_ids, k)
        metrics[f"mrr@{k}"] = mrr_at_k(results, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(results, relevant_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(results, relevant_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(results, relevant_ids, k)

    return metrics
