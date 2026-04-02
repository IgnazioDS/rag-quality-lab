# Metrics Reference

All metrics are evaluated at K values in {1, 3, 5, 10}. All return float in [0.0, 1.0].

---

## Hit Rate @ K

**What it measures**: Whether any relevant chunk appears in the top-K retrieved results.

**Formula**:
```
hit_rate@K = 1 if ∃ result ∈ top_K : result.chunk_id ∈ relevant_ids
             0 otherwise
```

For a query set, this is averaged: `mean(hit_rate@K across all queries)`.

**Range**: [0.0, 1.0]

**Good score**: ≥ 0.8. If below 0.6, retrieval is failing to surface relevant content.

**Bad score**: < 0.5. The system is missing more than half of queries entirely.

**When it matters**: Use as the first-pass check. If hit rate is low, nothing else
matters — the system is fundamentally broken for a large fraction of queries.

**Worked example** (5 results, relevant = {"doc3"}):

| Rank | chunk_id |
|------|----------|
| 1    | doc1     |
| 2    | doc2     |
| 3    | doc3 ✓   |
| 4    | doc4     |
| 5    | doc5     |

- `hit_rate@2` = 0.0 (doc3 not in top-2)
- `hit_rate@3` = 1.0 (doc3 is at rank 3)
- `hit_rate@5` = 1.0

---

## MRR @ K (Mean Reciprocal Rank)

**What it measures**: The position of the *first* relevant result. Rewards systems that
rank the relevant item as high as possible.

**Formula**:
```
mrr@K = 1 / rank(first relevant result in top-K)
      = 0  if no relevant result in top-K
```

Averaged across queries: `mean(mrr@K across all queries)`.

**Range**: [0.0, 1.0]

**Good score**: ≥ 0.7. The first relevant result appears in the top 1–2 positions
for most queries.

**Bad score**: < 0.4. Relevant results are consistently buried.

**When it matters**: When only the first result matters — e.g., a system that surfaces
one answer to a question. Also useful when downstream reranking is expensive and you
want the retriever to front-load quality.

**Worked example** (same setup as above):

- `mrr@3` = 1/3 ≈ 0.333 (first relevant result at rank 3)
- `mrr@5` = 1/3 ≈ 0.333 (same — rank 3 is within top-5)
- `mrr@2` = 0.0 (first relevant result is at rank 3, outside top-2)

---

## NDCG @ K (Normalized Discounted Cumulative Gain)

**What it measures**: Ranking quality — whether relevant results appear near the top,
with diminishing credit for lower-ranked results.

**Formula** (binary relevance):
```
DCG@K   = Σ_{i=1}^{K} rel_i / log₂(i + 1)
IDCG@K  = Σ_{i=1}^{min(|relevant|, K)} 1 / log₂(i + 1)
NDCG@K  = DCG@K / IDCG@K
```

Where `rel_i = 1 if result at rank i is relevant, else 0`.

**Range**: [0.0, 1.0]

**Good score**: ≥ 0.7.

**Bad score**: < 0.4. Relevant results are either missing or ranked poorly.

**When it matters**: When you have multiple relevant items and care about their relative
ordering. NDCG penalizes a system that finds all relevant items but puts them at the
bottom of the list. Use when ranking quality matters across the full top-K window.

**Worked example** (same setup, relevant = {"doc3"}, only 1 relevant item):

- DCG@5 = 1/log₂(3+1) = 1/log₂(4) = 1/2 = 0.5
- IDCG@5 = 1/log₂(1+1) = 1/log₂(2) = 1.0
- NDCG@5 = 0.5 / 1.0 = 0.5

---

## Precision @ K

**What it measures**: The fraction of retrieved results that are relevant.

**Formula**:
```
precision@K = |{relevant results in top-K}| / K
```

**Range**: [0.0, 1.0]

**Good score**: Depends heavily on the number of relevant items per query. For queries
with 1 relevant item, precision@5 = 0.2 is the maximum achievable score with a correct
top-1 result.

**When it matters**: When result quality matters more than recall. If a user sees 5
results and 3 are relevant, that's a better experience than 5 results with 1 relevant.

**Worked example** (same setup):

- `precision@3` = 1/3 ≈ 0.333 (1 relevant in top-3)
- `precision@5` = 1/5 = 0.2 (1 relevant in top-5)
- `precision@1` = 0.0 (rank 1 is not relevant)

---

## Recall @ K

**What it measures**: The fraction of *all* relevant items that appear in top-K.

**Formula**:
```
recall@K = |{relevant results in top-K}| / |relevant_ids|
```

**Range**: [0.0, 1.0]

**Good score**: ≥ 0.8 at K=10 is strong. Lower K recall is constrained by how many
relevant items exist.

**When it matters**: When missing a relevant chunk is the primary failure mode. Use in
scenarios where completeness matters — e.g., a research assistant that should surface
all relevant evidence for a claim.

**Worked example** (same setup, 1 relevant item total):

- `recall@5` = 1/1 = 1.0 (the single relevant item is in top-5)
- `recall@2` = 0.0 (the single relevant item is at rank 3)

---

## compute_all_metrics

Returns all five metrics at all requested K values in one call:

```python
metrics = compute_all_metrics(results, relevant_ids, k_values=[1, 3, 5, 10])
# → {"hit_rate@1": 0.0, "hit_rate@3": 1.0, "hit_rate@5": 1.0, ..., "recall@10": 1.0}
```

This is the function called by `BenchmarkPipeline` for each query. The outer aggregation
(mean across queries) produces the `aggregated` field in the result JSON.
