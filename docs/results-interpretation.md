# Results Interpretation

How to read a benchmark result, what to look at first, and how to determine when a
difference between strategies is meaningful vs. noise.

---

## Where to Start

Open the result JSON or the generated Markdown report. Look at these in order:

### 1. `hit_rate@5` in `aggregated`

This is the floor metric. If it's below 0.6, the system is broken — more than 40% of
queries return no relevant content in the top-5 results. Nothing else matters until
this is fixed.

A hit rate above 0.8 means the retriever is broadly surfacing relevant content. Look
deeper at MRR and NDCG to understand ranking quality.

### 2. `mrr@5`

Once hit rate is acceptable, MRR tells you whether the *first* relevant result is
near the top. A gap between `hit_rate@5` and `mrr@5` — e.g., hit rate 0.85 but
MRR 0.45 — means the system finds relevant content but buries it. The relevant chunk
is in the top-5 most of the time, but rarely at rank 1 or 2.

This pattern is common with sparse (BM25) retrieval on queries where the relevant
chunk uses different terminology than the query. The chunk is found but not ranked well.

### 3. `ndcg@5` vs `ndcg@10`

If `ndcg@10` is substantially higher than `ndcg@5`, relevant results exist in ranks
6–10 that the K=5 cutoff misses. This is a signal to increase K or improve ranking.

If `ndcg@5 ≈ ndcg@10`, the relevant results are either in the top-5 or not in the
top-10 at all — increasing K beyond 5 won't help.

---

## What Counts as a Meaningful Difference

With 100 queries (the SQuAD-200 default), differences in aggregated metrics smaller
than **0.03–0.04** are likely noise. A difference of 0.05 or more is worth investigating.

The rough rule: if the delta at `hit_rate@5` between two strategies is less than
5 percentage points (e.g., 0.72 vs 0.75), do not draw conclusions — run more queries
or use a bootstrap confidence interval before reporting.

A difference of 0.10 or more (e.g., 0.72 vs 0.82 on `hit_rate@5`) is robust enough
to be actionable without additional statistical testing.

---

## MRR vs NDCG: When to Prefer Which

**Prefer MRR when:**
- Only the top-1 or top-2 result will be shown to the user
- The downstream system uses the first retrieved chunk as a direct answer
- You have 1 relevant item per query and want to measure how often it's ranked first

**Prefer NDCG when:**
- Multiple relevant items exist per query (e.g., questions with multiple valid contexts)
- The downstream system uses all top-K results (e.g., a reranker or a multi-document
  summarizer)
- You want to reward systems that find more relevant items across a broader window

In the SQuAD-200 setup, each query has exactly one relevant document marked as ground
truth. NDCG and MRR will agree closely here. The more useful comparison is between
`ndcg@1` (did the best result get rank 1?) and `ndcg@5` (are relevant results
distributed across the top-5?).

---

## When Sparse Retrieval Beats Dense

BM25 tends to outperform vector similarity when:

1. **The query contains distinctive low-frequency terms.** BM25 strongly rewards exact
   term overlap, especially for rare words. If a query asks about "Hohenzollern dynasty"
   and the relevant chunk contains exactly that phrase, BM25 finds it reliably.
   Dense retrieval may match semantically adjacent but terminologically distinct text.

2. **The embedding model has poor coverage of the domain.** General-purpose embedding
   models (including `text-embedding-3-small`) are less discriminative in highly
   specialized domains (legal, medical, code). BM25 is domain-agnostic.

3. **Chunks are short.** Very short chunks (< 50 tokens) provide little context for
   semantic embedding. BM25 is relatively more competitive here.

When `sparse` scores higher than `dense` on your benchmark, consider:
- Whether the corpus has many distinctive technical terms
- Whether the chunks are long enough for embedding to be effective
- Whether hybrid retrieval recovers the dense signal loss

---

## Hybrid Retrieval Failure Modes

The hybrid retriever (RRF over dense + sparse) generally outperforms either signal
alone. But it has specific failure cases:

### 1. Both signals fail the same query

RRF is a rank-combination strategy — it cannot recover signal that neither retriever
has. If a query uses paraphrase (no term overlap) on a topic the embedding model
doesn't handle well, hybrid will produce the same poor result as either alone.

### 2. BM25 dominates noisy results

If the corpus has many documents with overlapping boilerplate (e.g., legal disclaimers,
repeated headers), BM25 may rank irrelevant but term-similar chunks highly. RRF will
propagate this noise into the hybrid ranking if the dense signal is weaker.

### 3. Corpus not chunked for retrieval context

Hybrid retrieval does not fix chunking errors. If a relevant passage is split across
two chunks by the fixed-size strategy, no retriever will reliably reconstruct it as a
single result. The fix is at the chunking layer, not the retrieval layer.

---

## Comparing Chunking Strategies

A few patterns to look for when comparing `fixed_size`, `recursive`, and `semantic`
results on the same corpus:

- **`recursive` typically beats `fixed_size`** on structured corpora: natural boundaries
  mean fewer split-sentence chunks, which improves precision at the cost of slightly
  higher variance in chunk size.

- **`semantic` may underperform** on short corpora or corpora where the embedding model
  isn't well-calibrated to the domain. If `semantic` is worse than `recursive`, check
  `ndcg@1` — if the best result is also the first result for dense/hybrid, the semantic
  grouping may be creating chunks that are too broad.

- **`fixed_size` is the most predictable**: identical chunk counts per document,
  stable retrieval latency, easiest to reason about failure cases.

---

## Latency Interpretation

`retrieval_p50_ms` is the median query latency. This is the number to cite when
estimating user-facing response time in a low-traffic system.

`retrieval_p99_ms` is the 99th percentile. Use this to estimate worst-case latency.
A P99 more than 5× the P50 suggests high variance — common when the dense retriever's
embedding call experiences occasional API latency spikes.

`ingestion_s` is the total chunking + embedding + database write time for the full
corpus. This is not user-facing but matters for understanding how expensive re-indexing
is when the chunking strategy changes.
