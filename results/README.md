# Benchmark Results

This directory contains committed benchmark result artifacts. Results are the primary
deliverable of this repository — they are not build artifacts and are not in `.gitignore`.

Each run produces one JSON file. The git history shows exactly when numbers changed and why.

---

## JSON Schema

Each result file contains a single JSON object with the following structure:

```json
{
  "run_id": "20260402T143022",
  "config": { ... },
  "per_query": [ ... ],
  "aggregated": { ... },
  "timing": { ... }
}
```

### `run_id`
ISO timestamp in compact format: `YYYYMMDDTHHmmss` (UTC). Used as the primary identifier
for a run. The filename is `{run_id}_{chunker}_{retriever}.json`.

### `config`
Full snapshot of the parameters used for this run. Reproducibility depends on this field
being an exact record — never modify it after a run.

| Field | Type | Description |
|-------|------|-------------|
| `corpus_name` | string | Name of the corpus directory used |
| `chunker_name` | string | Strategy name (`fixed_size`, `recursive`, `semantic`) |
| `chunker_params` | object | All chunker constructor parameters |
| `retriever_name` | string | Retriever name (`dense`, `sparse`, `hybrid`) |
| `retriever_params` | object | Retriever configuration (e.g. `rrf_k` for hybrid) |
| `embedding_model` | string | OpenAI model used for embeddings |

### `per_query`
Array of per-query metric values. One entry per query in the evaluation set.

```json
{
  "query_id": "q0042",
  "metrics": {
    "hit_rate@1": 0.0,
    "hit_rate@5": 1.0,
    "mrr@5": 0.333,
    "ndcg@5": 0.5,
    "precision@5": 0.2,
    "recall@5": 1.0
  }
}
```

### `aggregated`
Mean of each metric across all queries. Keys follow the pattern `{metric}@{K}`.

The K value is the cutoff: `ndcg@5` means NDCG evaluated considering only the top-5
retrieved results. A chunk ranked 6th or lower does not contribute to `ndcg@5`.

All values are floats in `[0.0, 1.0]`.

### `timing`
Latency measurements for the run.

| Field | Unit | Description |
|-------|------|-------------|
| `ingestion_s` | seconds | Total time to chunk, embed, and upsert all documents |
| `retrieval_p50_ms` | milliseconds | Median retrieval latency across all queries |
| `retrieval_p95_ms` | milliseconds | 95th percentile retrieval latency |
| `retrieval_p99_ms` | milliseconds | 99th percentile retrieval latency |

P50/P95/P99 are computed with `numpy.percentile` over the per-query retrieval times.
They measure wall-clock time from the start of `retriever.retrieve()` to the return of
the result list — including embedding the query for dense/hybrid retrievers.

---

## Comparing Runs

Use `generate_report.py --compare` to produce a diff table between two run IDs:

```bash
python scripts/generate_report.py --compare 20260402T143022 20260402T161500
```

This writes a Markdown file to `results/compare_{id_a}_vs_{id_b}.md` showing the delta
for every metric at every K. A positive delta means run B improved over run A.

---

## Interpreting the Numbers

See [`docs/results-interpretation.md`](../docs/results-interpretation.md) for a full guide
on what to look at first, what constitutes a meaningful difference, and when to prefer
one metric over another.

The short version:
- Start with `hit_rate@5`. If this is below 0.6, the chunking strategy or retriever is
  fundamentally failing to surface relevant content.
- Use `mrr@5` to understand if the first relevant result is being ranked highly.
- Use `ndcg@10` when you want to reward retrieval systems that surface multiple relevant
  chunks in a ranked order.
- Latency P99 matters when the retriever is called in a user-facing loop. P50 matters
  when you're estimating total benchmark runtime.
