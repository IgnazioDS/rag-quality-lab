# rag-quality-lab

A structured benchmarking environment for RAG retrieval quality. Evaluates chunking
strategies and retrieval approaches against measurable metrics on a reproducible dataset.
Results are committed to the repo — findings are not ephemeral.

Built to answer a specific engineering question: given a fixed corpus, which combination
of chunking strategy and retrieval method produces the highest quality results, and by
how much?

---

## Metrics

| Metric | Measures | When it matters |
|--------|----------|-----------------|
| Hit Rate @ K | Did any relevant chunk appear in top-K results? | Baseline coverage check — are you finding anything? |
| MRR @ K | Mean reciprocal rank of the first relevant result | When position of the first relevant hit matters more than total coverage |
| NDCG @ K | Normalized discounted cumulative gain (binary relevance) | When you care about ranking quality across all K results |
| Precision @ K | Fraction of top-K results that are relevant | When result quality matters more than recall |
| Recall @ K | Fraction of relevant chunks found in top-K | When missing a relevant chunk is the primary failure mode |

All metrics are evaluated at K ∈ {1, 3, 5, 10}. Aggregated values are means across
all queries in the evaluation set.

---

## Chunking Strategies

| Strategy | Method | Key Parameter | Best Suited For |
|----------|--------|---------------|-----------------|
| `fixed_size` | Split by character count with overlap | `chunk_size` (default: 512 chars), `overlap` (default: 64) | Homogeneous text, baseline comparison |
| `recursive` | Split by separator hierarchy: paragraphs → sentences → characters; bounded by token count | `max_tokens` (default: 400) | Structured documents with natural paragraph boundaries |
| `semantic` | Group consecutive sentences by embedding similarity; fall back to token limit | `similarity_threshold` (default: 0.8), `max_tokens` (default: 400) | Documents where topic shifts do not align with paragraph breaks |

All chunkers produce `Chunk` objects with full metadata: `source_doc_id`, `chunk_index`,
`char_start`, `char_end`, `token_count`, `strategy_name`, `strategy_params`.

---

## Retrievers

| Retriever | Method | Notes |
|-----------|--------|-------|
| `dense` | pgvector cosine similarity (`<=>` operator) | Requires embedding at query time; strong on semantic similarity |
| `sparse` | BM25 via rank_bm25 | Keyword-based; strong on exact term overlap; no embeddings at query time |
| `hybrid` | Reciprocal Rank Fusion over dense + sparse | Default RRF k=60; configurable per-signal weights |

Retrieval is implemented directly — no LangChain, no LlamaIndex. pgvector queries use
raw psycopg2. BM25 index is built in-memory at retriever initialization.

---

## Quickstart

```bash
# 1. Start the database
make up

# 2. Install dependencies
make install

# 3. Download and prepare the benchmark corpus (SQuAD, 200 contexts / 100 queries)
make ingest

# 4. Run the full benchmark — all chunking strategies × all retrievers
make benchmark

# 5. Generate the Markdown report from the latest results
make report

# 6. Run the test suite
make test
```

End-to-end time on first run: approximately 8–12 minutes depending on embedding latency.
Subsequent runs skip embedding if chunks are already indexed (idempotent ingest).

---

## Results

Published benchmark results are in [`results/`](results/).

Each run produces a JSON artifact with the full config snapshot, per-query results,
aggregated metrics, and retrieval latency distributions. Results are committed so
findings are reproducible and the delta between runs is visible in git history.

See [`results/README.md`](results/README.md) for the result schema and how to interpret the numbers.

---

## Design Decisions

**1. Raw SQL over ORM.**
The schema is three tables. An ORM adds more abstraction than the complexity warrants.
When a retrieval query uses the `<=>` operator with a cast and a LIMIT, you should
see exactly that — not a method chain that generates it invisibly.

**2. Token count over character count as the size constraint.**
LLM context limits are measured in tokens, not characters. A 512-character chunk of
code and a 512-character chunk of prose produce very different token counts. Chunkers
that use characters as the primary size unit produce unpredictable context window
behavior. tiktoken is the authoritative counter here.

**3. RRF over weighted score fusion.**
Hybrid retrieval via score fusion requires normalizing scores across two fundamentally
different score distributions (cosine similarity and BM25). RRF avoids this by operating
on ranks, which are already normalized. The loss in precision at score level is worth
the gain in robustness. Weights on `dense_weight` and `sparse_weight` are provided
for experimentation but RRF k=60 is the production-safe default.

**4. SQuAD as the default corpus.**
SQuAD provides: a large enough corpus to show real retrieval behavior (200 contexts),
pre-defined question-answer pairs that serve directly as evaluation queries, and
public availability with no license restrictions. The alternative — asking users to
provide their own corpus — makes benchmarks non-reproducible and non-comparable
across runs and forks.

**5. LLM judge uses structured output (function calling).**
Free-form judge responses require post-processing and produce inconsistent score
boundaries. Structured output via OpenAI function calling enforces `float` in `[0, 1]`
at the API level. Results are cached by `sha256(model + query + chunk + reference)` to
avoid redundant API calls — at scale, the LLM judge is the dominant cost.

**6. No async.**
Benchmarking tools need predictable execution order and straightforward profiling.
Asyncio adds scheduling non-determinism that complicates latency measurement.
The embedding and database calls here are blocking by design.

**7. No framework wrappers.**
This repo exists to show what retrieval quality actually depends on at the
implementation level. Wrapping the retrieval pipeline in a framework would obscure
the decisions that determine quality. Every parameter that matters is visible
in the code that uses it.

---

## Extending

### Add a new chunking strategy

1. Create `rag_quality_lab/chunkers/your_strategy.py`
2. Inherit from `BaseChunker` and implement `chunk(text: str, doc_id: str) -> list[Chunk]`
3. Ensure all `Chunk.metadata` fields are populated, including `strategy_name` and `strategy_params`
4. Register the chunker in `chunkers/__init__.py`
5. Add tests in `tests/test_chunkers.py` with deterministic assertions

### Add a new retriever

1. Create `rag_quality_lab/retrievers/your_retriever.py`
2. Inherit from `BaseRetriever` and implement `retrieve(query: str, top_k: int) -> list[RetrievalResult]`
3. Register the retriever in `retrievers/__init__.py`
4. Add it to the benchmark matrix in `scripts/run_benchmark.py`

### Use a custom corpus

The ingest script accepts any directory of `.txt` files with a companion `queries.jsonl`:

```bash
python scripts/ingest.py --corpus-dir /path/to/corpus --queries /path/to/queries.jsonl
```

`queries.jsonl` format — one record per line:

```json
{"query_id": "q001", "query_text": "What is X?", "relevant_chunk_ids": ["doc1_chunk3"], "reference_answer": "X is..."}
```

---

## Environment

Copy `.env.example` to `.env` and set:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | Used for embeddings and LLM judge |
| `DATABASE_URL` | Yes | — | PostgreSQL connection string with pgvector |
| `LAB_EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `LAB_JUDGE_MODEL` | No | `gpt-4o-mini` | Model used for LLM-as-judge scoring |
| `LAB_JUDGE_CACHE_PATH` | No | `.cache/judge.json` | Path for LLM judge result cache |
| `LAB_DEFAULT_TOP_K` | No | `10` | Maximum K for retrieval during benchmark |
| `LAB_RESULTS_DIR` | No | `results/` | Directory for benchmark result artifacts |

---

## Stack

Python 3.11+ · pgvector · psycopg2 · rank_bm25 · openai · tiktoken · numpy · pandas · pydantic v2 · rich · pytest · ruff · mypy · Docker Compose
