# CLAUDE.md — rag-quality-lab

This file defines how Claude Code should work within this repository.
Read it fully before making any changes. It is a constraint document, not a suggestion list.

---

## Project Purpose

This is a RAG benchmarking laboratory. It evaluates chunking strategies and retrieval approaches
against measurable quality metrics using real datasets. The output is reproducible, committed
benchmark results — not a framework, not a SaaS product, not a tutorial.

Everything here must look like internal engineering work, not a portfolio piece written for
an audience. Code should be written for the next engineer who has to maintain it, not
for someone learning RAG for the first time.

---

## Architecture Principles

**No LangChain. No LlamaIndex.**
All retrieval is implemented directly: raw SQL against pgvector, rank_bm25 for sparse,
manual RRF fusion. If you find yourself importing either library, stop and implement
the functionality directly. The point of this repo is demonstrating what those abstractions
are doing underneath. Wrapping them defeats the purpose.

**No async.**
This is a benchmarking tool. Everything is synchronous. Do not introduce asyncio, anyio,
or concurrent.futures unless explicitly asked. Benchmarks need predictable execution,
not concurrency.

**No web interface.**
No Streamlit, no FastAPI server, no Gradio. Everything runs from the CLI via scripts/
or make targets. If something needs to be visualized, it goes in a Markdown report.

**Raw SQL over ORM.**
pgvector queries use psycopg2 directly. The schema is simple enough that an ORM adds
more indirection than value. When you write a query, the person reading it should see
exactly what hits the database.

**Explicit over implicit.**
Every config value is declared in config.py as a Pydantic BaseSettings field with a
default and a description. Nothing reads from os.environ directly in business logic.
Nothing is hardcoded. .env.example covers every required variable.

---

## Code Standards

### Type hints

Required on every function signature. `Any` is banned unless the only alternative is
a 50-line generic type that obscures more than it clarifies. Pydantic v2 for config and
API-boundary models. Dataclasses for internal data structures (Chunk, RetrievalResult).

### Logging

Use Python's `logging` module. Module-level loggers: `logger = logging.getLogger(__name__)`.
No print statements in library code. Scripts (scripts/) may use rich for terminal output.
Log at DEBUG for per-item operations, INFO for pipeline stage boundaries, WARNING for
degraded behavior, ERROR for failures.

### Error handling

Fail loudly. Do not swallow exceptions with bare `except: pass`. If a configuration
value is missing, raise a clear ValueError that names the variable. If an embedding
call fails, let it propagate with context. The evaluation harness catches errors at
the pipeline level and records them in results — library code does not catch and hide.

### Tests

Tests in tests/ are real assertions against known values, not smoke tests.

- test_chunkers.py: deterministic inputs, assert exact chunk counts, offsets, token counts
- test_metrics.py: hand-calculated ground truth, assert exact float values (use pytest.approx)
- test_pipeline.py: integration test against a local pgvector instance, requires docker compose up

Do not write tests that just assert `len(results) > 0`. Write tests that assert
`results[0].chunk_index == 0` and `results[0].token_count == 47`.

---

## File and Module Conventions

### Chunkers

Every chunker inherits from `chunkers/base.py:BaseChunker`.
Required interface: `chunk(text: str, doc_id: str) -> list[Chunk]`
Every Chunk must include: text, metadata.source_doc_id, metadata.chunk_index,
metadata.char_start, metadata.char_end, metadata.token_count,
metadata.strategy_name, metadata.strategy_params.

### Retrievers

Every retriever inherits from `retrievers/base.py:BaseRetriever`.
Required interface: `retrieve(query: str, top_k: int) -> list[RetrievalResult]`
Every RetrievalResult must include: chunk_id, chunk_text, score, rank, retriever_name.

### Evaluators

metrics.py contains pure functions only. No class state. Signature pattern:
`def hit_rate_at_k(results: list[RetrievalResult], relevant_ids: set[str], k: int) -> float`
All metric functions return float in [0.0, 1.0].

llm_judge.py uses OpenAI structured output (function calling). Results are cached
to a JSON file keyed by `sha256(model + query + chunk_text + reference_answer)`.
Cache path is configurable via LAB_JUDGE_CACHE_PATH in config.

### Results

Every benchmark run writes a JSON artifact to results/ with:

- run_id: ISO timestamp
- config: full snapshot of chunker params, retriever params, model, corpus name
- per_query: list of per-query metric values
- aggregated: mean across all queries per metric/K combination
- timing: ingestion_s, retrieval_p50_ms, retrieval_p95_ms, retrieval_p99_ms

Results JSON files are committed to the repo. They are the primary deliverable.
Do not .gitignore them.

---

## Makefile

All common operations must have a make target. When adding new scripts, add a target.
Targets must be documented with a `## comment` for `make help` output.
No make target should require arguments to function with sensible defaults.

Current targets: up, down, install, ingest, benchmark, report, test, clean, help.

---

## What Not to Build

If you find yourself building any of the following, stop and reconsider:

- A REST API or any server process
- A configuration UI or interactive prompt system
- A plugin architecture or hook system
- A multi-tenant data model
- Anything that requires a message queue
- Notebook (.ipynb) files

This project has one job: load a corpus, chunk it multiple ways, retrieve against queries,
measure the quality difference, and report the numbers. Keep that scope.

---

## Dependencies

Pinned in pyproject.toml. Do not add new dependencies without a comment explaining
why the stdlib or an existing dependency cannot satisfy the requirement.

Core: psycopg2-binary, openai, tiktoken, rank_bm25, numpy, pandas, pydantic, rich
Dev: pytest, ruff, mypy

pgvector extension is managed by Docker Compose. Do not assume it is installed
on the host system. The Makefile `up` target is the only supported local setup path.

---

## Environment Variables

All variables are defined in .env.example with descriptions.
All variables are loaded through `config.py:LabConfig` (Pydantic BaseSettings).
Required variables with no default: OPENAI_API_KEY, DATABASE_URL.
Everything else has a documented default.

Never read os.environ directly in library code. Import config and use the settings object.

---

## Commit Scope

When making changes, limit each commit to one logical unit:

- Add a chunker → chunker file + tests + updated **init**
- Add a metric → metrics.py addition + test cases
- Benchmark run → results JSON + updated results/README.md

Do not bundle unrelated changes. Results files should be committed separately from
code changes so the git history shows when numbers changed and why.
