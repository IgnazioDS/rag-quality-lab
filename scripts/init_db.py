"""Initialize the pgvector schema for rag-quality-lab.

Run this after `docker compose up`:
    python scripts/init_db.py
"""
import logging
import sys

import psycopg2

sys.path.insert(0, ".")
from rag_quality_lab.config import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    source_doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_params JSONB NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks (corpus_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS corpora (
    corpus_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    doc_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    strategy_name TEXT NOT NULL,
    strategy_params JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""


def main() -> None:
    logger.info("Connecting to %s", config.database_url)
    conn = psycopg2.connect(config.database_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        logger.info("Schema initialized successfully")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
