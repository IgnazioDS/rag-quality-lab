"""Download and prepare the SQuAD-200 benchmark corpus.

Usage:
    python scripts/ingest.py [--corpus {squad-200}] [--force]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import requests
from rich.console import Console
from rich.progress import track

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
NUM_CONTEXTS = 200
NUM_QUERIES = 100

DATA_DIR = Path("data")
CORPORA_DIR = DATA_DIR / "corpora"
QUERIES_DIR = DATA_DIR / "queries"


def download_squad(force: bool = False) -> dict:
    cache_path = DATA_DIR / "squad_train.json"
    if cache_path.exists() and not force:
        logger.info("Using cached SQuAD data at %s", cache_path)
        return json.loads(cache_path.read_text())

    console.print("[bold blue]Downloading SQuAD v1.1 training set...[/bold blue]")
    response = requests.get(SQUAD_URL, timeout=60)
    response.raise_for_status()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text)
    logger.info("Downloaded SQuAD to %s", cache_path)
    return response.json()


def extract_squad_200(data: dict, output_dir: Path, queries_path: Path, force: bool) -> None:
    if output_dir.exists() and any(output_dir.glob("*.txt")) and not force:
        console.print(
            f"[yellow]Corpus already exists at {output_dir}. "
            "Use --force to re-extract.[/yellow]"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    seen_contexts: set[str] = set()
    contexts: list[tuple[str, str]] = []  # (doc_id, context_text)
    queries: list[dict] = []

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            if context in seen_contexts:
                continue
            seen_contexts.add(context)

            doc_id = f"squad_{len(contexts):04d}"
            contexts.append((doc_id, context))

            # Extract queries from this paragraph
            if len(queries) < NUM_QUERIES:
                for qa in paragraph["qas"]:
                    if len(queries) >= NUM_QUERIES:
                        break
                    if not qa.get("answers"):
                        continue
                    # Relevant chunk is determined at retrieval time by matching the context.
                    # For SQuAD, we mark the source doc as the relevant document.
                    # The exact chunk_id depends on the chunker strategy, so we store
                    # the doc_id and let the pipeline resolve it, or use a special sentinel.
                    # Convention: relevant_chunk_ids uses the doc_id as a prefix match key.
                    queries.append(
                        {
                            "query_id": f"q{len(queries):04d}",
                            "query_text": qa["question"],
                            "relevant_doc_id": doc_id,
                            # Populated by the pipeline after chunking — see note below.
                            "relevant_chunk_ids": [],
                            "reference_answer": qa["answers"][0]["text"],
                        }
                    )

            if len(contexts) >= NUM_CONTEXTS:
                break
        if len(contexts) >= NUM_CONTEXTS:
            break

    console.print(f"Extracted {len(contexts)} contexts and {len(queries)} queries")

    for doc_id, text in track(contexts, description="Writing corpus files..."):
        (output_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

    # Post-process queries: assign relevant_chunk_ids using fixed_size chunker defaults
    # as a baseline approximation (chunk_0 of the relevant doc is always relevant).
    # This is a pragmatic choice: SQuAD answers are in the full paragraph, so the
    # first chunk of the relevant doc will contain the answer for short paragraphs.
    for q in queries:
        doc_id = q["relevant_doc_id"]
        # Mark chunk_0 as the primary relevant chunk; the pipeline may augment this.
        q["relevant_chunk_ids"] = [f"{doc_id}_chunk_0"]

    queries_path.write_text(
        "\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8"
    )
    console.print(f"[green]Wrote {len(queries)} queries to {queries_path}[/green]")
    console.print(f"[green]Corpus written to {output_dir}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark corpus")
    parser.add_argument(
        "--corpus",
        choices=["squad-200"],
        default="squad-200",
        help="Corpus to prepare (default: squad-200)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if files exist",
    )
    args = parser.parse_args()

    if args.corpus == "squad-200":
        data = download_squad(force=args.force)
        corpus_dir = CORPORA_DIR / "squad-200"
        queries_path = QUERIES_DIR / "squad-200.jsonl"
        extract_squad_200(data, corpus_dir, queries_path, force=args.force)
    else:
        console.print(f"[red]Unknown corpus: {args.corpus}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
