from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_run(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_latest_result(results_dir: Path) -> Path:
    json_files = sorted(results_dir.glob("*.json"))
    result_files = [f for f in json_files if f.name != "README.md"]
    if not result_files:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")
    return result_files[-1]


def find_run_by_id(results_dir: Path, run_id: str) -> Path:
    matches = list(results_dir.glob(f"{run_id}*.json"))
    if not matches:
        raise FileNotFoundError(f"No result file found for run_id={run_id} in {results_dir}")
    return matches[0]


def _format_metrics_table(run: dict[str, Any]) -> str:
    aggregated = run["aggregated"]
    cfg = run["config"]

    k_values = [1, 3, 5, 10]
    metrics = ["hit_rate", "mrr", "ndcg", "precision", "recall"]

    header = f"### {cfg['chunker_name']} × {cfg['retriever_name']}\n\n"
    header += "| Metric |" + "".join(f" @{k} |" for k in k_values) + "\n"
    header += "|--------|" + "".join("-------|" for _ in k_values) + "\n"

    rows = []
    for metric in metrics:
        row = f"| {metric} |"
        for k in k_values:
            key = f"{metric}@{k}"
            val = aggregated.get(key, float("nan"))
            row += f" {val:.4f} |"
        rows.append(row)

    return header + "\n".join(rows) + "\n"


def _generate_observations(run: dict[str, Any]) -> list[str]:
    aggregated = run["aggregated"]
    observations = []

    hr5 = aggregated.get("hit_rate@5", 0.0)
    if hr5 >= 0.8:
        observations.append(f"Hit Rate @5 is strong at {hr5:.1%} — retrieval coverage is high.")
    elif hr5 < 0.5:
        observations.append(
            f"Hit Rate @5 is low at {hr5:.1%} — consider reviewing chunking granularity."
        )

    mrr5 = aggregated.get("mrr@5", 0.0)
    ndcg5 = aggregated.get("ndcg@5", 0.0)
    if ndcg5 > mrr5 + 0.1:
        observations.append(
            "NDCG@5 significantly exceeds MRR@5, suggesting relevant results exist "
            "in the top-5 but are not consistently ranked first."
        )

    p1 = aggregated.get("precision@1", 0.0)
    if p1 >= 0.7:
        observations.append(
            f"Precision@1 is {p1:.1%} — the top result is correct most of the time."
        )

    return observations


def generate_report(run: dict[str, Any]) -> str:
    cfg = run["config"]
    timing = run["timing"]
    run_id = run["run_id"]

    lines = [
        f"# Benchmark Report — {run_id}",
        "",
        "## Configuration",
        "",
        f"- **Corpus**: {cfg['corpus_name']}",
        f"- **Chunker**: {cfg['chunker_name']} — params: {cfg['chunker_params']}",
        f"- **Retriever**: {cfg['retriever_name']} — params: {cfg['retriever_params']}",
        f"- **Embedding model**: {cfg['embedding_model']}",
        "",
        "## Metrics Summary",
        "",
        _format_metrics_table(run),
        "",
        "## Key Findings",
        "",
    ]

    observations = _generate_observations(run)
    for obs in observations:
        lines.append(f"- {obs}")
    if not observations:
        lines.append("- No automated observations generated.")

    lines += [
        "",
        "## Latency",
        "",
        "| Phase | Value |",
        "|-------|-------|",
        f"| Ingestion | {timing.get('ingestion_s', 0.0):.2f}s |",
        f"| Retrieval P50 | {timing.get('retrieval_p50_ms', 0.0):.1f}ms |",
        f"| Retrieval P95 | {timing.get('retrieval_p95_ms', 0.0):.1f}ms |",
        f"| Retrieval P99 | {timing.get('retrieval_p99_ms', 0.0):.1f}ms |",
        "",
        "## Per-Query Breakdown",
        "",
        "| Query ID | hit_rate@5 | mrr@5 | ndcg@5 | precision@5 | recall@5 |",
        "|----------|-----------|-------|--------|-------------|----------|",
    ]

    for pq in run.get("per_query", []):
        m = pq["metrics"]
        lines.append(
            f"| {pq['query_id']} "
            f"| {m.get('hit_rate@5', 0.0):.4f} "
            f"| {m.get('mrr@5', 0.0):.4f} "
            f"| {m.get('ndcg@5', 0.0):.4f} "
            f"| {m.get('precision@5', 0.0):.4f} "
            f"| {m.get('recall@5', 0.0):.4f} |"
        )

    return "\n".join(lines)


def generate_comparison_report(run_a: dict[str, Any], run_b: dict[str, Any]) -> str:
    id_a = run_a["run_id"]
    id_b = run_b["run_id"]
    agg_a = run_a["aggregated"]
    agg_b = run_b["aggregated"]

    all_keys = sorted(set(agg_a.keys()) | set(agg_b.keys()))

    lines = [
        f"# Comparison Report: {id_a} vs {id_b}",
        "",
        f"**Run A**: {run_a['config']['chunker_name']} × {run_a['config']['retriever_name']}",
        f"**Run B**: {run_b['config']['chunker_name']} × {run_b['config']['retriever_name']}",
        "",
        "## Metric Comparison",
        "",
        "| Metric | Run A | Run B | Delta |",
        "|--------|-------|-------|-------|",
    ]

    for key in all_keys:
        val_a = agg_a.get(key, float("nan"))
        val_b = agg_b.get(key, float("nan"))
        delta = val_b - val_a
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {key} | {val_a:.4f} | {val_b:.4f} | {sign}{delta:.4f} |")

    return "\n".join(lines)
