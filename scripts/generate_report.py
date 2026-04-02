"""Generate Markdown report from benchmark result files.

Usage:
    python scripts/generate_report.py [--run-id RUN_ID] [--latest]
        [--compare RUN_ID RUN_ID] [--output PATH]
"""
import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, ".")
from rag_quality_lab.config import config
from rag_quality_lab.reporting import (
    find_latest_result,
    find_run_by_id,
    generate_comparison_report,
    generate_report,
    load_run,
)

console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Specific run ID to report on")
    group.add_argument("--latest", action="store_true", help="Report on the most recent run")
    group.add_argument(
        "--compare",
        nargs=2,
        metavar="RUN_ID",
        help="Compare two run IDs side by side",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (default: results/{run_id}_report.md)",
    )
    args = parser.parse_args()

    results_dir = config.results_dir

    if args.latest:
        try:
            path = find_latest_result(results_dir)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        run = load_run(path)
        report_md = generate_report(run)
        output_path = args.output or results_dir / f"{run['run_id']}_report.md"

    elif args.run_id:
        try:
            path = find_run_by_id(results_dir, args.run_id)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        run = load_run(path)
        report_md = generate_report(run)
        output_path = args.output or results_dir / f"{run['run_id']}_report.md"

    else:  # --compare
        run_id_a, run_id_b = args.compare
        try:
            path_a = find_run_by_id(results_dir, run_id_a)
            path_b = find_run_by_id(results_dir, run_id_b)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            sys.exit(1)
        run_a = load_run(path_a)
        run_b = load_run(path_b)
        report_md = generate_comparison_report(run_a, run_b)
        output_path = args.output or results_dir / f"compare_{run_id_a}_vs_{run_id_b}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    console.print(f"[green]Report written to {output_path}[/green]")
    console.print(report_md)


if __name__ == "__main__":
    main()
