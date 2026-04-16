#!/usr/bin/env python3
# experiments/merge_results.py
# Merge JSONL result files produced by different teammates on different machines.
#
# Each teammate runs one method on all 148 tasks and pushes their
# results/<method>_solutions.jsonl to their git branch.
# This script pulls them all together into a single combined results directory,
# checks for duplicates, and recomputes the comparison table.
#
# Usage:
#   python experiments/merge_results.py \
#       --input results/ results_gpt4omini/ \
#       --output results/merged/
#
#   python experiments/merge_results.py \
#       --input results/ \
#       --output results/merged/ \
#       --methods stategen stategen_no_memory stategen_full_backtrack

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics, load_results, print_comparison_table, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"{path}:{i} — skipping malformed line: {e}")
    return records


def _read_manifest(d: Path) -> dict:
    """Read manifest.json from an experiment folder if present."""
    p = d / "manifest.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _find_jsonl_files(input_dirs: list, methods: list) -> dict:
    """
    Scan input directories for *_solutions.jsonl files.
    Also prints the manifest (model/provider) for each folder found.
    Returns {method_name: [list of Path objects]}.
    """
    found = defaultdict(list)
    for raw_d in input_dirs:
        d = Path(raw_d)
        if not d.exists():
            logger.warning(f"Input directory not found: {d}")
            continue

        manifest = _read_manifest(d)
        if manifest:
            logger.info(
                f"  [{d}] model={manifest.get('model','?')} "
                f"provider={manifest.get('provider','?')} "
                f"date={manifest.get('date','?')}"
            )
        else:
            logger.info(f"  [{d}] (no manifest.json)")

        # Also scan one level of subdirectories (results/exp_name/)
        dirs_to_scan = [d] + [p for p in d.iterdir() if p.is_dir()]
        for scan_dir in dirs_to_scan:
            for path in sorted(scan_dir.glob("*_solutions.jsonl")):
                method = path.stem.replace("_solutions", "")
                if methods and method not in methods:
                    continue
                found[method].append(path)
                logger.info(f"    Found {path}")

    return dict(found)


def _merge_method(method: str, paths: list) -> list:
    """
    Merge all JSONL files for one method.
    Raises ValueError if any task_id appears more than once.
    """
    seen_ids = {}   # task_id → source file
    merged   = []

    for path in paths:
        records = _load_jsonl(path)
        for rec in records:
            tid = rec.get("task_id", "")
            if tid in seen_ids:
                raise ValueError(
                    f"[{method}] Duplicate task_id '{tid}' "
                    f"found in both {seen_ids[tid]} and {path}.\n"
                    f"Check that teammates ran non-overlapping task ranges."
                )
            seen_ids[tid] = path
            merged.append(rec)

    logger.info(f"[{method}] Merged {len(merged)} tasks from {len(paths)} file(s).")
    return merged


def _write_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _coverage_report(method: str, records: list, expected: int = 148) -> None:
    """Warn if any task_ids are missing from the merged results."""
    found = len(records)
    if found < expected:
        logger.warning(
            f"[{method}] Only {found}/{expected} tasks present — "
            f"{expected - found} tasks missing. Results may be incomplete."
        )
    else:
        logger.info(f"[{method}] Full coverage: {found}/{expected} tasks.")


def _eval_method_report(records: list, method: str) -> None:
    """Show how many tasks used real BCB eval vs quick_check proxy."""
    real  = sum(1 for r in records if r.get("attempt_details") and
                any(d.get("call_type") == "final_eval" for d in r.get("attempt_details", [])))
    proxy = len(records) - real
    if proxy:
        logger.warning(
            f"[{method}] {proxy} task(s) used quick_check proxy "
            f"(BCB server was unreachable). Pass@1 for those tasks is approximate."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge per-teammate result JSONL files and recompute metrics."
    )
    p.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="DIR",
        help="One or more results directories to merge (e.g. results/ results_gpt4omini/).",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for merged JSONL files and metrics.",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=None,
        metavar="METHOD",
        help=(
            "Methods to merge. Default: all *_solutions.jsonl files found. "
            "Example: --methods stategen stategen_no_memory direct_gen"
        ),
    )
    p.add_argument(
        "--expected_tasks",
        type=int,
        default=148,
        help="Expected number of tasks per method (default: 148). Used for coverage check.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning input directories: {args.input}")
    files_by_method = _find_jsonl_files(args.input, args.methods or [])

    if not files_by_method:
        logger.error("No *_solutions.jsonl files found in the input directories.")
        sys.exit(1)

    logger.info(f"\nMerging {len(files_by_method)} method(s): {list(files_by_method)}\n")

    for method, paths in sorted(files_by_method.items()):
        try:
            records = _merge_method(method, paths)
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

        _coverage_report(method, records, expected=args.expected_tasks)
        _eval_method_report(records, method)

        out_path = output_dir / f"{method}_solutions.jsonl"
        _write_jsonl(records, out_path)
        logger.info(f"[{method}] Written → {out_path}\n")

    # Recompute metrics on the merged directory
    logger.info("=" * 60)
    logger.info("Computing metrics on merged results ...")
    results  = load_results(str(output_dir))
    metrics  = compute_metrics(results)
    print_comparison_table(metrics)

    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, str(metrics_path))
    logger.info(f"Metrics saved → {metrics_path}")


if __name__ == "__main__":
    main()
