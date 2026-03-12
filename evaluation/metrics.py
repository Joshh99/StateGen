# evaluation/metrics.py
# Compute token efficiency and correctness metrics from run results.
# Reads the JSONL files produced by run_bigcodebench.py.

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def load_results(results_dir: str, methods: Optional[List[str]] = None) -> List[dict]:
    """Load all *_solutions.jsonl files from results_dir into a flat list."""
    records = []
    for path in Path(results_dir).glob("*_solutions.jsonl"):
        method = path.stem.replace("_solutions", "")
        if methods and method not in methods:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec.setdefault("method", method)
                    records.append(rec)
    logger.info(f"Loaded {len(records)} result records from {results_dir}")
    return records


def load_eval_results(eval_json_path: str) -> Dict[str, str]:
    """
    Load BigCodeBench's eval_results.json and return task_id → status mapping.
    Status values: "pass", "fail", "timeout", etc.
    """
    with open(eval_json_path) as f:
        data = json.load(f)

    status = {}
    for task_id, results in data.get("eval", {}).items():
        if results:
            status[task_id] = results[0]["status"]
    return status


def compute_metrics(
    results: List[dict],
    eval_status: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """
    Compute per-method metrics.

    Args:
        results:     List of result dicts from load_results()
        eval_status: Optional BigCodeBench eval status (task_id → "pass"/"fail")
                     If None, uses passed_quick_check as proxy for correctness.

    Returns:
        Dict[method_name → Dict[metric_name → value]]

    Metrics:
        pass_at_1                   — fraction of tasks solved
        avg_total_tokens            — average tokens per task (all attempts)
        avg_tokens_correct          — average tokens when task was solved
        avg_tokens_failed           — average tokens when task was not solved
        wasted_token_rate           — tokens on failed tasks / total tokens
        avg_num_attempts            — average LLM calls per task
        avg_retries                 — average retries per task
        avg_wall_time               — average seconds per task
        token_overhead_vs_direct    — (avg_tokens / direct_avg_tokens) - 1
    """
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for rec in results:
        by_method[rec["method"]].append(rec)

    metrics: Dict[str, Dict] = {}
    for method, recs in by_method.items():
        n = len(recs)
        if n == 0:
            continue

        passed_list = []
        for rec in recs:
            task_id = rec["task_id"]
            if eval_status is not None:
                passed = eval_status.get(task_id, "fail") == "pass"
            else:
                # Fallback: use quick_check result from last attempt
                details = rec.get("attempt_details", [])
                passed = details[-1].get("passed_quick_check", False) if details else False
            passed_list.append(passed)

        total_tokens = [rec.get("total_tokens", 0) for rec in recs]
        input_tokens = [rec.get("total_input_tokens", 0) for rec in recs]
        output_tokens = [rec.get("total_output_tokens", 0) for rec in recs]
        num_attempts = [rec.get("num_attempts", 1) for rec in recs]
        num_retries  = [rec.get("num_retries", 0) for rec in recs]
        wall_times   = [rec.get("wall_time", 0.0) for rec in recs]

        correct_tokens = [t for t, p in zip(total_tokens, passed_list) if p]
        failed_tokens  = [t for t, p in zip(total_tokens, passed_list) if not p]

        total_tok_sum  = sum(total_tokens)
        failed_tok_sum = sum(failed_tokens)

        metrics[method] = {
            "n_tasks":                n,
            "pass_at_1":              sum(passed_list) / n,
            "avg_total_tokens":       total_tok_sum / n,
            "avg_input_tokens":       sum(input_tokens) / n,
            "avg_output_tokens":      sum(output_tokens) / n,
            "avg_tokens_correct":     (sum(correct_tokens) / len(correct_tokens)) if correct_tokens else None,
            "avg_tokens_failed":      (sum(failed_tokens) / len(failed_tokens)) if failed_tokens else None,
            "wasted_token_rate":      (failed_tok_sum / total_tok_sum) if total_tok_sum else 0.0,
            "avg_num_attempts":       sum(num_attempts) / n,
            "avg_retries":            sum(num_retries) / n,
            "avg_wall_time":          sum(wall_times) / n,
            "total_tokens_all_tasks": total_tok_sum,
        }

    # Token overhead vs direct_gen baseline
    direct_avg = metrics.get("direct_gen", {}).get("avg_total_tokens")
    if direct_avg:
        for method, m in metrics.items():
            if method != "direct_gen" and m["avg_total_tokens"]:
                m["token_overhead_vs_direct"] = (m["avg_total_tokens"] / direct_avg) - 1.0
            else:
                m["token_overhead_vs_direct"] = 0.0

    return metrics


def print_comparison_table(metrics: Dict[str, Dict]) -> None:
    """Print a human-readable comparison table to stdout."""
    methods = list(metrics.keys())
    if not methods:
        print("No results to display.")
        return

    cols = [
        ("pass@1",      "pass_at_1",           ".3f"),
        ("avg_tokens",  "avg_total_tokens",     ".0f"),
        ("tok/correct", "avg_tokens_correct",   ".0f"),
        ("wasted%",     "wasted_token_rate",     ".1%"),
        ("avg_retries", "avg_retries",           ".2f"),
        ("overhead",    "token_overhead_vs_direct", "+.1%"),
    ]

    header = f"{'Method':<20}" + "".join(f"{label:>14}" for label, _, _ in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for method in sorted(methods):
        m = metrics[method]
        row = f"{method:<20}"
        for _, key, fmt in cols:
            val = m.get(key)
            if val is None:
                cell = "—"
            else:
                try:
                    cell = format(val, fmt)
                except (ValueError, TypeError):
                    cell = str(val)
            row += f"{cell:>14}"
        print(row)

    print("=" * len(header))


def save_metrics(metrics: Dict[str, Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {path}")
