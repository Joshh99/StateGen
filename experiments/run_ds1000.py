#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Load .env if present ──────────────────────────────────────────────────────
_env_file = ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from agents.execution_engine import ExecutionEngine
from agents.llm_agent import LLMAgent
from baselines import ALL_BASELINES
from baselines.base import extract_code
from config import (
    DS1000_SUBSET,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_BASE_URL,
    LOG_FILE,
    LOG_LEVEL,
)
from core.token_tracker import TokenTracker
from data.ds1000_loader import load_ds1000

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(log_level: str, log_file: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)

DS1000_RESULTS_DIR = "results/ds1000"
DS1000_EVAL_TIMEOUT = 10  # seconds


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run baseline experiments on DS-1000 and measure token efficiency."
    )
    p.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        choices=["all"] + list(ALL_BASELINES.keys()),
        help="Which baseline(s) to run. Pass one or more names, or 'all'.",
    )
    p.add_argument(
        "--max_tasks",
        type=int,
        default=DS1000_SUBSET,
        help="Maximum number of DS-1000 tasks to evaluate (default: %(default)s).",
    )
    p.add_argument(
        "--provider",
        default=LLM_PROVIDER,
        help="LLM API provider (default: %(default)s).",
    )
    p.add_argument(
        "--model",
        default=LLM_MODEL,
        help="Model name for the chosen provider (default: %(default)s).",
    )
    p.add_argument(
        "--base_url",
        default=None,
        help="Base URL for openai_compatible providers (default: from config.py).",
    )
    p.add_argument(
        "--api_key",
        default=None,
        help="API key (falls back to env var).",
    )
    p.add_argument(
        "--results_dir",
        default=DS1000_RESULTS_DIR,
        help="Directory for output files (default: %(default)s).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip tasks already written to the solutions JSONL (crash recovery).",
    )
    p.add_argument(
        "--no_skip_existing",
        dest="skip_existing",
        action="store_false",
        help="Re-run all tasks even if results already exist.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Load tasks and print prompts without calling the LLM.",
    )
    return p.parse_args()


def extract_setup_code(prompt: str) -> str:
    """Extract setup code from the first <code>...</code> block in DS-1000 prompts."""
    match = re.search(r"<code>\s*\n(.*?)</code>", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# ── DS-1000 evaluation ───────────────────────────────────────────────────────

def evaluate_ds1000(solution: str, test_code: str, timeout: int = DS1000_EVAL_TIMEOUT) -> tuple[bool, str]:
    """
    Execution-based evaluation for DS-1000.

    Runs the solution code, then the test_code in the same namespace.
    If no exception → pass. If exception → fail.

    Uses subprocess for isolation and timeout safety.

    Returns:
        (passed, error_message)
    """
    harness = f"""{solution}

# ── DS-1000 test harness ──
{test_code}
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(harness)
        tmp = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            error_lines = result.stderr.strip().split("\n")
            error_msg = error_lines[-1] if error_lines else "Unknown error"
            return False, error_msg
        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s"
    except Exception as exc:
        return False, str(exc)
    finally:
        os.unlink(tmp)


# ── Per-method runner ─────────────────────────────────────────────────────────

def _load_done_task_ids(solutions_path: Path) -> set:
    done = set()
    if solutions_path.exists():
        with open(solutions_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def run_method(
    method_name: str,
    baseline_cls,
    tasks: list[dict],
    llm: LLMAgent,
    executor: ExecutionEngine,
    tracker: TokenTracker,
    results_dir: Path,
    skip_existing: bool,
) -> int:
    """Run one baseline method on all DS-1000 tasks."""
    solutions_path = results_dir / f"{method_name}_solutions.jsonl"
    done_ids = _load_done_task_ids(solutions_path) if skip_existing else set()

    if done_ids:
        logger.info(f"[{method_name}] Skipping {len(done_ids)} already-completed tasks.")

    # DS-1000 tasks are dicts; baselines expect data.loader.Task objects.
    # We create Task objects on-the-fly.
    from data.loader import Task

    baseline = baseline_cls(llm=llm, executor=executor, tracker=tracker)

    ran = 0
    with open(solutions_path, "a") as out_f:
        for ds_task in tasks:
            task_id = ds_task["task_id"]
            if task_id in done_ids:
                continue

            logger.info(f"[{method_name}] Solving {task_id} ...")
            t_start = time.time()

            # Convert dict → Task for baseline.solve()
            task = Task(
                task_id=task_id,
                prompt=ds_task["prompt"],
                test_code=ds_task["test_code"],
                entry_point="solution",
                dataset="ds1000",
                canonical_solution=ds_task["reference_code"],
                metadata={
                    "library": ds_task["library"],
                    "perturbation_type": ds_task["perturbation_type"],
                },
            )

            try:
                result = baseline.solve(task)
            except KeyboardInterrupt:
                logger.warning("Interrupted — partial results saved.")
                raise
            except Exception as exc:
                logger.error(f"[{method_name}] {task_id} crashed: {exc}", exc_info=True)
                record = {
                    "task_id":            task_id,
                    "solution":           "",
                    "method":             method_name,
                    "library":            ds_task["library"],
                    "passed":             False,
                    "error":              str(exc),
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens":       0,
                    "latency_ms":         None,
                    "num_attempts":       0,
                    "wall_time":          time.time() - t_start,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                ran += 1
                continue

            # Evaluate solution against DS-1000 test harness
            # passed, error = evaluate_ds1000(result.solution, ds_task["test_code"])
            setup = extract_setup_code(ds_task["prompt"])
            full_solution = f"{setup}\n{result.solution}" if setup else result.solution
            passed, error = evaluate_ds1000(full_solution, ds_task["test_code"])

            record = {
                "task_id":            task_id,
                "solution":           result.solution,
                "method":             method_name,
                "library":            ds_task["library"],
                "passed":             passed,
                "error":              error,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
                "total_tokens":       result.total_tokens,
                "latency_ms":         llm.last_latency_ms,
                "num_attempts":       result.num_attempts,
                "wall_time":          result.wall_time,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            ran += 1

            logger.info(
                f"[{method_name}] {task_id} {'PASS' if passed else 'FAIL'} "
                f"in {result.wall_time:.1f}s | tokens={result.total_tokens}"
            )

    logger.info(f"[{method_name}] Finished. Ran {ran} tasks -> {solutions_path}")
    return ran


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_ds1000_metrics(results_dir: Path) -> dict:
    """
    Compute pass@1 per library + overall, avg tokens, avg latency_ms
    from all *_solutions.jsonl files in results_dir.
    """
    all_records = []
    for jsonl_path in results_dir.glob("*_solutions.jsonl"):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not all_records:
        return {}

    # Group by method
    by_method: dict[str, list[dict]] = {}
    for rec in all_records:
        by_method.setdefault(rec["method"], []).append(rec)

    metrics = {}
    for method, records in by_method.items():
        total = len(records)
        passed = sum(1 for r in records if r.get("passed"))
        tokens = [r["total_tokens"] for r in records if r["total_tokens"] > 0]
        latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]

        # Per-library breakdown
        libs: dict[str, dict] = {}
        for r in records:
            lib = r.get("library", "Unknown")
            entry = libs.setdefault(lib, {"total": 0, "passed": 0})
            entry["total"] += 1
            if r.get("passed"):
                entry["passed"] += 1

        per_lib = {
            lib: round(v["passed"] / v["total"], 4) if v["total"] else 0.0
            for lib, v in libs.items()
        }

        metrics[method] = {
            "pass_at_1":       round(passed / total, 4) if total else 0.0,
            "pass_at_1_by_lib": per_lib,
            "total_tasks":     total,
            "total_passed":    passed,
            "avg_tokens":      round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
            "avg_latency_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        }

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _setup_logging(LOG_LEVEL, LOG_FILE)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tasks ────────────────────────────────────────────────────────────
    logger.info(f"Loading DS-1000 — max_tasks={args.max_tasks}")
    tasks = load_ds1000(max_tasks=args.max_tasks)
    logger.info(f"Loaded {len(tasks)} tasks.")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(tasks)} tasks loaded, not calling LLM")
        print(f"{'='*60}\n")
        for i, t in enumerate(tasks):
            print(f"-- Task {i+1}: {t['task_id']} ({t['library']}) --")
            print(f"Perturbation: {t['perturbation_type']}")
            print(f"Prompt ({len(t['prompt'])} chars):")
            print(t["prompt"][:500])
            if len(t["prompt"]) > 500:
                print(f"  ... ({len(t['prompt']) - 500} more chars)")
            print()
        return

    # ── Shared infrastructure ─────────────────────────────────────────────────
    tracker_path = results_dir / "token_events.jsonl"
    tracker = TokenTracker()

    llm = LLMAgent(
        provider=args.provider,
        model=args.model,
        tracker=tracker,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    executor = ExecutionEngine()

    logger.info(f"LLM: provider={args.provider}, model={args.model}")

    # ── Select methods ────────────────────────────────────────────────────────
    if "all" in args.method:
        methods_to_run = list(ALL_BASELINES.items())
    else:
        methods_to_run = [(m, ALL_BASELINES[m]) for m in args.method]

    # ── Run ───────────────────────────────────────────────────────────────────
    total_start = time.time()
    for method_name, baseline_cls in methods_to_run:
        logger.info(f"\n{'='*60}\nRunning method: {method_name}\n{'='*60}")
        run_method(
            method_name=method_name,
            baseline_cls=baseline_cls,
            tasks=tasks,
            llm=llm,
            executor=executor,
            tracker=tracker,
            results_dir=results_dir,
            skip_existing=args.skip_existing,
        )

    logger.info(f"\nGeneration done in {time.time() - total_start:.1f}s.")

    # ── Save token events ─────────────────────────────────────────────────────
    tracker.save(str(tracker_path))
    logger.info(f"Token events saved to {tracker_path}")

    # ── Save latency log ──────────────────────────────────────────────────────
    latency_path = results_dir / "latency_log.jsonl"
    with open(latency_path, "w") as f:
        for entry in llm.latency_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Latency log saved: {len(llm.latency_log)} entries -> {latency_path}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    logger.info("Computing metrics ...")
    metrics = compute_ds1000_metrics(results_dir)

    if metrics:
        print(f"\n{'='*60}")
        print("DS-1000 Results")
        print(f"{'='*60}")
        for method, m in metrics.items():
            print(f"\n  {method}:")
            print(f"    pass@1 (overall): {m['pass_at_1']:.4f}  ({m['total_passed']}/{m['total_tasks']})")
            for lib, score in m["pass_at_1_by_lib"].items():
                print(f"    pass@1 ({lib}):    {score:.4f}")
            print(f"    avg tokens:       {m['avg_tokens']:.1f}")
            print(f"    avg latency:      {m['avg_latency_ms']:.1f} ms")

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
