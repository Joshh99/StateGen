#!/usr/bin/env python3
# experiments/run_bigcodebench.py
# Main experiment runner for the token-efficiency study on BigCodeBench.
#
# Usage examples:
#   python experiments/run_bigcodebench.py --method all --max_tasks 50
#   python experiments/run_bigcodebench.py --method all --max_tasks 50 --remote_eval
#   python experiments/run_bigcodebench.py --method self_debugging --max_tasks 20 --model gpt-4o-mini

import argparse
import json
import logging
import os
import subprocess
import sys
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
from core.bcb_evaluator import BCBEvaluator
from config import (
    BIGCODE_SUBSET,
    BIGCODE_USE_INSTRUCT,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_BASE_URL,
    LOG_FILE,
    LOG_LEVEL,
    RESULTS_DIR,
)
from core.token_tracker import TokenTracker
from data.loader import BigCodeBenchLoader
from evaluation.metrics import (
    compute_metrics,
    load_eval_results,
    load_results,
    print_comparison_table,
    save_metrics,
)

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run baseline experiments on BigCodeBench and measure token efficiency."
    )
    p.add_argument(
        "--method",
        nargs="+",
        default=["all"],
        choices=["all"] + list(ALL_BASELINES.keys()),
        help=(
            "Which baseline(s) to run. Pass one or more names, or 'all'. "
            "Examples: --method self_debugging self_planning"
        ),
    )
    p.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of BigCodeBench tasks to evaluate (default: all tasks in subset).",
    )
    p.add_argument(
        "--subset",
        default="hard",
        choices=["complete", "hard"],
        help="BigCodeBench subset to use (default: %(default)s).",
    )
    p.add_argument(
        "--use_instruct",
        action=argparse.BooleanOptionalAction,
        default=BIGCODE_USE_INSTRUCT,
        help="Use shorter instruct_prompt instead of complete_prompt.",
    )
    p.add_argument(
        "--provider",
        default=LLM_PROVIDER,
        choices=["openai", "anthropic", "openai_compatible"],
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
        help="API key (falls back to env var OPENAI_API_KEY / ANTHROPIC_API_KEY).",
    )
    p.add_argument(
        "--results_dir",
        default=RESULTS_DIR,
        help="Directory for output JSONL files (default: %(default)s).",
    )
    p.add_argument(
        "--task_ids",
        nargs="*",
        default=None,
        help="Specific task IDs to run (overrides --max_tasks).",
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
        "--bcb_server",
        default=None,
        metavar="URL",
        help=(
            "URL of the BigCodeBench API server for real-time evaluation during generation. "
            "e.g. --bcb_server http://localhost:8000  "
            "When set, _evaluate() calls /evaluate instead of quick_check. "
            "The server must be running (bigcodebench-new/bcb_server/server.py)."
        ),
    )
    p.add_argument(
        "--remote_eval",
        action="store_true",
        default=False,
        help="After generation, sanitize and evaluate for official pass@1 scores.",
    )
    p.add_argument(
        "--execution",
        default="docker",
        choices=["docker", "gradio"],
        help=(
            "Execution backend for official evaluation. "
            "'docker' runs locally via bigcodebench/bigcodebench-evaluate image "
            "(free, all packages pre-installed, recommended). "
            "'gradio' uses the remote HuggingFace Space API. "
            "Default: docker"
        ),
    )
    return p.parse_args()


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
    tasks,
    llm: LLMAgent,
    executor: ExecutionEngine,
    tracker: TokenTracker,
    results_dir: Path,
    skip_existing: bool,
    bcb_evaluator=None,
) -> int:
    solutions_path = results_dir / f"{method_name}_solutions.jsonl"
    done_ids = _load_done_task_ids(solutions_path) if skip_existing else set()

    if done_ids:
        logger.info(f"[{method_name}] Skipping {len(done_ids)} already-completed tasks.")

    baseline = baseline_cls(llm=llm, executor=executor, tracker=tracker, bcb_evaluator=bcb_evaluator)

    ran = 0
    with open(solutions_path, "a") as out_f:
        for task in tasks:
            if task.task_id in done_ids:
                continue

            logger.info(f"[{method_name}] Solving {task.task_id} ...")
            t_start = time.time()

            try:
                result = baseline.solve(task)
            except KeyboardInterrupt:
                logger.warning("Interrupted — partial results saved.")
                raise
            except Exception as exc:
                logger.error(f"[{method_name}] {task.task_id} crashed: {exc}", exc_info=True)
                placeholder = {
                    "task_id": task.task_id,
                    "solution": "",
                    "method": method_name,
                    "error": str(exc),
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "num_attempts": 0,
                    "num_retries": 0,
                    "wall_time": time.time() - t_start,
                    "attempt_details": [],
                }
                out_f.write(json.dumps(placeholder) + "\n")
                out_f.flush()
                ran += 1
                continue

            out_f.write(json.dumps(result.to_jsonl_dict()) + "\n")
            out_f.flush()
            ran += 1

            logger.info(
                f"[{method_name}] {task.task_id} done in {result.wall_time:.1f}s "
                f"| tokens={result.total_tokens} | attempts={result.num_attempts}"
            )

    logger.info(f"[{method_name}] Finished. Ran {ran} tasks → {solutions_path}")
    return ran


# ── Remote evaluation ─────────────────────────────────────────────────────────

def sanitize_and_evaluate(
    method_name: str,
    solutions_path: Path,
    subset: str,
    split: str,
    execution: str = "docker",
) -> Optional[Path]:
    """
    Run bigcodebench.sanitize then bigcodebench.evaluate.
    execution: "docker" (local, all packages pre-installed) or "gradio" (remote HF API).
    Returns path to eval_results.json, or None on failure.
    """
    python = sys.executable

    # ── Step 1: Sanitize ──────────────────────────────────────────────────────
    logger.info(f"[{method_name}] Sanitizing {solutions_path.name} ...")
    result = subprocess.run(
        [python, "-m", "bigcodebench.sanitize", "--samples", str(solutions_path), "--calibrate"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"[{method_name}] Sanitize failed:\n{result.stderr}")
        return None

    # bigcodebench.sanitize produces: <stem>-sanitized-calibrated.jsonl (all dashes)
    sanitized_path = solutions_path.parent / (
        solutions_path.stem + "-sanitized-calibrated.jsonl"
    )
    if not sanitized_path.exists():
        candidates = list(solutions_path.parent.glob(f"{method_name}*sanitized*.jsonl"))
        if not candidates:
            logger.error(f"[{method_name}] Could not find sanitized output.")
            return None
        sanitized_path = candidates[0]

    logger.info(f"[{method_name}] Sanitized → {sanitized_path.name}")

    # ── Step 2: Evaluate ──────────────────────────────────────────────────────
    if execution == "docker":
        evaluate_cmd = _docker_eval_cmd(sanitized_path, subset, split)
        timeout = 1800   # Docker local eval can take longer
        logger.info(
            f"[{method_name}] Running evaluation via Docker "
            f"(subset={subset}, split={split}) ..."
        )
    else:
        evaluate_cmd = [
            python, "-m", "bigcodebench.evaluate",
            "--execution", "gradio",
            "--split", split,
            "--subset", subset,
            "--samples", str(sanitized_path),
        ]
        timeout = 600
        logger.info(
            f"[{method_name}] Submitting to remote Gradio evaluator "
            f"(subset={subset}, split={split}) — takes 4-7 minutes ..."
        )

    result = subprocess.run(evaluate_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error(f"[{method_name}] Evaluate failed:\n{result.stderr}")
        return None

    candidates = list(sanitized_path.parent.glob(f"{sanitized_path.stem}*eval_results.json"))
    if not candidates:
        logger.error(f"[{method_name}] eval_results.json not found after evaluation.")
        return None

    eval_results_path = candidates[0]
    logger.info(f"[{method_name}] Official eval results → {eval_results_path.name}")
    return eval_results_path


def _docker_eval_cmd(sanitized_path: Path, subset: str, split: str) -> list:
    """
    Build the docker run command for local evaluation.
    Mounts the parent directory of the solutions file to /app inside the container.
    The container already has all BigCodeBench dependencies installed.
    """
    results_dir = sanitized_path.parent.resolve()
    return [
        "docker", "run", "--rm",
        "--platform", "linux/amd64",
        "-v", f"{results_dir}:/app",
        "bigcodebench/bigcodebench-evaluate:latest",
        "--execution", "local",
        "--split", split,
        "--subset", subset,
        "--samples", sanitized_path.name,   # relative to /app inside the container
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _setup_logging(LOG_LEVEL, LOG_FILE)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tasks ────────────────────────────────────────────────────────────
    logger.info(
        f"Loading BigCodeBench ({args.subset}) — max_tasks={args.max_tasks}, "
        f"use_instruct={args.use_instruct}"
    )
    tasks = BigCodeBenchLoader.load(
        subset=args.subset,
        max_tasks=args.max_tasks,
        use_instruct=args.use_instruct,
        task_ids=args.task_ids,
    )
    logger.info(f"Loaded {len(tasks)} tasks.")

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

    # ── BCB evaluation server (optional) ──────────────────────────────────────
    bcb_evaluator = None
    if args.bcb_server:
        bcb_evaluator = BCBEvaluator(base_url=args.bcb_server)
        if bcb_evaluator.health():
            logger.info(f"BCB server connected at {args.bcb_server} — using real test suite for retries.")
        else:
            logger.warning(f"BCB server at {args.bcb_server} is unreachable — falling back to quick_check.")
            bcb_evaluator = None

    logger.info(f"LLM: provider={args.provider}, model={args.model}")

    # ── Select methods ────────────────────────────────────────────────────────
    if "all" in args.method:
        methods_to_run = list(ALL_BASELINES.items())
    else:
        methods_to_run = [(m, ALL_BASELINES[m]) for m in args.method]

    # ── Phase 1: Generate solutions ───────────────────────────────────────────
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
            bcb_evaluator=bcb_evaluator,
        )

    logger.info(f"\nGeneration done in {time.time() - total_start:.1f}s.")

    tracker.save(str(tracker_path))
    logger.info(f"Token events saved to {tracker_path}")

    latency_path = results_dir / "latency_log.jsonl"
    with open(latency_path, "w") as f:
        for entry in llm.latency_log:
            f.write(json.dumps(entry) + "\n")
    print(f"Latency log saved: {len(llm.latency_log)} entries → {latency_path}")

    # ── Phase 2: Remote evaluation (optional) ─────────────────────────────────
    # split name BigCodeBench uses: "instruct" when use_instruct=True, else "complete"
    split = "instruct" if args.use_instruct else "complete"
    eval_status = {}

    if args.remote_eval:
        logger.info(f"\n{'='*60}\nPhase 2: Remote evaluation\n{'='*60}")
        for method_name, _ in methods_to_run:
            solutions_path = results_dir / f"{method_name}_solutions.jsonl"
            eval_json = sanitize_and_evaluate(
                method_name=method_name,
                solutions_path=solutions_path,
                subset=args.subset,
                split=split,
                execution=args.execution,
            )
            if eval_json:
                method_status = load_eval_results(str(eval_json))
                eval_status.update(method_status)
                logger.info(
                    f"[{method_name}] Loaded {len(method_status)} official results."
                )
    else:
        logger.info(
            "Skipping remote evaluation (pass --remote_eval to get official pass@1). "
            "Using quick_check proxy for correctness."
        )

    # ── Phase 3: Metrics ──────────────────────────────────────────────────────
    logger.info("Computing metrics ...")
    results = load_results(str(results_dir))
    metrics = compute_metrics(results, eval_status=eval_status if eval_status else None)
    print_comparison_table(metrics)

    metrics_path = results_dir / "metrics.json"
    save_metrics(metrics, str(metrics_path))
    logger.info(f"Metrics saved to {metrics_path}")

    if not args.remote_eval:
        print(
            f"\nNote: pass@1 above uses quick_check proxy (syntax + import check), "
            f"not BigCodeBench's official test suite.\n"
            f"Re-run with --remote_eval for official scores."
        )


if __name__ == "__main__":
    main()
