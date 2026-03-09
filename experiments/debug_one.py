#!/usr/bin/env python3
# experiments/debug_one.py
# Run a single task end-to-end and print every step.
# Useful for debugging why solutions are failing.
#
# Usage:
#   python experiments/debug_one.py
#   python experiments/debug_one.py --task_id BigCodeBench/13
#   python experiments/debug_one.py --task_id BigCodeBench/13 --method self_debugging
#   python experiments/debug_one.py --task_id BigCodeBench/13 --no_bcb   # skip BCB eval

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
_env = ROOT / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from agents.execution_engine import ExecutionEngine
from agents.llm_agent import LLMAgent
from baselines import ALL_BASELINES
from config import LLM_MODEL, LLM_PROVIDER, LLM_BASE_URL
from core.bcb_evaluator import BCBEvaluator
from core.token_tracker import TokenTracker
from data.loader import BigCodeBenchLoader

# ── ANSI ──────────────────────────────────────────────────────────────────────
BOLD  = "\033[1m";  RESET = "\033[0m"
GREEN = "\033[92m"; RED   = "\033[91m"; CYAN  = "\033[96m"; YELLOW = "\033[93m"; DIM = "\033[2m"

def section(title, colour=CYAN):
    print(f"\n{colour}{BOLD}{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}{RESET}")


def parse_args():
    p = argparse.ArgumentParser(description="Debug a single BigCodeBench task end-to-end.")
    p.add_argument("--task_id",  default=None,
                   help="Task to run, e.g. BigCodeBench/13 (default: first task in hard subset)")
    p.add_argument("--method",   default="direct_gen",
                   choices=list(ALL_BASELINES.keys()),
                   help="Baseline to use (default: direct_gen)")
    p.add_argument("--bcb_server", default="http://localhost:8000",
                   help="BCB evaluation server URL (default: http://localhost:8000)")
    p.add_argument("--no_bcb",   action="store_true",
                   help="Skip BCB server evaluation (use quick_check only)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load one task ─────────────────────────────────────────────────────────
    section("LOADING TASK", CYAN)
    tasks = BigCodeBenchLoader.load(subset="hard", max_tasks=50, use_instruct=True)

    if args.task_id:
        task = next((t for t in tasks if t.task_id == args.task_id), None)
        if task is None:
            print(f"{RED}Task '{args.task_id}' not found in hard subset.{RESET}")
            print("Available:", [t.task_id for t in tasks[:10]], "...")
            sys.exit(1)
    else:
        task = tasks[0]

    print(f"  task_id     : {BOLD}{task.task_id}{RESET}")
    print(f"  entry_point : {task.entry_point}")
    print(f"  method      : {args.method}")

    # ── Print problem ─────────────────────────────────────────────────────────
    section("PROBLEM PROMPT", CYAN)
    for line in task.prompt.strip().splitlines():
        print(f"  {line}")

    # ── Set up shared components ───────────────────────────────────────────────
    tracker = TokenTracker()
    llm     = LLMAgent(
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        tracker=tracker,
        base_url=LLM_BASE_URL,
    )
    executor = ExecutionEngine()

    # ── BCB server ────────────────────────────────────────────────────────────
    bcb_evaluator = None
    if not args.no_bcb:
        bcb_evaluator = BCBEvaluator(base_url=args.bcb_server)
        if bcb_evaluator.health():
            print(f"\n  {GREEN}BCB server connected at {args.bcb_server}{RESET}")
        else:
            print(f"\n  {YELLOW}BCB server not reachable — falling back to quick_check{RESET}")
            bcb_evaluator = None

    # ── Run baseline ──────────────────────────────────────────────────────────
    section("GENERATING SOLUTION", CYAN)
    baseline_cls = ALL_BASELINES[args.method]
    baseline     = baseline_cls(
        llm=llm,
        executor=executor,
        tracker=tracker,
        bcb_evaluator=bcb_evaluator,
    )

    print(f"  Calling {args.method}.solve() ...")
    result = baseline.solve(task)

    # ── Generated solution ────────────────────────────────────────────────────
    section("GENERATED SOLUTION", CYAN)
    for line in result.solution.strip().splitlines():
        print(f"  {line}")

    # ── Token usage ───────────────────────────────────────────────────────────
    section("TOKEN USAGE", CYAN)
    print(f"  input tokens  : {result.total_input_tokens:,}")
    print(f"  output tokens : {result.total_output_tokens:,}")
    print(f"  total tokens  : {result.total_tokens:,}")
    print(f"  LLM calls     : {result.num_attempts}")
    print(f"  retries       : {result.num_retries}")
    print(f"  wall time     : {result.wall_time:.1f}s")

    # ── Attempt trace ─────────────────────────────────────────────────────────
    if len(result.attempt_details) > 1:
        section("ATTEMPT TRACE", CYAN)
        for i, d in enumerate(result.attempt_details):
            ok  = d.get("passed", d.get("passed_quick_check", False))
            err = d.get("error", "")
            marker = f"{GREEN}✓ PASS{RESET}" if ok else f"{RED}✗ FAIL{RESET}"
            turn = d.get("turn", d.get("attempt", i))
            ct   = d.get("call_type", "")
            print(f"\n  Attempt {turn} [{ct}] → {marker}")
            if err:
                print(f"  {DIM}error: {err}{RESET}")

    # ── Final BCB evaluation ──────────────────────────────────────────────────
    section("FINAL EVALUATION", CYAN)

    if bcb_evaluator:
        print(f"  Calling BCB server /evaluate ...")
        passed, error, test_errors = bcb_evaluator.evaluate(task.task_id, result.solution)

        if passed:
            print(f"\n  Result: {GREEN}{BOLD}✓ PASS — all tests passed{RESET}")
        else:
            print(f"\n  Result: {RED}{BOLD}✗ FAIL{RESET}")
            if error:
                print(f"  Summary: {RED}{error}{RESET}")

            if test_errors:
                print(f"\n  {BOLD}Failed tests ({len(test_errors)}):{RESET}")
                for test_name, traceback in test_errors.items():
                    print(f"\n  {YELLOW}{test_name}{RESET}")
                    # Print last 10 lines of traceback
                    lines = traceback.strip().splitlines()
                    for line in lines[-10:]:
                        print(f"    {DIM}{line}{RESET}")
    else:
        # quick_check only
        print("  Running quick_check (no BCB server) ...")
        ok, err = baseline._quick_check(result.solution)
        if ok:
            print(f"\n  Result: {GREEN}✓ quick_check passed (code runs without crashing){RESET}")
            print(f"  {YELLOW}Note: this does NOT confirm the solution is correct.{RESET}")
        else:
            print(f"\n  Result: {RED}✗ quick_check failed{RESET}")
            print(f"  Error: {err}")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
