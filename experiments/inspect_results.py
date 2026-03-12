#!/usr/bin/env python3
# experiments/inspect_results.py
# Human-readable inspection of experiment results.
# Shows: problem prompt, generated code, BigCodeBench verdict, token usage.
#
# Usage:
#   python experiments/inspect_results.py                          # summary table
#   python experiments/inspect_results.py --task_id BigCodeBench/13
#   python experiments/inspect_results.py --method self_debugging --failed
#   python experiments/inspect_results.py --method all --passed

import argparse
import json
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
                import os; os.environ.setdefault(k.strip(), v.strip())

RESULTS_DIR = ROOT / "results"

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_solutions(results_dir: Path, methods=None) -> dict:
    """Returns {task_id: {method: record}} from *_solutions.jsonl files."""
    data = {}
    for path in sorted(results_dir.glob("*_solutions.jsonl")):
        method = path.stem.replace("_solutions", "")
        if methods and method not in methods:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tid = rec["task_id"]
                data.setdefault(tid, {})[method] = rec
    return data


def load_eval_results(results_dir: Path) -> dict:
    """Returns {task_id: {method: 'pass'|'fail'}} from *eval_results.json files."""
    verdicts = {}
    for path in results_dir.glob("*eval_results.json"):
        method = path.name.split("_solutions")[0]
        with open(path) as f:
            data = json.load(f)
        for task_id, results in data.get("eval", {}).items():
            if results:
                status = results[0]["status"]
                verdicts.setdefault(task_id, {})[method] = status
    return verdicts


def load_problems() -> dict:
    """Load BigCodeBench problem prompts. Returns {task_id: {prompt, test}}."""
    try:
        import sys
        sys.path.insert(0, str(ROOT / "bigcodebench"))
        from bigcodebench.data import get_bigcodebench
        raw = get_bigcodebench(subset="hard")
        return {
            tid: {
                "prompt":   item["instruct_prompt"],
                "test":     item["test"],
                "entry_point": item["entry_point"],
            }
            for tid, item in raw.items()
        }
    except Exception as e:
        print(yellow(f"Warning: could not load BigCodeBench problems ({e})"))
        return {}


# ── Display helpers ───────────────────────────────────────────────────────────

def separator(char="═", width=80):
    print(char * width)


def print_task(task_id, problems, solutions, verdicts, methods, show_code=True):
    separator()
    prob = problems.get(task_id, {})

    # Header
    print(bold(f"  {task_id}  —  {prob.get('entry_point', '')}"))
    separator("─")

    # Problem prompt
    prompt = prob.get("prompt", dim("(prompt not loaded)"))
    print(bold("PROBLEM:"))
    for line in prompt.strip().splitlines():
        print(f"  {line}")
    print()

    # Per-method results
    task_solutions = solutions.get(task_id, {})
    task_verdicts  = verdicts.get(task_id, {})

    for method in methods:
        rec = task_solutions.get(method)
        if not rec:
            continue

        verdict = task_verdicts.get(method)
        if verdict == "pass":
            status_str = green("✓ PASS")
        elif verdict == "fail":
            status_str = red("✗ FAIL")
        else:
            # Fall back to quick_check from attempt_details
            details = rec.get("attempt_details", [])
            last_ok = details[-1].get("passed_quick_check", False) if details else False
            status_str = yellow(f"{'✓' if last_ok else '✗'} quick_check only")

        in_tok  = rec.get("total_input_tokens", 0)
        out_tok = rec.get("total_output_tokens", 0)
        total   = rec.get("total_tokens", in_tok + out_tok)
        attempts = rec.get("num_attempts", 1)
        retries  = rec.get("num_retries", 0)
        wall     = rec.get("wall_time", 0.0)

        print(bold(f"[{method}]") + f"  {status_str}")
        print(f"  tokens : {total:,}  (in={in_tok:,} / out={out_tok:,})")
        print(f"  calls  : {attempts} total, {retries} retries")
        print(f"  time   : {wall:.1f}s")

        # Attempt trace
        details = rec.get("attempt_details", [])
        if len(details) > 1:
            print(f"  trace  :", end="")
            for d in details:
                ok = d.get("passed", d.get("passed_quick_check", False))
                err = d.get("error") or ""
                marker = green("✓") if ok else red("✗")
                print(f" {marker}", end="")
                if err:
                    short_err = err.split("\n")[-1][:50]
                    print(f"({dim(short_err)})", end="")
            print()

        # Generated code
        if show_code:
            code = rec.get("solution", "").strip()
            print(f"\n  {cyan('── generated code ──')}")
            for line in code.splitlines():
                print(f"  {line}")
        print()


def print_summary_table(solutions, verdicts, methods):
    """Print a compact per-task summary across all methods."""
    all_task_ids = sorted(set(
        tid for method_data in solutions.values() for tid in [method_data]
        # restructure: solutions is {task_id: {method: rec}}
    ) | set(solutions.keys()))

    # Header
    method_w = 14
    header = f"{'Task ID':<25}"
    for m in methods:
        header += f"{m[:method_w]:>{method_w}}"
    header += f"{'tokens':>10}"
    separator()
    print(bold(header))
    separator("─")

    for tid in all_task_ids:
        row = f"{tid:<25}"
        tok_parts = []
        for m in methods:
            rec = solutions.get(tid, {}).get(m)
            if not rec:
                row += f"{'—':>{method_w}}"
                continue
            verdict = verdicts.get(tid, {}).get(m)
            if verdict == "pass":
                cell = green("pass")
            elif verdict == "fail":
                cell = red("fail")
            else:
                details = rec.get("attempt_details", [])
                ok = details[-1].get("passed_quick_check", False) if details else False
                cell = yellow("✓qc" if ok else "✗qc")
            row += f"{cell:>{method_w + 9}}"   # +9 for ANSI codes
            tok_parts.append(rec.get("total_tokens", 0))

        avg_tok = int(sum(tok_parts) / len(tok_parts)) if tok_parts else 0
        row += f"{avg_tok:>10,}"
        print(row)

    separator()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect BigCodeBench experiment results."
    )
    p.add_argument("--task_id", default=None,
                   help="Show details for one task, e.g. BigCodeBench/13")
    p.add_argument("--method", nargs="+",
                   default=["direct_gen", "self_debugging", "self_planning"],
                   help="Methods to show (default: all three baselines)")
    p.add_argument("--failed", action="store_true",
                   help="Show only tasks where at least one method failed")
    p.add_argument("--passed", action="store_true",
                   help="Show only tasks where all methods passed")
    p.add_argument("--no_code", action="store_true",
                   help="Hide generated code (show metadata only)")
    p.add_argument("--results_dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    methods = args.method
    show_code = not args.no_code

    print(bold("\nLoading results..."))
    solutions = load_solutions(results_dir, methods)
    verdicts  = load_eval_results(results_dir)
    problems  = load_problems()

    if not solutions:
        print(red("No solution files found in " + str(results_dir)))
        return

    # ── Single task detail view ───────────────────────────────────────────────
    if args.task_id:
        print_task(args.task_id, problems, solutions, verdicts, methods, show_code)
        return

    # ── Filter tasks ──────────────────────────────────────────────────────────
    task_ids = sorted(solutions.keys())

    if args.failed:
        def any_failed(tid):
            for m in methods:
                v = verdicts.get(tid, {}).get(m)
                if v == "fail":
                    return True
                # fall back to quick_check
                rec = solutions.get(tid, {}).get(m)
                if rec:
                    d = rec.get("attempt_details", [])
                    if d and not d[-1].get("passed_quick_check", True):
                        return True
            return False
        task_ids = [t for t in task_ids if any_failed(t)]

    if args.passed:
        def all_passed(tid):
            for m in methods:
                v = verdicts.get(tid, {}).get(m)
                if v and v != "pass":
                    return False
            return True
        task_ids = [t for t in task_ids if all_passed(t)]

    # ── Summary table (default) ───────────────────────────────────────────────
    if not show_code or (not args.failed and not args.passed and not args.task_id):
        print_summary_table(solutions, verdicts, methods)
        print(f"\n{len(task_ids)} tasks  |  methods: {', '.join(methods)}")
        print(dim("Re-run with --task_id BigCodeBench/13 to see full detail for one task."))
        print(dim("Add --failed or --passed to filter. Add --no_code to hide code.\n"))

    # ── Detail view for filtered set ──────────────────────────────────────────
    if args.failed or args.passed:
        for tid in task_ids:
            print_task(tid, problems, solutions, verdicts, methods, show_code)
        print(f"\n{len(task_ids)} tasks shown.")


if __name__ == "__main__":
    main()
