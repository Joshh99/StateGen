#!/usr/bin/env python3
"""
Dry-run sanity check: 2 tasks × 4 methods × 6 models before Azure VM deployment.

Usage:
    .venv/bin/python experiments/sanity_new_models.py

For each model it runs direct_gen / self_planning / self_debugging / stategen on
the first 2 BigCodeBench tasks and checks:
  - API connectivity (non-zero output)
  - No empty solutions (thinking-model token budget issue)
  - No crashes mid-run

Results land in results/sanity_<model_name>/ and are NOT overwritten on re-run
unless you delete the directory.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

MODELS = [
    dict(
        name="deepseek_r1",
        model="deepseek-ai/DeepSeek-R1",
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        max_tokens=16384,  # reasoning model
    ),
    dict(
        name="deepseek_v3_1",
        model="deepseek-ai/DeepSeek-V3.1",
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        max_tokens=4096,
    ),
    dict(
        name="llama3_3_70b",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        max_tokens=4096,
    ),
    dict(
        name="llama3_8b_lite",
        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
        base_url="https://api.together.xyz/v1",
        api_key=os.getenv("TOGETHER_API_KEY", ""),
        max_tokens=4096,
    ),
    dict(
        name="gemini_flash",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=os.getenv("GEMINI_API_KEY", ""),
        max_tokens=16384,  # flash is also a thinking model
    ),
    dict(
        name="gpt4o_azure",
        model="gpt-4o-stategen",          # Azure deployment name
        base_url="https://stategen.openai.azure.com/",
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        max_tokens=4096,
        provider="azure_openai",
        api_version="2024-12-01-preview",
    ),
    dict(
        name="phi4_reasoning",
        model="Phi-4-reasoning",
        base_url="https://coder1.openai.azure.com/openai/v1/",
        api_key=os.getenv("AZURE_CODER1_API_KEY", ""),
        max_tokens=16384,  # reasoning model
    ),
    dict(
        name="kimi_k2_5",
        model="Kimi-K2.5",
        base_url="https://coder1.openai.azure.com/openai/v1/",
        api_key=os.getenv("AZURE_CODER1_API_KEY", ""),
        max_tokens=4096,
    ),
]

METHODS = ["direct_gen", "self_planning", "self_debugging", "stategen"]
N_TASKS  = 2
PYTHON   = str(ROOT / ".venv/bin/python")
RUNNER   = str(ROOT / "experiments/run_bigcodebench.py")


def check_results(results_dir: Path, method: str) -> dict:
    """Parse a JSONL results file and return a status dict."""
    f = results_dir / f"{method}_solutions.jsonl"
    if not f.exists():
        return dict(status="NO_FILE", non_empty=0, total=0, quick_pass=0, avg_out_tokens=0)

    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    total      = len(rows)
    non_empty  = sum(1 for r in rows if r.get("solution", "").strip())
    quick_pass = sum(
        1 for r in rows
        if r.get("attempt_details") and r["attempt_details"][0].get("passed", False)
    )
    avg_out = sum(r.get("total_output_tokens", 0) for r in rows) / max(total, 1)

    if total == 0:
        status = "NO_OUTPUT"
    elif non_empty == 0:
        status = "EMPTY"       # thinking-model token budget too low
    elif non_empty < total:
        status = "PARTIAL"
    else:
        status = "OK"

    return dict(
        status=status, non_empty=non_empty, total=total,
        quick_pass=quick_pass, avg_out_tokens=round(avg_out),
    )


def run_model(cfg: dict) -> dict:
    experiment   = f"sanity_{cfg['name']}"
    results_dir  = ROOT / "results" / experiment

    # Fresh directory so the runner never skips tasks from a previous partial run
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    env = os.environ.copy()
    env["DEEPSEEK_API_KEY"] = cfg["api_key"]   # LLMAgent key lookup for openai_compatible
    env["MAX_NEW_TOKENS"]   = str(cfg["max_tokens"])
    env["MPLBACKEND"]       = "Agg"

    print(f"\n{'='*70}")
    print(f"  MODEL : {cfg['name']}  ({cfg['model']})")
    print(f"  URL   : {cfg['base_url']}")
    print(f"  TOKENS: {cfg['max_tokens']}")
    print(f"{'='*70}")

    method_results = {}

    for method in METHODS:
        print(f"\n  [{method}] running {N_TASKS} tasks ...", flush=True)

        provider = cfg.get("provider", "openai_compatible")
        cmd = [
            PYTHON, RUNNER,
            "--method",     method,
            "--max_tasks",  str(N_TASKS),
            "--experiment", experiment,
            "--provider",   provider,
            "--model",      cfg["model"],
            "--base_url",   cfg["base_url"],
            "--api_key",    cfg["api_key"],
        ]
        if "api_version" in cfg:
            cmd += ["--api_version", cfg["api_version"]]

        proc = subprocess.run(cmd, env=env, cwd=ROOT)
        r    = check_results(results_dir, method)

        flag = ""
        if r["status"] != "OK":
            flag = f"  <-- {r['status']}"
            if r["status"] == "EMPTY":
                flag += " (raise MAX_NEW_TOKENS?)"

        print(f"  [{method}] non_empty={r['non_empty']}/{r['total']}  "
              f"quick_pass={r['quick_pass']}/{r['total']}  "
              f"avg_out_tokens={r['avg_out_tokens']}{flag}")

        method_results[method] = r

    return dict(name=cfg["name"], model=cfg["model"], methods=method_results)


def print_summary(all_results: list):
    print(f"\n\n{'='*75}")
    print("DRY-RUN SUMMARY  (2 tasks × 4 methods × 6 models)")
    print(f"{'='*75}")

    header = f"{'Model':<18}" + "".join(f"  {m[:14]:<14}" for m in METHODS)
    print(header)
    print("-" * 75)

    all_ok = True
    for res in all_results:
        row = f"{res['name']:<18}"
        for method in METHODS:
            r = res["methods"][method]
            cell = f"{r['quick_pass']}/{r['total']} {r['status'][:2]}"  # e.g. "2/2 OK"
            row += f"  {cell:<14}"
        print(row)
        if any(r["status"] != "OK" for r in res["methods"].values()):
            all_ok = False

    print()
    print("Legend: quick_pass / total  status")
    print("        OK=all non-empty  EMPTY=thinking bug  PARTIAL=some empty  NO_FILE=crash")
    print()
    if all_ok:
        print("All models × methods passed. Safe to deploy VMs.")
    else:
        print("Issues found (see above). Fix before deploying.")
        sys.exit(1)


def main():
    all_results = []
    for cfg in MODELS:
        r = run_model(cfg)
        all_results.append(r)

    print_summary(all_results)


if __name__ == "__main__":
    main()
