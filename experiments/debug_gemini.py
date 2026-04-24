#!/usr/bin/env python3
"""
Diagnose why Gemini direct_gen shows 0% pass@1.
Calls the CMU gateway Gemini model with the exact direct_gen prompt
and prints the raw response + what extract_code() produces.

Usage:
    OPENAI_API_KEY=sk-zKfuYnpvkK_dWG2Iu1m3Dg python experiments/debug_gemini.py
"""

import os
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI
from baselines.base import extract_code
from data.loader import BigCodeBenchLoader

CMU_BASE_URL = "https://ai-gateway.andrew.cmu.edu/"
MODEL = "gemini-2.5-pro"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("CMU_API_KEY")

if not API_KEY:
    print("ERROR: set OPENAI_API_KEY=<cmu_key> before running.")
    sys.exit(1)

_SYSTEM = (
    "You are an expert Python programmer. "
    "Write clean, correct, well-structured Python code."
)

_PROMPT = """\
Write a complete Python implementation for the following problem.
Include all necessary imports. Return ONLY the code, no explanation.

Problem:
{problem}
"""

BOLD  = "\033[1m"; RESET = "\033[0m"
GREEN = "\033[92m"; RED   = "\033[91m"; CYAN  = "\033[96m"; DIM = "\033[2m"

def hr(title=""):
    print(f"\n{CYAN}{BOLD}{'─'*60}")
    if title:
        print(f"  {title}")
        print(f"{'─'*60}{RESET}")
    else:
        print(RESET, end="")


def main():
    tasks = BigCodeBenchLoader.load(subset="hard", max_tasks=5, use_instruct=True)
    task  = tasks[0]

    print(f"\nTask: {BOLD}{task.task_id}{RESET}")
    hr("PROBLEM (first 400 chars)")
    print(task.prompt[:400])

    client = OpenAI(api_key=API_KEY, base_url=CMU_BASE_URL)

    hr("CALLING GEMINI via CMU gateway ...")
    print(f"  model    : {MODEL}")
    print(f"  base_url : {CMU_BASE_URL}")

    prompt_text = _PROMPT.format(problem=task.prompt)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt_text},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
    except Exception as e:
        print(f"\n{RED}API call failed: {e}{RESET}")
        sys.exit(1)

    raw = response.choices[0].message.content or ""
    in_tok  = response.usage.prompt_tokens
    out_tok = response.usage.completion_tokens
    print(f"  tokens   : {in_tok} in / {out_tok} out")

    hr("RAW RESPONSE (full)")
    print(raw)

    hr("AFTER extract_code()")
    extracted = extract_code(raw)
    print(extracted if extracted else f"{RED}<empty string>{RESET}")

    hr("DIAGNOSIS")
    if not extracted.strip():
        print(f"{RED}extract_code() returned empty — this is why pass@1 = 0%{RESET}")
        print("\nRaw response starts with:")
        print(repr(raw[:200]))
        # Check what patterns are present
        has_python_fence = bool(re.search(r"```python", raw))
        has_plain_fence  = bool(re.search(r"```", raw))
        print(f"\n  has ```python fence : {has_python_fence}")
        print(f"  has ``` fence       : {has_plain_fence}")
        print(f"  raw length          : {len(raw)} chars")
    else:
        # Run a quick syntax check
        import py_compile, tempfile
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(extracted)
            fname = f.name
        try:
            py_compile.compile(fname, doraise=True)
            print(f"{GREEN}extract_code() produced valid Python — no parse issue.{RESET}")
            print("The 0% may come from runtime / import errors in the BCB test suite.")
        except py_compile.PyCompileError as e:
            print(f"{RED}Syntax error in extracted code: {e}{RESET}")
        finally:
            os.unlink(fname)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
