"""
Self-Planning baseline for StateGen.

Implements the two-phase self-planning approach described in:
  Jiang et al., "Self-planning Code Generation with Large Language Models",
  ACM TOSEM 2024.

Phase 1 — Planning:  Ask the LLM to produce a numbered step-by-step plan for the
                      problem (no code).
Phase 2 — Implementation: Give the LLM the problem + plan and ask it to implement
                           the solution.
"""

import re

from src.llm_backend import generate

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PLANNING_SYSTEM = (
    "You are an expert Python programmer. Given a coding problem, produce a clear, "
    "numbered step-by-step plan to solve it. Do not write any code — only describe "
    "the logical steps."
)

_PLANNING_USER = (
    "Problem:\n{problem}\n\n"
    "Write a numbered plan to solve this problem."
)

_IMPL_SYSTEM = (
    "You are an expert Python programmer. You will be given a problem and a plan. "
    "Implement the solution in Python strictly following the plan."
)

_IMPL_USER = (
    "Problem:\n{problem}\n\n"
    "Plan:\n{plan}\n\n"
    "Now implement the solution in Python. Return only the code block."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Strip markdown code fences and return the raw code string."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def run_self_planning(
    problem: str,
    backend: str = "together",
    temperature: float = 0.2,
    max_tokens: int = 2048,
    verbose: bool = False,
) -> dict:
    """
    Run the two-phase self-planning pipeline on a coding problem.

    Args:
        problem:     Natural-language description of the coding task.
        backend:     LLM provider — "together" (default) or "thetaedge".
        temperature: Sampling temperature passed to both LLM calls.
        max_tokens:  Token budget per call.
        verbose:     If True, print the plan and generated code to stdout.

    Returns:
        A dict with keys:
            "plan"              — raw plan text from phase 1.
            "code"              — extracted code string from phase 2.
            "total_tokens_used" — always -1 (token counts unavailable via generate()).
            "backend"           — the provider used.
    """
    # ------------------------------------------------------------------
    # Phase 1: Planning
    # ------------------------------------------------------------------
    plan: str = generate(
        prompt=_PLANNING_USER.format(problem=problem),
        system_prompt=_PLANNING_SYSTEM,
        temperature=temperature,
        max_tokens=max_tokens,
        backend=backend,
    )

    if verbose:
        print("=== Plan ===")
        print(plan)
        print()

    # ------------------------------------------------------------------
    # Phase 2: Implementation
    # ------------------------------------------------------------------
    impl_response: str = generate(
        prompt=_IMPL_USER.format(problem=problem, plan=plan),
        system_prompt=_IMPL_SYSTEM,
        temperature=temperature,
        max_tokens=max_tokens,
        backend=backend,
    )

    code: str = _extract_code(impl_response)

    if verbose:
        print("=== Generated Code ===")
        print(code)
        print()

    return {
        "plan": plan,
        "code": code,
        "total_tokens_used": -1,  # generate() returns content only; no usage metadata
        "backend": backend,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _PROBLEM = (
        "Given a DataFrame df with columns 'age' and 'salary', "
        "return the mean salary of people older than 30."
    )

    print("=== self_planning.py smoke test ===\n")
    result = run_self_planning(problem=_PROBLEM, verbose=True)

    print("--- Final result dict ---")
    print(f"backend           : {result['backend']}")
    print(f"total_tokens_used : {result['total_tokens_used']}")
    print(f"plan (first 120 chars): {result['plan'][:120]}...")
    print(f"code:\n{result['code']}")
