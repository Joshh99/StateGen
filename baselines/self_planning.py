import time
import logging

from baselines.base import BaseBaseline, SolveResult, extract_code
from data.loader import Task
from config import MAX_RETRIES

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert Python programmer. "
    "Write clean, correct, well-structured Python code."
)

# ── Phase 1: Planning ─────────────────────────────────────────────
_PLAN_PROMPT = """\
Provide a step-by-step plan to solve the following Python programming problem.

Problem:
{problem}

Instructions:
- Output a numbered list of steps (e.g. "1. Import ...", "2. Load ...")
- Each step should be a single, clear, high-level action
- Do NOT write any code
- Focus on the algorithmic approach, not syntax

Plan:"""

# ── Phase 2: Implementation ───────────────────────────────────────
_CODE_PROMPT = """\
Implement the following Python function by following the plan below step by step.

Problem:
{problem}

Plan:
{plan}

Instructions:
- Follow each step of the plan in order
- Include all necessary imports
- Return ONLY the code, no explanation

Implementation:"""

# ── Retry prompt (keeps the plan, just asks to fix the code) ──────
_RETRY_CODE_PROMPT = """\
Your previous implementation failed. Try again following the plan.

Problem:
{problem}

Plan:
{plan}

Error from previous attempt:
{error}

Write a corrected implementation. Include all imports. Return ONLY the code.

Implementation:"""


class SelfPlanningBaseline(BaseBaseline):
    """
    Self-Planning (Jiang et al., ACM TOSEM 2024).

    Token cost vs. Direct:
        +1 planning call per attempt (extra cost)
        But plans often lead to better first-attempt accuracy.

    Token cost vs. StateGen:
        On failure: regenerates the ENTIRE program (plan + all code).
        StateGen only regenerates the one failed state.
    """

    METHOD_NAME = "self_planning"

    def solve(self, task: Task) -> SolveResult:
        t0       = time.time()
        details  = []
        solution = ""
        plan     = ""

        for attempt in range(MAX_RETRIES):
            # ── Phase 1: Generate plan ─────────────────────────────
            plan = self.llm.generate(
                _PLAN_PROMPT.format(problem=task.prompt),
                _SYSTEM,
                task_id=task.task_id,
                method=self.METHOD_NAME,
                call_type="planning",
                attempt=attempt,
            )

            # ── Phase 2: Generate code from plan ──────────────────
            if attempt == 0 or not solution:
                code_prompt = _CODE_PROMPT.format(
                    problem=task.prompt,
                    plan=plan,
                )
            else:
                code_prompt = _RETRY_CODE_PROMPT.format(
                    problem=task.prompt,
                    plan=plan,
                    error=last_error,
                )

            raw = self.llm.generate(
                code_prompt,
                _SYSTEM,
                task_id=task.task_id,
                method=self.METHOD_NAME,
                call_type="generation",
                attempt=attempt,
            )

            solution = extract_code(raw)
            ok, last_error, test_errors = self._evaluate(solution, task.task_id)

            details.append({
                "attempt":     attempt,
                "passed":      ok,
                "error":       last_error,
                "test_errors": test_errors,
                "plan_length": len(plan.split("\n")),
            })

            logger.info(
                f"[{task.task_id}] self_planning attempt {attempt + 1}/{MAX_RETRIES}: "
                f"{'PASS' if ok else 'FAIL'}  {last_error[:80] if last_error else ''}"
            )

            if ok:
                break

        in_tok, out_tok = self._token_totals(task.task_id)
        total_calls     = len(details) * 2   # planning + generation per attempt

        return SolveResult(
            task_id=task.task_id,
            method=self.METHOD_NAME,
            solution=solution,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            num_attempts=total_calls,
            num_retries=total_calls - 2,     # first attempt = 2 calls (plan + code)
            wall_time=time.time() - t0,
            attempt_details=details,
        )
