import time
import logging

from baselines.base import BaseBaseline, SolveResult, extract_code
from data.loader import Task

logger = logging.getLogger(__name__)

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


class DirectGenBaseline(BaseBaseline):
    """
    Baseline: single LLM call, take whatever comes back.
    No retries, no planning, no debugging.
    Establishes the token-cost floor and the accuracy ceiling for zero-effort generation.
    """

    METHOD_NAME = "direct_gen"

    def solve(self, task: Task) -> SolveResult:
        t0 = time.time()

        raw = self.llm.generate(
            _PROMPT.format(problem=task.prompt),
            _SYSTEM,
            task_id=task.task_id,
            method=self.METHOD_NAME,
            call_type="generation",
            attempt=0,
        )

        solution = extract_code(raw)
        ok, err, test_errors = self._evaluate(solution, task.task_id)
        in_tok, out_tok = self._token_totals(task.task_id)

        logger.info(
            f"[{task.task_id}] direct_gen: {'PASS' if ok else 'FAIL'}  "
            f"tokens={in_tok + out_tok}"
        )

        return SolveResult(
            task_id=task.task_id,
            method=self.METHOD_NAME,
            solution=solution,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            num_attempts=1,
            num_retries=0,
            wall_time=time.time() - t0,
            attempt_details=[{
                "attempt": 0,
                "passed":  ok,
                "error":   err,
                "test_errors": test_errors,
            }],
        )
