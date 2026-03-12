# baselines/self_debugging.py
# Self-Debugging baseline — faithful implementation of Chen et al. (2304.05128).
#
# Each debugging turn:
#   1. Execute code → get error message
#   2. Explain the code (rubber duck debugging)
#   3. Generate feedback message
#   4. Repair: regenerate the WHOLE program with explanation + error as context
#
# Key cost driver: every repair call re-feeds the complete program + explanation.
# On BigCodeBench (avg 1112-char prompts + growing explanation), this grows fast.

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

# ── Initial generation ────────────────────────────────────────────
_GENERATE_PROMPT = """\
Write a complete Python implementation for the following problem.
Include all necessary imports. Return ONLY the code, no explanation.

Problem:
{problem}
"""

# ── Explain the code (rubber duck debugging) ─────────────────────
# From the paper: "the model explains the code line by line"
_EXPLAIN_PROMPT = """\
Explain the following Python code line by line.
Describe what each part computes and what values variables hold.

Problem this code is trying to solve:
{problem}

Code:
```python
{code}
```

Execution result:
{execution_result}

Line-by-line explanation:"""

# ── Repair using explanation + error ─────────────────────────────
_REPAIR_PROMPT = """\
The Python code below is incorrect. Fix it based on the explanation and error.

Problem:
{problem}

Current code:
```python
{code}
```

Code explanation:
{explanation}

Error:
{error}

Write the corrected implementation. Include all imports. Return ONLY the code.

Fixed implementation:"""


class SelfDebuggingBaseline(BaseBaseline):
    """
    Self-Debugging (Chen et al., ICLR 2024 — Google DeepMind).

    Each debugging turn costs:
        1 explanation call  (input: problem + full_code + error)
        1 repair call       (input: problem + full_code + explanation + error)

    Token cost grows with each turn because the full code is re-fed every round.
    This is the core inefficiency that StateGen's local backtracking addresses.
    """

    METHOD_NAME = "self_debugging"

    def solve(self, task: Task) -> SolveResult:
        t0      = time.time()
        details = []

        # ── Turn 0: Initial generation ────────────────────────────
        raw = self.llm.generate(
            _GENERATE_PROMPT.format(problem=task.prompt),
            _SYSTEM,
            task_id=task.task_id,
            method=self.METHOD_NAME,
            call_type="generation",
            attempt=0,
        )
        code = extract_code(raw)
        ok, error, test_errors = self._evaluate(code, task.task_id)

        details.append({
            "turn":        0,
            "call_type":   "generation",
            "passed":      ok,
            "error":       error,
            "test_errors": test_errors,
        })

        logger.info(
            f"[{task.task_id}] self_debugging turn 0 (generate): "
            f"{'✓' if ok else '✗'}  {error[:80] if error else ''}"
        )

        # ── Debugging turns ───────────────────────────────────────
        for turn in range(1, MAX_RETRIES):
            if ok:
                break

            exec_result = f"Error: {error}" if error else "Code executed but tests failed"

            # Step 1: Explain the code
            explanation = self.llm.generate(
                _EXPLAIN_PROMPT.format(
                    problem=task.prompt,
                    code=code,
                    execution_result=exec_result,
                ),
                _SYSTEM,
                task_id=task.task_id,
                method=self.METHOD_NAME,
                call_type="explanation",
                attempt=turn,
            )

            # Step 2: Repair with explanation
            raw = self.llm.generate(
                _REPAIR_PROMPT.format(
                    problem=task.prompt,
                    code=code,
                    explanation=explanation,
                    error=error or "Tests did not pass",
                ),
                _SYSTEM,
                task_id=task.task_id,
                method=self.METHOD_NAME,
                call_type="repair",
                attempt=turn,
            )

            code = extract_code(raw)
            ok, error, test_errors = self._evaluate(code, task.task_id)

            details.append({
                "turn":        turn,
                "call_type":   "repair",
                "passed":      ok,
                "error":       error,
                "test_errors": test_errors,
            })

            logger.info(
                f"[{task.task_id}] self_debugging turn {turn} (repair): "
                f"{'✓' if ok else '✗'}  {error[:80] if error else ''}"
            )

        in_tok, out_tok = self._token_totals(task.task_id)

        # num_attempts = initial generation + (explain + repair) per debug turn
        repair_turns = len(details) - 1
        num_attempts = 1 + repair_turns * 2

        return SolveResult(
            task_id=task.task_id,
            method=self.METHOD_NAME,
            solution=code,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            num_attempts=num_attempts,
            num_retries=repair_turns,
            wall_time=time.time() - t0,
            attempt_details=details,
        )
