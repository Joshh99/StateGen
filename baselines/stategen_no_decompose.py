# baselines/stategen_no_decompose.py
# Ablation: StateGen without problem decomposition.
#
# Instead of asking the LLM to split the problem into N states, we generate
# the complete solution in one shot (single state).  The local repair loop
# still runs on failures, but repairs the whole program each time.
#
# Research question answered:
#   Does structured decomposition into states improve pass@1 over
#   single-shot generation with repair?
#   Compare stategen_no_decompose vs stategen to isolate decomposition.
#
# Relationship to other baselines:
#   self_planning:        textual plan → full code → global retry
#   stategen_no_decompose: no plan    → full code → local repair loop
#   stategen:             executable decomposition → per-state repair
#
# The gap self_planning → stategen_no_decompose shows whether a repair loop
# beats global retry.  The gap stategen_no_decompose → stategen shows whether
# decomposition beats single-shot generation.

import logging
import time
from typing import List, Dict, Tuple

from baselines.stategen_baseline import StateGenBaseline, _SYSTEM
from baselines.base import SolveResult, extract_code
from data.loader import Task
from core.models import State, ProgramContext
from config import MAX_RETRIES

logger = logging.getLogger(__name__)


_SINGLE_SHOT_PROMPT = """\
Write a complete Python implementation for the following problem.
Include all necessary imports. Return ONLY the code, no explanation.

Problem:
{problem}
"""

_SINGLE_REPAIR_PROMPT = """\
The Python implementation below is incorrect. Fix it.

Problem:
{problem}

Current (broken) implementation:
```python
{code}
```

Error:
{error}

Write the corrected complete implementation.
Include all necessary imports. Return ONLY the code, no explanation.
"""


class StateGenNoDecomposeBaseline(StateGenBaseline):
    """
    StateGen without decomposition — single state covering the whole solution.

    The full solution is generated in one LLM call.  On failure, the whole
    program is repaired (up to MAX_RETRIES attempts).  Memory and the
    persistent sandbox are still active.

    Ablation purpose:
        stategen_no_decompose pass@1  vs  stategen pass@1
        ↑ this gap = decomposition contribution to accuracy
    """

    METHOD_NAME = "stategen_no_decompose"

    def _decompose(self, task: Task, details: List[Dict]) -> List[Tuple[State, str]]:
        """
        Override: generate the complete solution as a single state.
        No structured decomposition is performed.
        """
        raw = self.llm.generate(
            _SINGLE_SHOT_PROMPT.format(problem=task.prompt),
            _SYSTEM,
            task_id=task.task_id,
            method=self.METHOD_NAME,
            call_type="generation",
            attempt=0,
        )

        details.append({
            "turn":      0,
            "call_type": "generation",
            "raw":       raw[:300],
        })

        code = extract_code(raw)

        # Wrap the whole solution in a single state
        single_state = State(
            index=0,
            description="Complete solution",
            preconditions=[],
            postconditions=[],
        )
        return [(single_state, code)]

    def _repair_state(self, state, bad_code, error, context, task, attempt,
                      few_shot=None, downstream_hint=None) -> str:
        """
        Override: repair the WHOLE program (there is only one state anyway).
        Uses a simpler whole-program repair prompt — no state-boundary context.
        """
        prompt = _SINGLE_REPAIR_PROMPT.format(
            problem=task.prompt,
            code=bad_code,
            error=error,
        )
        return self.llm.generate(
            prompt,
            _SYSTEM,
            task_id=task.task_id,
            method=self.METHOD_NAME,
            call_type="repair",
            attempt=attempt,
        )
