# baselines/stategen_full_backtrack.py
# Ablation: StateGen with full-context (true MDP) backtracking.
#
# When state n fails, the LLM sees the FULL accumulated program (state 0
# through state n) as its repair context — not just state n in isolation.
# It returns corrected code for state n only, but informed by the complete
# history of what has been built so far.
#
# Research question answered:
#   Is it better to give the LLM minimal context (just the failing state)
#   or full context (everything accumulated so far) when repairing?
#
#   stategen (current):       repair context = state n only      (minimal)
#   stategen_full_backtrack:  repair context = state 0..n        (full MDP)
#
#   If full_backtrack > stategen:
#       More context helps — the LLM needs to see prior states to understand
#       WHY state n failed (e.g. wrong variable name from state 0).
#
#   If stategen > full_backtrack:
#       Minimal context is better — focusing the LLM on just the failing
#       fragment avoids distraction from irrelevant prior code.
#
# MDP framing:
#   In MDP terms, the "state" is the full accumulated program.
#   full_backtrack treats it correctly — the full history IS the MDP state.
#   stategen_current is a greedy approximation that ignores prior states
#   when repairing, trading context for token efficiency.

import logging
from typing import List, Optional

from baselines.stategen_baseline import StateGenBaseline, _SYSTEM
from baselines.base import extract_code

logger = logging.getLogger(__name__)


_FULL_BACKTRACK_REPAIR_PROMPT = """\
The Python code fragment below is failing. Fix it.

Problem:
{problem}

Full accumulated program up to and including the failing state:
```python
{full_assembled_code}
```

The part that needs fixing is State {state_idx} ({state_desc}).
It begins after the previous states' code ends.

Error produced when running the full assembled code:
{error}
{few_shot_section}
Return ONLY the corrected code for State {state_idx}.
- Do NOT rewrite or repeat previous states' code.
- You may use any variable or function defined in the previous states.
- No markdown fences, no explanations.
"""


class StateGenFullBacktrackBaseline(StateGenBaseline):
    """
    StateGen with full-context repair (true MDP backtracking).

    Identical to StateGenBaseline except the repair prompt for a failing
    state includes the complete accumulated program (states 0..n) rather
    than just state n in isolation.

    The LLM still returns code for the failing state only — but it can
    reason about all prior states when deciding what to fix.

    Ablation purpose:
        stategen_full_backtrack pass@1  vs  stategen pass@1
        ↑ positive gap = full context helps repair accuracy
        ↓ negative gap = minimal context (current StateGen) is better
    """

    METHOD_NAME = "stategen_full_backtrack"

    def _repair_state(
        self,
        state,
        bad_code: str,
        error: str,
        context,
        task,
        attempt: int,
        few_shot: Optional[List[str]] = None,
        downstream_hint: Optional[str] = None,
    ) -> str:
        """
        Full-context repair: assemble everything up to and including the
        failing state and send it all to the LLM.

        The LLM is instructed to return ONLY the corrected fragment for
        state n — but it can inspect all prior states to understand root cause.
        """
        # Assemble: prior states + failing state together
        prior_code = context.full_code() if not context.is_empty() else ""
        full_assembled = (
            (prior_code + "\n" + bad_code).strip() if prior_code else bad_code
        )

        from baselines.stategen_baseline import _fmt_few_shot
        few_shot_section = _fmt_few_shot(few_shot) if few_shot else ""

        prompt = _FULL_BACKTRACK_REPAIR_PROMPT.format(
            problem=task.prompt,
            full_assembled_code=full_assembled,
            state_idx=state.index,
            state_desc=state.description,
            error=error,
            few_shot_section=few_shot_section,
        )

        return self.llm.generate(
            prompt,
            _SYSTEM,
            task_id=task.task_id,
            method=self.METHOD_NAME,
            call_type="repair",
            attempt=attempt,
        )
