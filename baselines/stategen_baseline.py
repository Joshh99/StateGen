# baselines/stategen_baseline.py
# StateGen MDP-based code generation — comparable baseline wrapper.
#
# Algorithm:
#   1. Decompose problem into N states via LLM (1 call).
#      The decomposition response also contains initial code for each state.
#   2. For each state:
#        a. Check memory for a cached pattern.
#        b. Use the initial code from decomposition (or memory) as attempt 0.
#        c. quick_run the assembled code so far.
#        d. If it passes → store in memory, advance.
#        e. If it fails → ask LLM to FIX only this state (local backtracking).
#   3. _evaluate() the fully assembled code for final correctness.
#
# Token efficiency vs self_debugging:
#   Repair prompts contain ONLY the failing state's context + error (few lines).
#   self_debugging re-feeds the ENTIRE program + explanation every repair turn.

import re
import time
import logging
from typing import List, Optional, Tuple, Dict

from baselines.base import BaseBaseline, SolveResult, extract_code
from data.loader import Task
from agents.memory_manager import MemoryManager
from core.models import State, ProgramContext
from config import MAX_RETRIES

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert Python programmer. "
    "Write clean, correct, well-structured Python code."
)

# ── Prompts ───────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
Break the following Python problem into 2–3 sequential implementation states.
For each state, provide the state metadata AND the Python code for that state.

Rules for states:
  - State 0: ONLY import statements and module-level constants (no defs, no logic).
  - State 1: The complete function definition(s) — from "def" to the last "return".
    Include ALL helper functions before the main function.
  - State 2 (optional): Only if a clearly separate second function is needed.

Each state's code, when appended to all previous states' code, must be runnable
Python that executes without crashing (function definitions are fine — they don't
need to be called).

Problem:
{problem}

Output format — repeat this block for each state:

STATE <n>
Description: <one line>
Preconditions: <comma-separated list of modules/variables already in scope, or "none">
Postconditions: <comma-separated list of what this state defines/produces>
CODE:
<python code for this state — correct indentation, complete, runnable>
END_STATE
"""

_REPAIR_PROMPT = """\
The Python code fragment below is incorrect. Fix it.

Problem:
{problem}

Code written in previous states (DO NOT rewrite or repeat this):
```python
{context_code}
```

Current state to fix (State {state_idx}: {state_desc}):
```python
{bad_code}
```

Error when this state's code is appended to the previous code:
{error}

Write the corrected code for State {state_idx} ONLY.
- Do NOT rewrite the previous states' code.
- Start from where the previous states' code left off.
- Use correct Python indentation.
- No markdown fences, no explanations.
"""

_FINAL_REPAIR_PROMPT = """\
The Python implementation below is incorrect. Fix it.

Problem:
{problem}

Current (broken) implementation:
```python
{full_code}
```

Test failure:
{error}

Write the corrected, complete implementation.
Include all necessary imports. Return ONLY the code, no explanation.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_decomposition(raw: str) -> List[Tuple[State, str]]:
    """
    Parse the LLM's decomposition output into (State, code_block) pairs.

    Expected format:
        STATE 0
        Description: ...
        Preconditions: ...
        Postconditions: ...
        CODE:
        <python code>
        END_STATE
    """
    results: List[Tuple[State, str]] = []

    # Split on END_STATE to get each state block
    blocks = re.split(r'\bEND_STATE\b', raw)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        idx_m = re.search(r'\bSTATE\s+(\d+)\b', block, re.IGNORECASE)
        if not idx_m:
            continue
        idx = int(idx_m.group(1))

        def _field(label: str) -> str:
            m = re.search(rf'^{label}:\s*(.+)', block, re.IGNORECASE | re.MULTILINE)
            return m.group(1).strip() if m else ""

        def _list_field(label: str) -> List[str]:
            raw_val = _field(label)
            if not raw_val or raw_val.lower() in ("none", "n/a", "-"):
                return []
            return [x.strip() for x in raw_val.split(",") if x.strip()]

        desc = _field("Description")
        if not desc:
            continue

        # Extract code between CODE: and END_STATE
        code_m = re.search(r'CODE:\s*\n(.*)', block, re.DOTALL | re.IGNORECASE)
        code = code_m.group(1).strip() if code_m else ""

        # Strip any accidental markdown fences inside the code block
        code = extract_code(code) if code.startswith("```") else code

        state = State(
            index=idx,
            description=desc,
            preconditions=_list_field("Preconditions"),
            postconditions=_list_field("Postconditions"),
        )
        results.append((state, code))

    results.sort(key=lambda x: x[0].index)
    return results


# ── Baseline class ────────────────────────────────────────────────────────────

class StateGenBaseline(BaseBaseline):
    """
    StateGen: MDP-based code generation with local backtracking.

    Single decomposition call generates state metadata + initial code for each
    state. Each state's code is verified with quick_run; failures trigger a
    targeted repair of ONLY that state (not the full program).

    Memory manager caches successful (state, context) → code patterns and
    reuses them across tasks in the same run.
    """

    METHOD_NAME = "stategen"

    def __init__(self, llm, executor, tracker, bcb_evaluator=None):
        super().__init__(llm, executor, tracker, bcb_evaluator)
        self.memory = MemoryManager()

    def solve(self, task: Task) -> SolveResult:
        t0 = time.time()
        details: List[Dict] = []

        # ── Step 1: Decompose (get states + initial code) ──────────────────
        state_code_pairs = self._decompose(task, details)
        logger.info(
            f"[{task.task_id}] stategen: decomposed into {len(state_code_pairs)} states"
        )

        # ── Step 2: Process each state ─────────────────────────────────────
        context = ProgramContext()
        total_retries = 0

        for state, initial_code in state_code_pairs:
            success, retries, fragment, state_details = self._process_state(
                state, initial_code, context, task
            )
            total_retries += retries
            details.extend(state_details)

            if success:
                context.add(fragment)
            else:
                logger.warning(
                    f"[{task.task_id}] stategen: failed at state {state.index}"
                )
                if fragment:
                    context.add(fragment)
                break

        # ── Step 3: Final evaluation + repair loop ─────────────────────────
        # Local backtracking caught syntax/runtime errors per-state.
        # Now check correctness (BCB tests). If it fails, do whole-solution
        # repairs (shorter than self_debugging — no explain step, no re-feed
        # of explanation; just full_code + error → repaired full_code).
        final_code = context.full_code()
        ok, error, test_errors = self._evaluate(final_code, task.task_id)

        details.append({
            "turn":        len(details),
            "call_type":   "final_eval",
            "passed":      ok,
            "error":       error,
            "test_errors": test_errors,
        })

        logger.info(
            f"[{task.task_id}] stategen final: {'PASS' if ok else 'FAIL'}  "
            f"retries={total_retries}  {error[:60] if error else ''}"
        )

        for repair_turn in range(1, MAX_RETRIES):
            if ok:
                break

            repaired = self.llm.generate(
                _FINAL_REPAIR_PROMPT.format(
                    problem=task.prompt,
                    full_code=final_code,
                    error=error or "Tests did not pass",
                ),
                _SYSTEM,
                task_id=task.task_id,
                method=self.METHOD_NAME,
                call_type="final_repair",
                attempt=repair_turn,
            )
            final_code = extract_code(repaired)
            ok, error, test_errors = self._evaluate(final_code, task.task_id)
            total_retries += 1

            details.append({
                "turn":        len(details),
                "call_type":   "final_repair",
                "passed":      ok,
                "error":       error,
                "test_errors": test_errors,
            })

            logger.info(
                f"[{task.task_id}] stategen final_repair {repair_turn}: "
                f"{'PASS' if ok else 'FAIL'}  {error[:60] if error else ''}"
            )

        in_tok, out_tok = self._token_totals(task.task_id)
        # Count actual LLM calls: 1 decomposition + per-state repairs + final repairs
        llm_calls = 1 + sum(
            1 for d in details if d.get("call_type") in ("repair", "final_repair")
        )

        return SolveResult(
            task_id=task.task_id,
            method=self.METHOD_NAME,
            solution=final_code,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            num_attempts=max(llm_calls, 1),
            num_retries=total_retries,
            wall_time=time.time() - t0,
            attempt_details=details,
        )

    # ── Decompose ──────────────────────────────────────────────────────────────

    def _decompose(
        self, task: Task, details: List[Dict]
    ) -> List[Tuple[State, str]]:
        raw = self.llm.generate(
            _DECOMPOSE_PROMPT.format(problem=task.prompt),
            _SYSTEM,
            task_id=task.task_id,
            method=self.METHOD_NAME,
            call_type="decomposition",
            attempt=0,
        )

        details.append({
            "turn":      0,
            "call_type": "decomposition",
            "raw":       raw[:300],
        })

        pairs = _parse_decomposition(raw)

        if not pairs:
            logger.warning(
                f"[{task.task_id}] Decomposition parsing failed — single-state fallback"
            )
            pairs = [(
                State(
                    index=0,
                    description="Implement the complete solution",
                    preconditions=[],
                    postconditions=["complete solution"],
                ),
                raw,  # use raw LLM output as-is
            )]

        return pairs

    # ── Process one state ──────────────────────────────────────────────────────

    def _process_state(
        self,
        state: State,
        initial_code: str,
        context: ProgramContext,
        task: Task,
    ) -> Tuple[bool, int, str, List[Dict]]:
        """
        Attempt 0: use initial_code (from decomposition) or memory.
        Attempts 1+: ask LLM to repair ONLY this state (local backtracking).

        Returns (success, retries_used, final_code_fragment, detail_records).
        """
        state_details: List[Dict] = []
        retries_used = 0
        last_code = initial_code

        for attempt in range(MAX_RETRIES):
            # Attempt 0: prefer memory, else initial decomp code
            if attempt == 0:
                cached = self.memory.lookup(state, context)
                if cached:
                    code = cached
                    from_memory = True
                    call_type = "memory_hit"
                else:
                    code = initial_code
                    from_memory = False
                    call_type = "decomp_verify"  # reusing code from decomposition
            else:
                # Repair: regenerate ONLY this state with the error
                code = self._repair_state(
                    state, last_code, last_error, context, task, attempt
                )
                from_memory = False
                call_type = "repair"

            code = extract_code(code) if code.strip().startswith("```") else code
            last_code = code

            # Assemble + quick_run
            assembled = (
                (context.full_code() + "\n" + code).strip()
                if not context.is_empty()
                else code
            )
            verify = self.executor.quick_run(assembled)

            state_details.append({
                "turn":        state.index * 10 + attempt,
                "call_type":   call_type,
                "state_idx":   state.index,
                "state_desc":  state.description,
                "attempt":     attempt,
                "from_memory": from_memory,
                "passed":      verify.success,
                "error":       verify.error or "",
            })

            if verify.success:
                if not from_memory:
                    self.memory.store(state, context, code)
                logger.info(
                    f"[{task.task_id}] state {state.index} attempt {attempt}: PASS"
                )
                return True, retries_used, code, state_details

            last_error = verify.error or "code ran but produced unexpected output"
            retries_used += 1
            logger.info(
                f"[{task.task_id}] state {state.index} attempt {attempt}: "
                f"FAIL  {last_error[:60]}"
            )

        return False, retries_used, last_code, state_details

    # ── Repair ────────────────────────────────────────────────────────────────

    def _repair_state(
        self,
        state: State,
        bad_code: str,
        error: str,
        context: ProgramContext,
        task: Task,
        attempt: int,
    ) -> str:
        ctx_code = context.full_code() if not context.is_empty() else "(none)"
        prompt = _REPAIR_PROMPT.format(
            problem=task.prompt,
            context_code=ctx_code,
            state_idx=state.index,
            state_desc=state.description,
            bad_code=bad_code,
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
