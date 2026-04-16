# baselines/stategen_baseline.py
# StateGen MDP-based code generation — comparable baseline wrapper.
#
# Algorithm:
#   1. Decompose problem into N states via LLM (1 call).
#      The decomposition response also contains initial code for each state.
#   2. For each state:
#        a. Check memory for top-k cached patterns (few-shot injection).
#        b. Use the initial code from decomposition (or top-1 memory hit) as attempt 0.
#        c. quick_run the assembled code via PersistentSandbox (no cold start).
#        d. Check structural postconditions (type/callability assertions).
#        e. Pass → store in memory, advance.
#        f. Fail → ask LLM to FIX only this state (local backtracking).
#        g. Budget exhausted → backtrack to previous state (1 level upstream).
#   3. _evaluate() the fully assembled code for final correctness.
#      On failure, up to MAX_RETRIES-1 whole-solution repairs.
#
# Gap closure vs. original:
#   Gap 1 (postconditions): decompose prompt requests executable structural
#          assertions; they are verified after each quick_run pass.
#   Gap 2 (traceback + upstream): full traceback passed to LLM; 1-level
#          upstream backtrack when a state exhausts its repair budget.
#   Gap 3 (memory top-k + library tags): lookup_top_k() returns top-k
#          patterns with library-tagged embeddings; injected as few-shot.
#   Gap 4 (sandbox): PersistentSandbox keeps one interpreter alive per
#          StateGenBaseline instance — no cold starts for pandas/numpy.
#   Gap 5 (hybrid fallback): already present — kept and unchanged.

import re
import time
import logging
from typing import List, Optional, Tuple, Dict

from baselines.base import BaseBaseline, SolveResult, extract_code
from data.loader import Task
from agents.memory_manager import MemoryManager
from agents.persistent_sandbox import PersistentSandbox
from core.models import State, ProgramContext
from config import MAX_RETRIES, EXEC_TIMEOUT, MEMORY_TOP_K

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
Postconditions: <semicolon-separated Python assertion expressions that verify this \
state's structural output — check callability, type, or variable existence ONLY; \
do NOT assert exact values or floating-point equality>
  Good examples: callable(my_func); 'pd' in dir(); isinstance(result, (list, dict))
  Bad examples:  result == 3.14; df.shape == (100, 5)
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
{hint_section}
{few_shot_section}\
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

    Postconditions are semicolon-separated executable Python assertions
    (updated from the old comma-separated natural-language descriptions).
    """
    results: List[Tuple[State, str]] = []
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

        def _semi_list_field(label: str) -> List[str]:
            """Semicolon-separated list (used for postconditions)."""
            raw_val = _field(label)
            if not raw_val or raw_val.lower() in ("none", "n/a", "-"):
                return []
            return [x.strip() for x in raw_val.split(";") if x.strip()]

        desc = _field("Description")
        if not desc:
            continue

        code_m = re.search(r'CODE:\s*\n(.*)', block, re.DOTALL | re.IGNORECASE)
        code = code_m.group(1).strip() if code_m else ""
        code = extract_code(code) if code.startswith("```") else code

        state = State(
            index=idx,
            description=desc,
            preconditions=_list_field("Preconditions"),
            postconditions=_semi_list_field("Postconditions"),   # executable assertions
        )
        results.append((state, code))

    results.sort(key=lambda x: x[0].index)
    return results


def _fmt_few_shot(patterns: List[str]) -> str:
    """Format top-k memory patterns as few-shot examples for the repair prompt."""
    if not patterns:
        return ""
    lines = ["Similar verified patterns for reference (adapt as needed, do not copy verbatim):"]
    for i, code in enumerate(patterns, 1):
        lines.append(f"\nExample {i}:\n```python\n{code}\n```")
    lines.append("")
    return "\n".join(lines) + "\n"


# ── Baseline class ────────────────────────────────────────────────────────────

class StateGenBaseline(BaseBaseline):
    """
    StateGen: MDP-based code generation with local backtracking.

    Key behaviours (all gaps addressed):
      - PersistentSandbox for intermediate checks (no cold starts)
      - Full traceback in repair prompts (ExecutionEngine already handles this)
      - 1-level upstream backtracking when a state exhausts its budget
      - top-k memory retrieval with library-tagged embeddings → few-shot injection
      - Structural postcondition assertions verified after each state passes
    """

    METHOD_NAME = "stategen"

    def __init__(self, llm, executor, tracker, bcb_evaluator=None):
        super().__init__(llm, executor, tracker, bcb_evaluator)
        self.memory  = MemoryManager()
        # Persistent sandbox: double the per-call timeout to accommodate the
        # first import (cold start is paid once per PersistentSandbox lifetime).
        self.sandbox = PersistentSandbox(timeout=EXEC_TIMEOUT * 2)

    def __del__(self):
        try:
            self.sandbox.close()
        except Exception:
            pass

    # ── Main solve loop ────────────────────────────────────────────────────────

    def solve(self, task: Task) -> SolveResult:
        t0      = time.time()
        details: List[Dict] = []

        # Step 1: Decompose (1 LLM call — gets states + initial code)
        state_code_pairs = self._decompose(task, details)
        logger.info(
            f"[{task.task_id}] stategen: decomposed into {len(state_code_pairs)} states"
        )

        # Step 2: Process each state with local backtracking + 1-level upstream
        context       = ProgramContext()
        accepted      : List[Tuple[State, str]] = []   # (state, accepted_code) pairs
        total_retries = 0

        for i, (state, initial_code) in enumerate(state_code_pairs):
            success, retries, fragment, state_details, last_err = self._process_state(
                state, initial_code, context, task
            )
            details.extend(state_details)
            total_retries += retries

            if success:
                context.add(fragment)
                accepted.append((state, fragment))

            else:
                # ── Upstream backtracking (Gap 2b) ─────────────────────────
                # If there is a previous accepted state, try repairing it with
                # a hint that its output was insufficient for the downstream state.
                if i > 0 and accepted:
                    prev_state, prev_fragment = accepted[-1]
                    logger.warning(
                        f"[{task.task_id}] state {state.index} exhausted budget — "
                        f"backtracking to state {prev_state.index}"
                    )

                    # Rebuild context without the previous state
                    upstream_ctx = ProgramContext()
                    for _, blk in accepted[:-1]:
                        upstream_ctx.add(blk)

                    hint = (
                        f"The next state (State {state.index}: '{state.description}') "
                        f"failed with: {last_err}. Ensure this state's output "
                        f"satisfies the downstream state's preconditions."
                    )
                    p_ok, p_ret, p_frag, p_det, _ = self._process_state(
                        prev_state, prev_fragment, upstream_ctx, task,
                        downstream_hint=hint,
                    )
                    details.extend(p_det)
                    total_retries += p_ret

                    if p_ok:
                        accepted[-1] = (prev_state, p_frag)
                        # Rebuild context with repaired previous state
                        context = ProgramContext()
                        for _, blk in accepted:
                            context.add(blk)

                        # Retry current state once with the repaired context
                        s_ok, s_ret, s_frag, s_det, s_err = self._process_state(
                            state, initial_code, context, task
                        )
                        details.extend(s_det)
                        total_retries += s_ret

                        if s_ok:
                            context.add(s_frag)
                            accepted.append((state, s_frag))
                            continue
                        last_err = s_err  # noqa: F841  (used in log below)

                logger.warning(
                    f"[{task.task_id}] stategen: failed at state {state.index} "
                    f"(upstream backtrack exhausted)"
                )
                if fragment:
                    context.add(fragment)
                break

        # Step 3: Final evaluation + whole-solution repair loop (Gap 5 — hybrid fallback)
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
            f"[{task.task_id}] stategen final: {'✓' if ok else '✗'}  "
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
                f"{'✓' if ok else '✗'}  {error[:60] if error else ''}"
            )

        in_tok, out_tok = self._token_totals(task.task_id)
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
                    postconditions=[],
                ),
                raw,
            )]

        return pairs

    # ── Process one state ──────────────────────────────────────────────────────

    def _process_state(
        self,
        state: State,
        initial_code: str,
        context: ProgramContext,
        task: Task,
        downstream_hint: Optional[str] = None,
    ) -> Tuple[bool, int, str, List[Dict], Optional[str]]:
        """
        Attempt 0:  use initial_code (from decomposition) or top-1 memory hit.
        Attempts 1+: ask LLM to repair ONLY this state (local backtracking).
                     top-k memory patterns are injected as few-shot examples.

        Returns (success, retries_used, final_code_fragment, detail_records, last_error).
        """
        state_details : List[Dict] = []
        retries_used  = 0
        last_code     = initial_code
        last_error    : Optional[str] = None

        # Retrieve top-k patterns once (used for few-shot on repair attempts)
        top_k = self.memory.lookup_top_k(state, context, k=MEMORY_TOP_K)

        for attempt in range(MAX_RETRIES):
            if attempt == 0:
                # Prefer top-1 memory hit; otherwise use decomposition code
                if top_k:
                    code       = top_k[0]
                    from_memory = True
                    call_type   = "memory_hit"
                else:
                    code       = initial_code
                    from_memory = False
                    call_type   = "decomp_verify"
            else:
                # Repair: regenerate ONLY this state (local backtracking)
                # Inject top-k[1:] as few-shot (top-1 was already tried and failed)
                few_shot_for_repair = top_k[1:] if len(top_k) > 1 else []
                code = self._repair_state(
                    state, last_code, last_error or "unknown error",
                    context, task, attempt,
                    few_shot=few_shot_for_repair,
                    downstream_hint=downstream_hint,
                )
                from_memory = False
                call_type   = "repair"

            code      = extract_code(code) if code.strip().startswith("```") else code
            last_code = code

            # Assemble + quick_run via PersistentSandbox (Gap 4)
            assembled = (
                (context.full_code() + "\n" + code).strip()
                if not context.is_empty()
                else code
            )
            verify = self.sandbox.quick_run(assembled)

            # If execution passed, also check structural postconditions (Gap 1)
            post_error: Optional[str] = None
            if verify.success and state.postconditions:
                post_error = self._check_postconditions(state, assembled)

            passed = verify.success and post_error is None
            err_msg = (
                post_error if post_error
                else (verify.error or "code ran but produced unexpected output")
            ) if not passed else None

            state_details.append({
                "turn":        state.index * 10 + attempt,
                "call_type":   call_type,
                "state_idx":   state.index,
                "state_desc":  state.description,
                "attempt":     attempt,
                "from_memory": from_memory,
                "passed":      passed,
                "error":       err_msg or "",
                "post_error":  post_error or "",
            })

            if passed:
                if not from_memory:
                    self.memory.store(state, context, code)
                logger.info(
                    f"[{task.task_id}] state {state.index} attempt {attempt}: ✓"
                )
                return True, retries_used, code, state_details, None

            last_error = err_msg
            retries_used += 1
            logger.info(
                f"[{task.task_id}] state {state.index} attempt {attempt}: "
                f"✗  {(last_error or '')[:80]}"
            )

        return False, retries_used, last_code, state_details, last_error

    # ── Structural postcondition checker (Gap 1) ───────────────────────────────

    def _check_postconditions(
        self, state: State, assembled_code: str
    ) -> Optional[str]:
        """
        Run each postcondition assertion against the assembled code.
        Returns the first failure description, or None if all pass.

        Postconditions are structural (callable/isinstance/existence), never
        exact-value equality — the decompose prompt enforces this.
        """
        for condition in state.postconditions:
            condition = condition.strip()
            if not condition:
                continue
            result = self.executor.check_postcondition(assembled_code, condition)
            if not result.success:
                return f"{condition!r} failed: {result.error}"
        return None

    # ── Repair ────────────────────────────────────────────────────────────────

    def _repair_state(
        self,
        state: State,
        bad_code: str,
        error: str,
        context: ProgramContext,
        task: Task,
        attempt: int,
        few_shot: Optional[List[str]] = None,
        downstream_hint: Optional[str] = None,
    ) -> str:
        """
        Ask the LLM to fix ONLY this state.

        few_shot:        top-k memory patterns injected as in-context examples.
        downstream_hint: upstream-backtrack context (why the next state failed).
        error:           full traceback from ExecutionEngine (Gap 2a).
        """
        ctx_code = context.full_code() if not context.is_empty() else "(none)"

        hint_section = (
            f"\nUpstream context: {downstream_hint}\n" if downstream_hint else ""
        )
        few_shot_section = _fmt_few_shot(few_shot) if few_shot else ""

        prompt = _REPAIR_PROMPT.format(
            problem=task.prompt,
            context_code=ctx_code,
            state_idx=state.index,
            state_desc=state.description,
            bad_code=bad_code,
            error=error,
            hint_section=hint_section,
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
