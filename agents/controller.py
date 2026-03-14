import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from config import MAX_RETRIES
from core.models import State, ProgramContext, SolveResult
from agents.llm_agent import LLMAgent
from agents.execution_engine import ExecutionEngine
from agents.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class StateController:
    """
    Orchestrates the full StateGen pipeline:

      Problem → Decompose → [For each state: Memory? → Generate → Verify] → Solution

    The feedback loop lives inside `_process_state`:
      - If verification FAILS  → extract error → retry ONLY this state
      - If verification PASSES → store pattern → advance to next state
    """

    def __init__(
        self,
        llm: LLMAgent,
        executor: ExecutionEngine,
        memory: MemoryManager,
    ):
        self.llm      = llm
        self.executor = executor
        self.memory   = memory

    # ─────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────

    def solve(self, problem: str, test_cases: List[dict]) -> SolveResult:
        """
        Main entry point.

        Args:
            problem:    Natural language problem description
            test_cases: List of {"setup": ..., "assertion": ...} dicts

        Returns:
            SolveResult with final code and statistics
        """
        logger.info("=" * 60)
        logger.info(f"Solving: {problem[:80]}...")

        # ── Step 1: Decompose problem into states ──────────────
        states = self._decompose(problem)
        logger.info(f"Decomposed into {len(states)} states")
        for s in states:
            logger.info(f"  {s}")

        # ── Step 2: Iterate through states ─────────────────────
        context = ProgramContext()
        total_retries   = 0
        patterns_reused = 0

        for state in states:
            logger.info(f"\nProcessing {state}")
            success, retries, reused = self._process_state(
                state, context, test_cases
            )
            total_retries   += retries
            patterns_reused += reused

            if not success:
                logger.error(f"Failed to complete {state} after {MAX_RETRIES} attempts")
                return SolveResult(
                    success=False,
                    failed_at_state=state.index,
                    total_retries=total_retries,
                    patterns_reused=patterns_reused,
                    states_completed=state.index,
                )

        # ── Step 3: Return completed solution ──────────────────
        logger.info("\n✓ All states complete!")
        return SolveResult(
            success=True,
            code=context.full_code(),
            total_retries=total_retries,
            patterns_reused=patterns_reused,
            states_completed=len(states),
        )

    # ─────────────────────────────────────────
    # CORE FEEDBACK LOOP
    # ─────────────────────────────────────────

    def _process_state(
        self,
        state: State,
        context: ProgramContext,
        test_cases: List[dict],
    ) -> Tuple[bool, int, int]:
        """
        Process one state with retry logic.

        Returns:
            (success, retries_used, patterns_reused)

        This is where local backtracking happens:
          FAIL → retry only this state (context is NOT rolled back)
        """
        retries_used    = 0
        patterns_reused = 0
        last_error      = None

        for attempt in range(MAX_RETRIES):
            logger.info(f"  Attempt {attempt + 1}/{MAX_RETRIES}")

            # ── 1. Check memory first ──────────────────────────
            code = self.memory.lookup(state, context)

            if code:
                logger.info("  → Using cached pattern")
                patterns_reused += 1
            else:
                # ── 2. Generate via LLM ────────────────────────
                logger.info("  → Generating via LLM...")
                code = self.llm.generate_for_state(state, context, last_error)

            # ── 3. Verify ──────────────────────────────────────
            full_code = context.full_code() + "\n" + code if not context.is_empty() else code
            result    = self.executor.verify(
                full_code,
                test_cases,
                postconditions=state.postconditions,
            )

            if result.success:
                # ── 4a. SUCCESS: store pattern, update context ─
                logger.info(f"  ✓ Verified in attempt {attempt + 1}")
                self.memory.store(state, context, code)
                context.add(code)
                return True, retries_used, patterns_reused

            else:
                # ── 4b. FAIL: save error, retry ONLY this state ─
                last_error   = result.error
                retries_used += 1
                logger.warning(f"  ✗ Failed: {last_error}")
                logger.info("  → Local backtracking: retrying only this state")

        # All attempts exhausted
        return False, retries_used, patterns_reused

    # ─────────────────────────────────────────
    # DECOMPOSITION
    # ─────────────────────────────────────────

    def _decompose(self, problem: str) -> List[State]:
        """
        Ask the LLM to break the problem into sequential states,
        then parse the structured output.
        """
        raw    = self.llm.generate_decomposition(problem)
        states = self._parse_states(raw)

        # Fallback: if parsing fails, create a single state
        if not states:
            logger.warning("State parsing failed — using single-state fallback")
            states = [
                State(
                    index=0,
                    description=problem,
                    preconditions=[],
                    postconditions=[],
                )
            ]

        return states

    def _parse_states(self, raw: str) -> List[State]:
        """
        Parse the LLM's state decomposition output.

        Expected format:
            STATE 0
            Description: Import libraries
            Preconditions: none
            Postconditions: pandas imported as pd
            ---
        """
        states = []
        blocks = raw.split("---")

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            try:
                # Extract state index
                idx_match = re.search(r"STATE\s+(\d+)", block, re.IGNORECASE)
                if not idx_match:
                    continue
                idx = int(idx_match.group(1))

                # Extract fields
                desc = self._extract_field(block, "Description")
                pre  = self._extract_list(block, "Preconditions")
                post = self._extract_list(block, "Postconditions")

                if desc:
                    states.append(
                        State(
                            index=idx,
                            description=desc,
                            preconditions=pre,
                            postconditions=post,
                        )
                    )
            except Exception as e:
                logger.debug(f"Failed to parse state block: {e}")
                continue

        return sorted(states, key=lambda s: s.index)

    @staticmethod
    def _extract_field(text: str, label: str) -> str:
        match = re.search(rf"{label}:\s*(.+)", text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_list(text: str, label: str) -> List[str]:
        match = re.search(rf"{label}:\s*(.+)", text, re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1).strip()
        if raw.lower() in ("none", "n/a", "-"):
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]
