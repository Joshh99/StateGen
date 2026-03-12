# baselines/base.py
# Abstract base class that every baseline (and the StateGen wrapper) must satisfy.
# Guarantees: same LLM, same executor, same token tracker → fair comparison.

import re
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agents.execution_engine import ExecutionEngine
from core.token_tracker import TokenTracker
from data.loader import Task

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """
    Unified output from any baseline or StateGen.
    Written to JSONL after each task so the run is crash-safe.
    """
    task_id: str
    method: str
    solution: str               # final generated code (may still fail eval)

    # Token accounting
    total_input_tokens: int
    total_output_tokens: int

    # Attempt accounting
    num_attempts: int           # total LLM calls made
    num_retries: int            # LLM calls after the first one

    # Timing
    wall_time: float

    # Per-attempt trace (for post-hoc analysis)
    attempt_details: List[Dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_jsonl_dict(self) -> dict:
        """BigCodeBench-compatible output with our extra fields."""
        return {
            "task_id":              self.task_id,
            "solution":             self.solution,
            "method":               self.method,
            "total_input_tokens":   self.total_input_tokens,
            "total_output_tokens":  self.total_output_tokens,
            "total_tokens":         self.total_tokens,
            "num_attempts":         self.num_attempts,
            "num_retries":          self.num_retries,
            "wall_time":            self.wall_time,
            "attempt_details":      self.attempt_details,
        }


class BaseBaseline(ABC):
    """
    Contract for all baselines and StateGen.

    When bcb_evaluator is provided, _evaluate() calls the BigCodeBench API
    server for real test results during generation. Otherwise falls back to
    the local quick_check (syntax + import check only).
    """

    METHOD_NAME = "base"

    def __init__(
        self,
        llm,
        executor: ExecutionEngine,
        tracker: TokenTracker,
        bcb_evaluator=None,         # core.bcb_evaluator.BCBEvaluator (optional)
    ):
        self.llm           = llm
        self.executor      = executor
        self.tracker       = tracker
        self.bcb_evaluator = bcb_evaluator

    @abstractmethod
    def solve(self, task: Task) -> SolveResult:
        """
        Attempt to solve one task.
        - Must call self.llm.generate() with task_id/method/call_type kwargs.
        - Must NOT read task.canonical_solution.
        - Returns SolveResult with all fields populated.
        """
        ...

    # ─────────────────────────────────────────
    # SHARED HELPERS
    # ─────────────────────────────────────────

    def _evaluate(self, code: str, task_id: str) -> Tuple[bool, str, Dict]:
        """
        Evaluate code correctness.

        When BCBEvaluator is available: calls the BigCodeBench API server.
          → runs the full unittest suite, returns real pass/fail + per-test errors.

        Fallback (no server): runs code locally via ExecutionEngine.quick_run().
          → only checks that code executes without crashing (syntax + import check).

        Returns:
            passed      — True if code is correct
            error       — short error summary string (empty on pass)
            test_errors — {test_name: traceback} dict (empty without BCB server)
        """
        if self.bcb_evaluator is not None:
            passed, error, test_errors = self.bcb_evaluator.evaluate(task_id, code)
            if passed is not None:          # None means server was unreachable
                return passed, error, test_errors
            # Fall through to quick_check on server error

        result = self.executor.quick_run(code)
        return result.success, (result.error or ""), {}

    def _quick_check(self, code: str) -> Tuple[bool, str]:
        """Legacy helper — kept for compatibility. Prefer _evaluate()."""
        result = self.executor.quick_run(code)
        return result.success, (result.error or "")

    def _token_totals(self, task_id: str) -> tuple:
        """Return (input_tokens, output_tokens) for this task and method only."""
        events = [
            e for e in self.tracker.events_for_task(task_id)
            if e.method == self.METHOD_NAME
        ]
        return (
            sum(e.input_tokens  for e in events),
            sum(e.output_tokens for e in events),
        )


# ─────────────────────────────────────────────────────────────────
# SHARED UTILITY
# ─────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """
    Strip markdown fences from LLM output and return clean Python code.
    Handles ```python ... ```, ``` ... ```, and raw code.
    """
    text = text.strip()

    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text
