# baselines/stategen_no_memory.py
# Ablation: StateGen with memory manager disabled.
#
# Every state is generated fresh from the LLM — no cached patterns are
# retrieved or stored.  Everything else (decomposition, local repair,
# postcondition checks, persistent sandbox) is identical to StateGenBaseline.
#
# Research question answered:
#   How much does pattern reuse contribute to pass@1 and token efficiency?
#   Compare stategen_no_memory vs stategen to isolate the memory component.

import logging
from baselines.stategen_baseline import StateGenBaseline
from agents.execution_engine import ExecutionEngine
from core.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class _NoOpMemory:
    """Drop-in replacement that always misses on lookup and ignores stores."""

    def lookup(self, state, context):
        return None

    def lookup_top_k(self, state, context, k=3):
        return []

    def store(self, state, context, code):
        pass

    def stats(self):
        return {"total_patterns": 0, "total_reuses": 0}


class StateGenNoMemoryBaseline(StateGenBaseline):
    """
    StateGen without memory.

    Identical to StateGenBaseline except the MemoryManager is replaced with
    a no-op stub.  Every LLM call starts from scratch — no cached patterns
    are retrieved or written.

    Ablation purpose:
        stategen_no_memory pass@1  vs  stategen pass@1
        ↑ this gap = memory contribution to accuracy

        stategen_no_memory avg_tokens  vs  stategen avg_tokens
        ↑ this gap = tokens saved by reusing verified patterns
    """

    METHOD_NAME = "stategen_no_memory"

    def __init__(self, llm, executor, tracker, bcb_evaluator=None):
        super().__init__(llm, executor, tracker, bcb_evaluator)
        # Replace real memory with a no-op stub
        self.memory = _NoOpMemory()
        logger.info("StateGenNoMemory: memory disabled (no-op stub installed)")
