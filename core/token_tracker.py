# core/token_tracker.py
# Tracks every LLM call: input tokens, output tokens, call type.
# Shared across all baselines and StateGen so comparisons are fair.

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenEvent:
    task_id: str
    method: str          # "direct_gen" | "self_planning" | "self_debugging" | "stategen"
    call_type: str       # "generation" | "planning" | "explanation" | "repair" | "decomposition"
    input_tokens: int
    output_tokens: int
    attempt: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenTracker:
    """
    Records every LLM call for every task and method.

    Usage:
        tracker = TokenTracker()

        # Inside LLMAgent._call_api():
        tracker.record(task_id, method, call_type, input_tokens, output_tokens)

        # After a task completes:
        total = tracker.total_for_task(task_id)
        events = tracker.events_for_task(task_id)

        # After all tasks:
        tracker.save("results/token_events.jsonl")
        summary = tracker.summary_by_method()
    """

    def __init__(self):
        self._events: List[TokenEvent] = []

    def record(
        self,
        task_id: str,
        method: str,
        call_type: str,
        input_tokens: int,
        output_tokens: int,
        attempt: int = 0,
    ) -> None:
        event = TokenEvent(
            task_id=task_id,
            method=method,
            call_type=call_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attempt=attempt,
        )
        self._events.append(event)
        logger.debug(
            f"[token] {method}/{call_type} task={task_id} "
            f"in={input_tokens} out={output_tokens} total={event.total}"
        )

    # ─────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────

    def events_for_task(self, task_id: str) -> List[TokenEvent]:
        return [e for e in self._events if e.task_id == task_id]

    def total_for_task(self, task_id: str) -> int:
        return sum(e.total for e in self.events_for_task(task_id))

    def input_tokens_for_task(self, task_id: str) -> int:
        return sum(e.input_tokens for e in self.events_for_task(task_id))

    def output_tokens_for_task(self, task_id: str) -> int:
        return sum(e.output_tokens for e in self.events_for_task(task_id))

    def summary_by_method(self) -> Dict[str, Dict]:
        """Per-method aggregate token counts and call counts."""
        result: Dict[str, Dict] = {}
        for e in self._events:
            m = result.setdefault(e.method, {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "by_call_type": {},
            })
            m["input_tokens"]  += e.input_tokens
            m["output_tokens"] += e.output_tokens
            m["total_tokens"]  += e.total
            m["calls"]         += 1

            ct = m["by_call_type"].setdefault(e.call_type, {"input": 0, "output": 0, "calls": 0})
            ct["input"]  += e.input_tokens
            ct["output"] += e.output_tokens
            ct["calls"]  += 1

        return result

    def wasted_tokens(self, task_ids_failed: List[str]) -> Dict[str, int]:
        """
        Tokens spent on tasks that ultimately failed.
        Useful for measuring how much each method wastes on unsolvable/hard tasks.
        """
        failed = set(task_ids_failed)
        result: Dict[str, int] = {}
        for e in self._events:
            if e.task_id in failed:
                result[e.method] = result.get(e.method, 0) + e.total
        return result

    # ─────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event)) + "\n")
        logger.info(f"Saved {len(self._events)} token events to {path}")

    @classmethod
    def load(cls, path: str) -> "TokenTracker":
        tracker = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    # timestamp may not be in old files
                    data.setdefault("timestamp", 0.0)
                    tracker._events.append(TokenEvent(**data))
        logger.info(f"Loaded {len(tracker._events)} token events from {path}")
        return tracker
