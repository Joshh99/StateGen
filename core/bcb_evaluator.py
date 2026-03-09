# core/bcb_evaluator.py
# HTTP client for the BigCodeBench API server (bigcodebench-new/bcb_server).
# Replaces the offline Docker evaluation workflow with real-time per-task evaluation.
#
# Server endpoints used:
#   GET  /health      — confirm the server is up
#   POST /evaluate    — submit code, get pass/fail + per-test errors

import logging
from typing import Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


class BCBEvaluator:
    """
    Client for the BigCodeBench evaluation server.

    Usage:
        evaluator = BCBEvaluator("http://localhost:8000")
        passed, error, errors = evaluator.evaluate("BigCodeBench/13", code)

    Falls back gracefully if the server is unreachable.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def health(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def evaluate(
        self,
        task_id: str,
        code: str,
        min_time_limit: float = 10.0,
    ) -> Tuple[bool, str, Dict[str, str]]:
        """
        Submit code to the server and get the full test suite result.

        Returns:
            passed      — True if all tests passed
            error       — short human-readable error summary (empty string on pass)
            test_errors — {test_name: traceback} dict (empty on pass)
        """
        try:
            resp = requests.post(
                f"{self.base_url}/evaluate",
                json={
                    "task_id":        task_id,
                    "code":           code,
                    "min_time_limit": min_time_limit,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            passed      = data["passed"]
            test_errors = data.get("errors", {})
            status      = data.get("status", "fail")

            # Build a concise error string for the LLM retry prompt
            if test_errors:
                first_tb = next(iter(test_errors.values()))
                # Take the last meaningful line of the traceback
                error = first_tb.strip().splitlines()[-1][:300]
            elif not passed:
                error = f"status={status}"
            else:
                error = ""

            logger.debug(
                f"BCB evaluate {task_id}: passed={passed} "
                f"errors={len(test_errors)} status={status}"
            )
            return passed, error, test_errors

        except requests.exceptions.ConnectionError:
            logger.warning(
                f"BCBEvaluator: server unreachable at {self.base_url}. "
                f"Falling back to quick_check for {task_id}."
            )
            return None, "server_unreachable", {}

        except Exception as exc:
            logger.warning(f"BCBEvaluator: unexpected error for {task_id}: {exc}")
            return None, str(exc), {}
