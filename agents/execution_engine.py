# agents/execution_engine.py
# Safely runs generated code and tells us if it worked.
# Uses subprocess isolation — the code CANNOT affect this process.

import subprocess
import tempfile
import os
import time
import logging
from typing import List, Optional

from config import EXEC_TIMEOUT, EXEC_TRACEBACK_LINES
from core.models import VerificationResult

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Execute and verify Python code in a sandboxed subprocess.

    Usage:
        engine = ExecutionEngine()
        result = engine.verify(code, test_cases, postconditions)
    """

    def __init__(self, timeout: float = EXEC_TIMEOUT):
        self.timeout = timeout


    def verify(
        self,
        code: str,
        test_cases: List[dict],
        postconditions: Optional[List[str]] = None,
    ) -> VerificationResult:
        """
        Full verification pipeline:
          1. Syntax check
          2. Execute code
          3. Run test cases
          4. Check postconditions

        Args:
            code:             Python code to verify
            test_cases:       List of {"input": ..., "expected": ...} dicts
            postconditions:   Optional assertions to check after execution

        Returns:
            VerificationResult
        """
        # Step 1: syntax check (fast, no subprocess needed)
        syntax_result = self._check_syntax(code)
        if not syntax_result.success:
            return syntax_result

        # Step 2: execute the code itself
        exec_result = self._run_in_sandbox(code)
        if not exec_result.success:
            return exec_result

        # Step 3: run test cases
        for i, tc in enumerate(test_cases):
            tc_result = self._run_test_case(code, tc)
            if not tc_result.success:
                return VerificationResult(
                    success=False,
                    error=f"Test case {i+1} failed: {tc_result.error}",
                )

        # Step 4: check postconditions (optional)
        if postconditions:
            for condition in postconditions:
                pc_result = self._check_postcondition(code, condition)
                if not pc_result.success:
                    return VerificationResult(
                        success=False,
                        error=f"Postcondition not met: {condition}",
                    )

        return VerificationResult(success=True)

    def quick_run(self, code: str) -> VerificationResult:
        """
        Just run the code — no test cases or postconditions.
        Useful for checking that imports / variable definitions work.
        """
        syntax = self._check_syntax(code)
        if not syntax.success:
            return syntax
        return self._run_in_sandbox(code)

    # ─────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────

    def _check_syntax(self, code: str) -> VerificationResult:
        try:
            compile(code, "<string>", "exec")
            return VerificationResult(success=True)
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                error=f"SyntaxError at line {e.lineno}: {e.msg}",
            )

    def _run_in_sandbox(self, code: str) -> VerificationResult:
        """Write code to a temp file and run it in a subprocess."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(code)
            tmp = f.name

        try:
            start = time.time()
            result = subprocess.run(
                ["python", tmp],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = time.time() - start

            if result.returncode != 0:
                stderr = result.stderr.strip()
                lines  = stderr.splitlines()
                # Pass the last N lines to the LLM — enough for context without
                # overwhelming the prompt.  Full stderr is stored in .traceback.
                error_msg = "\n".join(lines[-EXEC_TRACEBACK_LINES:]) if lines else "Unknown error"
                return VerificationResult(
                    success=False,
                    error=error_msg,
                    traceback=stderr,
                    exec_time=elapsed,
                )

            return VerificationResult(
                success=True,
                output=result.stdout,
                exec_time=elapsed,
            )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                success=False,
                error=f"Timed out after {self.timeout}s (possible infinite loop)",
            )
        except Exception as e:
            return VerificationResult(success=False, error=str(e))
        finally:
            os.unlink(tmp)

    def _run_test_case(self, code: str, test_case: dict) -> VerificationResult:
        """
        Wrap code with a test harness and run it.

        test_case format: {"setup": "...", "assertion": "assert result == ..."}
        """
        setup     = test_case.get("setup", "")
        assertion = test_case.get("assertion", "")

        if not assertion:
            # No assertion provided — just check the code runs
            return self.quick_run(code)

        harness = f"""
{code}

# ── test harness ──
{setup}
try:
    {assertion}
    print("PASS")
except AssertionError as e:
    import sys
    print(f"FAIL: {{e}}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    import sys
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        return self._run_in_sandbox(harness)

    def check_postcondition(self, code: str, condition: str) -> VerificationResult:
        """
        Run a structural postcondition assertion after code executes.

        condition may or may not start with 'assert' — both forms are accepted:
          "callable(my_func)"
          "assert isinstance(result, (list, dict))"
          "'df' in dir()"

        Floating-point equality assertions (e.g. result == 3.14) will work but
        are discouraged; the decompose prompt instructs the LLM to write only
        structural checks (type, callability, existence).
        """
        cond = condition.strip()
        if not cond.startswith("assert "):
            cond = f"assert {cond}, 'postcondition failed: {condition}'"

        check_code = f"""\
{code}

# postcondition check
try:
    {cond}
    print("PASS")
except AssertionError as _e:
    import sys
    print(f"FAIL: {{_e}}", file=sys.stderr)
    sys.exit(1)
except Exception as _e:
    import sys
    print(f"ERROR: {{_e}}", file=sys.stderr)
    sys.exit(1)
"""
        return self._run_in_sandbox(check_code)

    # Keep old private name for backward compatibility
    _check_postcondition = check_postcondition
