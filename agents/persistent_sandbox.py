# agents/persistent_sandbox.py
# Long-lived Python subprocess for fast intermediate state verification.
#
# Problem solved:
#   execution_engine.quick_run() spawns a fresh interpreter for every call.
#   For data-science tasks, that means re-importing pandas/numpy/torch on
#   every state check — typically 1-3 seconds per import per call.
#
# Solution:
#   Keep one Python process alive across all state checks for a task.
#   Each quick_run() call executes code in a FRESH NAMESPACE (no variable
#   leakage between calls), but the process's sys.modules cache persists,
#   so "import pandas" after the first call is a fast dict lookup.
#
# Safety:
#   Code runs inside exec() with a dedicated namespace dict — it cannot
#   affect the host process or bleed state between calls.
#   On timeout, the worker is killed and a new one is started.

import sys
import os
import json
import tempfile
import logging
import threading
from typing import Optional

from core.models import VerificationResult

logger = logging.getLogger(__name__)

# ── Worker script ──────────────────────────────────────────────────────────────
# Runs inside the long-lived subprocess.
# Reads JSON lines from stdin, executes each in a fresh namespace, writes JSON.

_WORKER = r"""
import sys, json, traceback, io
from contextlib import redirect_stdout, redirect_stderr

def run_one(code):
    ns = {}
    out = io.StringIO()
    err = io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(code, "<state>", "exec"), ns)
        return {"ok": True, "stdout": out.getvalue()}
    except Exception:
        tb = traceback.format_exc()
        lines = tb.strip().splitlines()
        return {
            "ok": False,
            "error": lines[-1] if lines else "Unknown error",
            "traceback": tb,
        }

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        msg  = json.loads(raw)
        result = run_one(msg["code"])
    except Exception:
        result = {"ok": False, "error": traceback.format_exc(), "traceback": ""}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
"""


class PersistentSandbox:
    """
    A long-lived Python subprocess for intermediate state checks.

    Usage (as context manager, recommended):
        with PersistentSandbox(timeout=10.0) as sandbox:
            result = sandbox.quick_run(assembled_code)

    Or plain usage with explicit close:
        sandbox = PersistentSandbox()
        result  = sandbox.quick_run(code)
        sandbox.close()

    Thread-safe: one request at a time (internal lock).
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self._proc   = None
        self._script = None
        self._lock   = threading.Lock()
        self._start()

    # ── Public API ─────────────────────────────────────────────────────────────

    def quick_run(self, code: str) -> VerificationResult:
        """
        Execute code in a fresh namespace inside the persistent worker.
        Heavy libraries stay cached in sys.modules between calls.
        """
        with self._lock:
            return self._send(code)

    def restart(self):
        """Restart the worker (clears the sys.modules cache too)."""
        with self._lock:
            self._stop()
            self._start()

    def close(self):
        with self._lock:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self._stop()
        except Exception:
            pass

    # ── Internals ───────────────────────────────────────────────────────────────

    def _start(self):
        import subprocess
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="sgworker_"
        )
        tmp.write(_WORKER)
        tmp.close()
        self._script = tmp.name

        self._proc = subprocess.Popen(
            [sys.executable, "-u", self._script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        logger.debug("PersistentSandbox started (pid=%d)", self._proc.pid)

    def _stop(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        if self._script and os.path.exists(self._script):
            try:
                os.unlink(self._script)
            except Exception:
                pass
            self._script = None

    def _send(self, code: str) -> VerificationResult:
        """Send code to the worker; wait up to self.timeout for a response."""
        if self._proc is None or self._proc.poll() is not None:
            logger.warning("PersistentSandbox: worker exited, restarting")
            self._stop()
            self._start()

        result_holder: list = [None]

        def _read():
            try:
                result_holder[0] = self._proc.stdout.readline()
            except Exception as exc:
                result_holder[0] = json.dumps({"ok": False, "error": str(exc)})

        try:
            self._proc.stdin.write(json.dumps({"code": code}) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            return VerificationResult(success=False, error=f"Write failed: {exc}")

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(self.timeout)

        if result_holder[0] is None:
            logger.warning("PersistentSandbox: timed out, restarting worker")
            self._stop()
            self._start()
            return VerificationResult(
                success=False,
                error=f"Timed out after {self.timeout}s (possible infinite loop)",
            )

        try:
            data = json.loads(result_holder[0])
        except Exception as exc:
            return VerificationResult(success=False, error=f"Response parse error: {exc}")

        if data.get("ok"):
            return VerificationResult(success=True, output=data.get("stdout", ""))

        return VerificationResult(
            success=False,
            error=data.get("error", "Unknown error"),
            traceback=data.get("traceback"),
        )
