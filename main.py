import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/stategen.log"),
    ],
)
logger = logging.getLogger(__name__)

from agents.llm_agent       import LLMAgent
from agents.execution_engine import ExecutionEngine
from agents.memory_manager  import MemoryManager
from agents.controller import StateController


def run_single(problem: str, test_cases: list[dict]) -> None:
    """Run StateGen on one problem and print the result."""

    print("\n" + "=" * 60)
    print("StateGen  —  Single Problem Test")
    print("=" * 60)

    # ── Initialise agents ─────────────────────────────────────
    print("\n[1/4] Loading LLM agent...")
    llm = LLMAgent()

    print("[2/4] Setting up execution engine...")
    executor = ExecutionEngine()

    print("[3/4] Loading memory manager...")
    memory = MemoryManager()

    print("[4/4] Creating state controller...")
    controller = StateController(llm, executor, memory)

    # ── Solve ─────────────────────────────────────────────────
    print(f"\nProblem:\n{problem}\n")
    result = controller.solve(problem, test_cases)

    # ── Print result ──────────────────────────────────────────
    print("\n" + "─" * 60)
    if result.success:
        print("✓  SOLVED")
        print("─" * 60)
        print(result.code)
        print("─" * 60)
    else:
        print(f"✗  FAILED at state {result.failed_at_state}")

    print(f"\nStats:")
    print(f"  States completed : {result.states_completed}")
    print(f"  Total retries    : {result.total_retries}")
    print(f"  Patterns reused  : {result.patterns_reused}")

    # ── Save result ───────────────────────────────────────────
    Path("results").mkdir(exist_ok=True)
    with open("results/last_run.json", "w") as f:
        json.dump(
            {
                "success": result.success,
                "code": result.code,
                "stats": {
                    "states_completed": result.states_completed,
                    "total_retries": result.total_retries,
                    "patterns_reused": result.patterns_reused,
                },
            },
            f,
            indent=2,
        )
    print("\nResult saved to results/last_run.json")


if __name__ == "__main__":
    # ── Sample problem (from DS-1000 style) ──────────────────
    problem = """
    Given a pandas DataFrame `df` with columns 'age' and 'salary',
    filter the rows where age > 25, then calculate and return
    the mean salary of the filtered rows.
    """

    test_cases = [
        {
            "setup": """
import pandas as pd
df = pd.DataFrame({'age': [20, 30, 25, 35], 'salary': [1000, 2000, 1500, 2500]})
""",
            "assertion": "assert abs(result - 2250.0) < 0.01",
        }
    ]

    run_single(problem, test_cases)
