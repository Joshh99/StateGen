import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

# Make the cloned bigcodebench package importable
sys.path.insert(0, str(Path(__file__).parent.parent / "bigcodebench"))

from config import BIGCODE_SUBSET, DS1000_SUBSET

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """
    A single benchmark problem. Same structure for BigCodeBench and DS-1000.

    Fields:
        task_id:            Unique identifier  e.g. "BigCodeBench/42", "DS1000/Pandas/3"
        prompt:             What the model sees (instruct or complete prompt)
        test_code:          Used by the evaluator to check correctness
        entry_point:        Function name the model must implement
        dataset:            "bigcodebench" | "ds1000"
        code_prompt:        Function signature skeleton (BigCodeBench only)
        canonical_solution: Ground truth — NEVER passed to the model
        metadata:           Extra fields (library, subset, prompt_type, etc.)
    """
    task_id: str
    prompt: str
    test_code: str
    entry_point: str
    dataset: str
    code_prompt: str = ""
    canonical_solution: str = ""
    metadata: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# BIGCODEBENCH
# ─────────────────────────────────────────────────────────────────

class BigCodeBenchLoader:
    """
    Loads tasks from the cloned BigCodeBench repo.

    Args:
        subset:      "full" (1140 tasks) or "hard" (subset)
        max_tasks:   Cap the number of tasks returned (for quick dev runs)
        use_instruct: True  → instruct_prompt  (~663 chars avg, NL-oriented)
                      False → complete_prompt  (~1112 chars avg, full docstring)
        task_ids:    Load only these specific IDs (for selective evaluation)
    """

    @staticmethod
    def load(
        subset: str = "full",
        max_tasks: Optional[int] = None,
        use_instruct: bool = True,
        task_ids: Optional[List[str]] = None,
    ) -> List[Task]:
        from bigcodebench.data import get_bigcodebench

        logger.info(f"Loading BigCodeBench (subset={subset}, instruct={use_instruct})")
        raw = get_bigcodebench(subset=subset)

        if task_ids:
            raw = {k: v for k, v in raw.items() if k in task_ids}
            logger.info(f"Filtered to {len(raw)} specified task IDs")

        tasks = []
        for task_id, item in raw.items():
            prompt = (
                item["instruct_prompt"]
                if use_instruct
                else item["complete_prompt"]
            )
            tasks.append(Task(
                task_id=task_id,
                prompt=prompt,
                test_code=item["test"],
                entry_point=item["entry_point"],
                dataset="bigcodebench",
                code_prompt=item.get("code_prompt", ""),
                canonical_solution=item["canonical_solution"],
                metadata={
                    "subset": subset,
                    "prompt_type": "instruct" if use_instruct else "complete",
                },
            ))

        if max_tasks:
            tasks = tasks[:max_tasks]

        logger.info(f"Loaded {len(tasks)} BigCodeBench tasks")
        return tasks


# ─────────────────────────────────────────────────────────────────
# DS-1000
# ─────────────────────────────────────────────────────────────────

class DS1000Loader:
    """
    Loads tasks from DS-1000 via HuggingFace datasets.

    DS-1000 covers: Pandas, NumPy, Matplotlib, Scipy, Sklearn, PyTorch, TensorFlow.
    Each problem is a function-level data science task sourced from StackOverflow.

    Args:
        libraries:   Filter to specific libraries e.g. ["Pandas", "Numpy"]
        max_tasks:   Cap the number returned
        split:       HuggingFace split (only "test" exists for DS-1000)
    """

    # DS-1000 library names as they appear in the dataset
    ALL_LIBRARIES = ["Pandas", "Numpy", "Matplotlib", "Scipy", "Sklearn", "PyTorch", "Tensorflow"]

    @staticmethod
    def load(
        libraries: Optional[List[str]] = None,
        max_tasks: Optional[int] = None,
        split: str = "test",
    ) -> List[Task]:
        from datasets import load_dataset

        lib_filter = libraries or DS1000Loader.ALL_LIBRARIES
        logger.info(f"Loading DS-1000 (libraries={lib_filter})")

        ds = load_dataset("xlangai/DS-1000", split=split)

        tasks = []
        for item in ds:
            lib = item["metadata"]["library"]
            if lib not in lib_filter:
                continue

            problem_id = item["metadata"]["problem_id"]
            task_id    = f"DS1000/{lib}/{problem_id}"

            tasks.append(Task(
                task_id=task_id,
                prompt=item["prompt"],
                test_code=item["test_code"],
                entry_point="solution",     # DS-1000 convention
                dataset="ds1000",
                canonical_solution=item["reference_code"],
                metadata={
                    "library": lib,
                    "perturbation_type": item["metadata"]["perturbation_type"],
                    "problem_id": problem_id,
                },
            ))

        if max_tasks:
            tasks = tasks[:max_tasks]

        logger.info(f"Loaded {len(tasks)} DS-1000 tasks")
        return tasks


# ─────────────────────────────────────────────────────────────────
# SOLUTION WRITER (BigCodeBench format)
# ─────────────────────────────────────────────────────────────────

def write_solutions_jsonl(solutions: List[Dict], path: str) -> None:
    """
    Write solutions in BigCodeBench's expected JSONL format.

    Each entry: {"task_id": ..., "solution": ..., "method": ...}
    BigCodeBench's evaluator reads "solution" or "completion".
    """
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for sol in solutions:
            f.write(json.dumps(sol) + "\n")
    logger.info(f"Wrote {len(solutions)} solutions to {path}")
