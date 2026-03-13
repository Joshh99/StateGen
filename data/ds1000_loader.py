# data/ds1000_loader.py
# Standalone DS-1000 loader returning list[dict] for the experiment runner.
#
# Loads from HuggingFace: "xlangai/DS-1000"
# Filters to Pandas + Numpy (200-problem subset by default).

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Libraries included in our 200-problem subset
_DEFAULT_LIBRARIES = ["Pandas", "Numpy"]


def load_ds1000(
    max_tasks: Optional[int] = None,
    max_per_library: int = 100,
    split: str = "test",
    libraries: Optional[List[str]] = None,
) -> List[dict]:
    """
    Load DS-1000 problems from HuggingFace.

    Args:
        max_tasks:        Total cap across all libraries. When set, overrides
                          max_per_library as a ceiling (useful for quick test runs).
        max_per_library:  Max problems per library (default 100 → 100+100=200).
        split:            HuggingFace split — DS-1000 only exposes "test".
        libraries:        Which libraries to include (default: ["Pandas", "Numpy"]).

    Returns:
        List of dicts with keys:
            task_id, prompt, reference_code, library, test_code, perturbation_type

    Raises:
        RuntimeError: If the dataset cannot be downloaded.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required. Install with: pip install datasets"
        )

    lib_filter = libraries or _DEFAULT_LIBRARIES

    logger.info(f"Loading DS-1000 (split={split}, libraries={lib_filter})")
    try:
        ds = load_dataset("xlangai/DS-1000", split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download DS-1000 from HuggingFace: {exc}"
        ) from exc

    # Group tasks by library
    by_lib: dict[str, list[dict]] = {lib: [] for lib in lib_filter}
    for item in ds:
        lib = item["metadata"]["library"]
        if lib not in lib_filter:
            continue

        problem_id = item["metadata"]["problem_id"]
        by_lib[lib].append({
            "task_id":            f"DS-1000/{lib}/{problem_id}",
            "prompt":             item["prompt"],
            "reference_code":     item["reference_code"],
            "library":            lib,
            "test_code":          item["code_context"],
            "perturbation_type":  item["metadata"].get("perturbation_type", "Unknown"),
        })

    # Apply per-library cap
    # If max_tasks is set, distribute evenly across libraries
    if max_tasks is not None:
        per_lib_cap = max_tasks // len(lib_filter)
        # Give remainder to earlier libraries
        remainder = max_tasks % len(lib_filter)
    else:
        per_lib_cap = max_per_library
        remainder = 0

    tasks = []
    lib_counts = []
    for i, lib in enumerate(lib_filter):
        cap = per_lib_cap + (1 if i < remainder else 0)
        selected = by_lib[lib][:cap]
        tasks.extend(selected)
        lib_counts.append(f"{len(selected)} {lib}")

    # Final total cap (in case max_tasks is odd or libraries have fewer tasks)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    breakdown = " + ".join(lib_counts)
    logger.info(f"Loaded {breakdown} = {len(tasks)} tasks")

    return tasks
