# core/models.py
# Shared data classes used by all components
# Think of these as the "types" that agents pass to each other

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ─────────────────────────────────────────────
# STATE
# One step in the decomposed problem
# ─────────────────────────────────────────────
@dataclass
class State:
    index: int
    description: str               # e.g. "Filter DataFrame where age > 25"
    preconditions: List[str]       # e.g. ["df is a DataFrame"]
    postconditions: List[str]      # e.g. ["filtered_df is a DataFrame"]

    def __repr__(self):
        return f"State({self.index}: {self.description})"


# ─────────────────────────────────────────────
# PROGRAM CONTEXT
# Everything the LLM needs to know about
# what has already been written
# ─────────────────────────────────────────────
@dataclass
class ProgramContext:
    code_blocks: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)  # name → type

    def add(self, code: str):
        """Append a verified code block to the context."""
        self.code_blocks.append(code)
        self._infer_variables(code)

    def full_code(self) -> str:
        """Return the complete program so far."""
        return "\n".join(self.code_blocks)

    def is_empty(self) -> bool:
        return len(self.code_blocks) == 0

    def _infer_variables(self, code: str):
        """
        Very simple variable tracker — just looks for assignments.
        Will be improved with AST parsing later.
        """
        import re
        for var, val in re.findall(r'(\w+)\s*=\s*(.+)', code):
            if "DataFrame" in val or "read_csv" in val or "read_excel" in val:
                self.variables[var] = "DataFrame"
            elif val.strip().startswith("["):
                self.variables[var] = "list"
            elif val.strip().startswith("{"):
                self.variables[var] = "dict"
            elif val.strip().replace(".", "").lstrip("-").isdigit():
                self.variables[var] = "number"
            else:
                self.variables[var] = "object"


# ─────────────────────────────────────────────
# VERIFICATION RESULT
# Returned by the Execution Engine
# ─────────────────────────────────────────────
@dataclass
class VerificationResult:
    success: bool
    error: Optional[str] = None     # e.g. "NameError: name 'age' not defined"
    output: Optional[str] = None    # stdout from the execution
    exec_time: float = 0.0


# ─────────────────────────────────────────────
# CODE PATTERN
# One entry in the Memory Manager's store
# ─────────────────────────────────────────────
@dataclass
class CodePattern:
    state_description: str
    context_signature: str    # comma-separated "var:type" pairs
    code: str
    success_count: int = 1
    embedding: Any = None     # numpy array, set by MemoryManager


# ─────────────────────────────────────────────
# SOLVE RESULT
# Final output of StateController.solve()
# ─────────────────────────────────────────────
@dataclass
class SolveResult:
    success: bool
    code: Optional[str] = None
    failed_at_state: Optional[int] = None
    total_retries: int = 0
    patterns_reused: int = 0
    token_count: int = 0
    states_completed: int = 0
