from baselines.direct_gen import DirectGenBaseline
from baselines.self_planning import SelfPlanningBaseline
from baselines.self_debugging import SelfDebuggingBaseline
from baselines.stategen_baseline import StateGenBaseline
from baselines.stategen_no_memory import StateGenNoMemoryBaseline
from baselines.stategen_no_decompose import StateGenNoDecomposeBaseline
from baselines.stategen_full_backtrack import StateGenFullBacktrackBaseline

# ── Original baselines ────────────────────────────────────────────────────────
# direct_gen    : single LLM call, no retries
# self_planning : plan + code, global retry on failure (Jiang et al.)
# self_debugging: generate → explain → repair whole program (Chen et al.)
# stategen      : MDP with local backtracking + memory (ours)
#
# ── Ablation variants ─────────────────────────────────────────────────────────
# stategen_no_memory      : stategen without pattern reuse
#                           → isolates memory contribution
# stategen_no_decompose   : stategen without decomposition (single state)
#                           → isolates decomposition contribution
# stategen_full_backtrack : stategen with full-context repair
#                           → tests minimal vs full MDP context in repair

ALL_BASELINES = {
    # Original
    "direct_gen":              DirectGenBaseline,
    "self_planning":           SelfPlanningBaseline,
    "self_debugging":          SelfDebuggingBaseline,
    "stategen":                StateGenBaseline,
    # Ablations
    "stategen_no_memory":      StateGenNoMemoryBaseline,
    "stategen_no_decompose":   StateGenNoDecomposeBaseline,
    "stategen_full_backtrack": StateGenFullBacktrackBaseline,
}
