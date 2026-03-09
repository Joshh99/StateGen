from baselines.direct_gen import DirectGenBaseline
from baselines.self_planning import SelfPlanningBaseline
from baselines.self_debugging import SelfDebuggingBaseline
from baselines.stategen_baseline import StateGenBaseline

ALL_BASELINES = {
    "direct_gen":    DirectGenBaseline,
    "self_planning": SelfPlanningBaseline,
    "self_debugging": SelfDebuggingBaseline,
    "stategen":      StateGenBaseline,
}
