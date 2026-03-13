# StateGen Baseline Implementation Overview

## Directory Structure

```
StateGen/
├── agents/                    # Core execution components
│   ├── llm_agent.py          # LLM API wrapper (OpenAI, Anthropic, DeepSeek)
│   ├── execution_engine.py   # Sandboxed code execution
│   ├── memory_manager.py     # Pattern caching across tasks
│   ├── controller.py         # State controller (old API, not used in baselines)
│   └── __init__.py
│
├── baselines/                # Baseline implementations (4 methods)
│   ├── base.py              # Abstract BaseBaseline class + SolveResult dataclass
│   ├── direct_gen.py        # Single LLM call, no retries
│   ├── self_planning.py     # Jiang et al. 2024: Planning → Implementation
│   ├── self_debugging.py    # Chen et al. 2024: Explain → Repair cycles
│   ├── stategen_baseline.py # StateGen: Decompose → Verify per-state → Repair
│   └── __init__.py          # Exports ALL_BASELINES dict
│
├── core/                    # Data models & infrastructure
│   ├── models.py           # State, ProgramContext, VerificationResult, CodePattern
│   ├── token_tracker.py    # Records token usage per task/method/call
│   ├── bcb_evaluator.py    # Calls BigCodeBench API server for real test results
│   └── __init__.py
│
├── data/                    # Dataset loaders
│   └── loader.py           # BigCodeBenchLoader, Task dataclass
│
├── evaluation/              # Metrics & analysis
│   ├── metrics.py          # Pass@1, token efficiency, reuse rate
│   └── (results analysis utilities)
│
├── experiments/             # Experiment runners
│   ├── run_bigcodebench.py # Main runner (CLI: --method, --max_tasks, --remote_eval)
│   └── inspect_results.py  # Human-readable result inspection
│
├── config.py               # Centralized config (all .getenv() calls)
├── main.py                 # Local single-task test runner
├── requirements.txt        # Dependencies
├── .env.example            # Template for API keys
└── README.md               # Brief overview
```

---

## Baseline Methods Summary

### 1. **DirectGenBaseline** (`baselines/direct_gen.py`)
**What it does:** Single LLM call, no retries. Establishes token cost floor.

**LLM calls:**
- 1 generation call with prompt: `"Write a complete Python implementation for: {problem}"`

**Results saved as:** `results/direct_gen_solutions.jsonl`
- Each line: `{"task_id": "...", "solution": "...", "method": "direct_gen", "total_input_tokens": N, "total_output_tokens": M, ...}`

**Token efficiency:** Lowest (single call) but lowest accuracy.

---

### 2. **SelfPlanningBaseline** (`baselines/self_planning.py`)
**What it does:** Faithful implementation of Jiang et al. (ACM TOSEM 2024).
- **Phase 1 (Planning):** LLM generates numbered step-by-step plan
- **Phase 2 (Implementation):** LLM generates code guided by plan
- **On failure:** Regenerate BOTH plan and code from scratch (global retry)

**LLM calls per attempt:**
- 2 calls per attempt: 1 planning + 1 generation
- Retries up to `MAX_RETRIES=3` times

**Results saved as:** `results/self_planning_solutions.jsonl`

**Token efficiency:**
- Extra planning call (cost) but often better first-attempt accuracy
- Full regeneration on failure is expensive vs. StateGen

---

### 3. **SelfDebuggingBaseline** (`baselines/self_debugging.py`)
**What it does:** Faithful implementation of Chen et al. (ICLR 2024 — Google DeepMind).

Each debugging turn (after initial generation):
1. **Execution:** Run code, catch error
2. **Explanation:** LLM explains code line-by-line (rubber duck debugging)
3. **Repair:** LLM regenerates FULL program with explanation + error as context

**LLM calls:**
- Initial: 1 generation call
- Per debug turn: 1 explanation + 1 repair call (2 calls/turn)
- Max `MAX_RETRIES=3` debug turns

**Token efficiency:**
- **Token growth issue:** Every repair re-feeds entire program + full explanation
- On BigCodeBench (avg 1112-char prompts), this scales poorly
- This is the inefficiency StateGen solves via local backtracking

---

### 4. **StateGenBaseline** (`baselines/stategen_baseline.py`)
**What it does:** MDP-based code generation with local backtracking and memory caching.

**Algorithm:**
1. **Decomposition (1 call):** LLM breaks problem into 2–3 sequential states
   - State 0: Imports + module-level constants
   - State 1: Main function definition(s)
   - State 2 (optional): Secondary function
   - Parser extracts state metadata + initial code from response

2. **Per-state verification loop (local backtracking):**
   - Attempt 0: Try memory cache, else use decomposition code
   - Assemble: Concatenate all previous states' code + current state
   - Quick-run (syntax + import check): If passes → store in memory, advance
   - If fails: Call LLM to repair ONLY this state (not full program), retry
   - Max `MAX_RETRIES=3` retries per state

3. **Final evaluation + repair (if needed):**
   - Assemble full code, call _evaluate() (BCB server or quick_check)
   - If fails: regenerate full program (simpler than self_debugging — no explain step)
   - Max `MAX_RETRIES=3` final repair attempts

**Memory caching:**
- Pattern lookup: State description + context signature (variable types) → cached code
- On successful state completion: Store (state, context) → code in memory
- Reuse across tasks within same run

**Results saved as:** `results/stategen_solutions.jsonl`

**Token efficiency vs. self_debugging:**
- Repair prompts contain ONLY failing state's context (few lines)
- No re-feeding of entire program or explanation
- Reduces token waste significantly

---

## All Baseline Methods at a Glance

| Method | LLM Calls/Problem | Token Scaling | Key Advantage |
|--------|-------------------|----------------|---|
| DirectGen | 1 | Fixed | Baseline; fast |
| SelfPlanning | 2 × #attempts | Linear (2-6 calls) | Better accuracy via planning |
| SelfDebugging | 1 + 2 × #debug_turns | **Quadratic** (grows with code size) | Explicit error analysis |
| StateGen | 1 + per-state repairs + final repairs | **Linear** (repairs are smaller) | Local backtracking + memory |

---

## LLM API Configuration

**File:** `config.py` (lines 8–15) + `agents/llm_agent.py` (lines 31–90)

**Default provider:** `DeepSeek` (openai_compatible)
```python
LLM_PROVIDER      = "openai_compatible"  # "openai" | "anthropic" | "openai_compatible"
LLM_MODEL         = "deepseek-coder"
LLM_BASE_URL      = "https://api.deepseek.com/v1"
```

**Supported providers:**
1. **OpenAI** (`provider="openai"`)
   - Model: any GPT model (gpt-4o, gpt-4-turbo, etc.)
   - API key: `OPENAI_API_KEY`
   - Uses `openai.OpenAI` client

2. **Anthropic** (`provider="anthropic"`)
   - Model: any Claude model
   - API key: `ANTHROPIC_API_KEY`
   - Uses `anthropic.Anthropic` client

3. **OpenAI-compatible** (`provider="openai_compatible"`)
   - Supports: DeepSeek, Together AI, Fireworks, local vLLM, etc.
   - API key: `DEEPSEEK_API_KEY` (or fallback to `OPENAI_API_KEY`)
   - Base URL: Configurable via `LLM_BASE_URL` or `--base_url` CLI arg

**API key resolution** (LLMAgent.__init__, lines 72–80):
1. Explicit `api_key=` parameter (if provided)
2. Environment variable:
   - Anthropic → `ANTHROPIC_API_KEY`
   - OpenAI-compatible → `DEEPSEEK_API_KEY` (first choice) or `OPENAI_API_KEY`
   - OpenAI → `OPENAI_API_KEY`
3. Raises error if not found

**Token tracking:**
- Built into LLMAgent: `_call_openai()` and `_call_anthropic()` extract `prompt_tokens` and `completion_tokens` from API response
- Recorded to `TokenTracker` with: task_id, method, call_type, attempt index
- Saved to `results/token_events.jsonl` after run

---

## Running Experiments

### Main Entry Point: `experiments/run_bigcodebench.py`

**Basic usage:**
```bash
python experiments/run_bigcodebench.py \
  --method all \
  --max_tasks 50 \
  --provider openai_compatible \
  --model deepseek-coder
```

**Full command reference:**
```bash
python experiments/run_bigcodebench.py \
  --method direct_gen self_planning self_debugging stategen  # or 'all'
  --max_tasks 50                      # number of problems to solve
  --subset hard                       # "hard" (smaller) or "complete" (1140 tasks)
  --use_instruct                      # shorter instruct_prompt vs full complete_prompt
  --provider openai_compatible        # "openai" | "anthropic" | "openai_compatible"
  --model deepseek-coder              # model name
  --base_url https://api.deepseek.com/v1  # for openai_compatible
  --api_key YOUR_KEY                  # override environment variable
  --results_dir results/              # output directory
  --skip_existing                     # skip already-computed tasks (default: True)
  --bcb_server http://localhost:8000  # optional: BigCodeBench API server for real test suite
  --remote_eval                       # run official BigCodeBench evaluation after generation
  --execution docker                  # "docker" (local) or "gradio" (remote HF API)
```

**Examples:**
```bash
# Quick test: 10 tasks, all methods, DeepSeek
python experiments/run_bigcodebench.py --method all --max_tasks 10

# Self-debugging only, GPT-4o, with official BigCodeBench evaluation
python experiments/run_bigcodebench.py \
  --method self_debugging \
  --max_tasks 50 \
  --provider openai \
  --model gpt-4o \
  --remote_eval

# StateGen vs. Direct on specific tasks
python experiments/run_bigcodebench.py \
  --method direct_gen stategen \
  --task_ids BigCodeBench/13 BigCodeBench/42 BigCodeBench/100
```

---

## Output Files & Schema

### Solutions JSONL (`results/{method}_solutions.jsonl`)
One JSON object per line, written after each task:
```json
{
  "task_id": "BigCodeBench/42",
  "solution": "import pandas as pd\ndef my_func(df):\n  return df[df['age'] > 25].mean()",
  "method": "self_planning",
  "total_input_tokens": 2145,
  "total_output_tokens": 523,
  "total_tokens": 2668,
  "num_attempts": 2,
  "num_retries": 1,
  "wall_time": 12.34,
  "attempt_details": [
    {
      "attempt": 0,
      "passed": false,
      "error": "NameError: name 'df' not defined",
      "test_errors": {},
      "plan_length": 8
    },
    {
      "attempt": 1,
      "passed": true,
      "error": "",
      "test_errors": {}
    }
  ]
}
```

### Token Events JSONL (`results/token_events.jsonl`)
Raw token tracking:
```json
{
  "task_id": "BigCodeBench/42",
  "method": "self_planning",
  "call_type": "planning",
  "input_tokens": 1200,
  "output_tokens": 150,
  "attempt": 0,
  "timestamp": "2026-03-12T15:45:30.123"
}
```

### Metrics Summary (`results/metrics.json`)
Aggregated stats per method:
```json
{
  "self_planning": {
    "pass_at_1": 0.65,
    "avg_tokens": 2840,
    "avg_attempts": 1.8,
    "memory_reuse_rate": 0.0
  },
  "stategen": {
    "pass_at_1": 0.72,
    "avg_tokens": 2120,
    "avg_attempts": 2.3,
    "memory_reuse_rate": 0.23
  }
}
```

---

## Machine Portability Issues 🚩

### 🔴 Critical: API Keys
**Problem:** No `.env` file committed; defaults to DeepSeek API.
- Running on another machine requires creating `.env` with API credentials
- Template available at `.env.example`

**Fix before running:**
```bash
cp .env.example .env
# Edit .env with your API keys:
# - DEEPSEEK_API_KEY (default provider)
# - OPENAI_API_KEY (if using OpenAI)
# - ANTHROPIC_API_KEY (if using Anthropic)
```

---

### 🟡 Medium: BigCodeBench Dataset
**Problem:** Loader expects cloned BigCodeBench repo at `./bigcodebench/`
```python
# data/loader.py line 12
sys.path.insert(0, str(Path(__file__).parent.parent / "bigcodebench"))
from bigcodebench.data import get_bigcodebench
```

**Status:** Not present in this repo (not tracked in git).

**Fix:** Clone and place at project root:
```bash
git clone https://github.com/bigcode-project/bigcodebench.git
```

---

### 🟡 Medium: Hardcoded Paths
**Config paths** (`config.py`):
- `MEMORY_FILE = "memory/patterns.json"` — relative path, created on-demand
- `DATA_DIR = "data/"` — relative path
- `RESULTS_DIR = "results/"` — relative path
- `LOG_FILE = "logs/stategen.log"` — relative path

All created with `os.makedirs(..., exist_ok=True)` so should work on any machine **IF** run from project root.

**Best practice:** Always run from `StateGen/` root:
```bash
cd /path/to/StateGen
python experiments/run_bigcodebench.py --method all --max_tasks 10
```

---

### 🟡 Medium: Memory File Persistence
**Problem:** Memory patterns stored in `memory/patterns.json` (git-ignored).
- Patterns learned from one run don't carry to the next run on a different machine
- Workaround: Copy `memory/patterns.json` across machines if needed

---

### 🟢 Low: Temp File Handling
**Code execution:** ExecutionEngine uses `tempfile.NamedTemporaryFile()` (OS-native, auto-cleaned).
- Should work on any OS (Windows, macOS, Linux)

**Subprocess timeouts:** `EXEC_TIMEOUT = 5.0` (seconds) may need adjustment on slow machines.

---

## Code Walkthrough: One Baseline (self_planning)

**File:** `baselines/self_planning.py` (162 lines)

```python
class SelfPlanningBaseline(BaseBaseline):
    METHOD_NAME = "self_planning"

    def solve(self, task: Task) -> SolveResult:
        for attempt in range(MAX_RETRIES):  # default 3
            # Phase 1: Generate plan
            plan = self.llm.generate(
                _PLAN_PROMPT.format(problem=task.prompt),
                _SYSTEM,
                task_id=task.task_id,
                method=self.METHOD_NAME,
                call_type="planning",
                attempt=attempt,
            )

            # Phase 2: Generate code from plan
            raw = self.llm.generate(
                _CODE_PROMPT.format(problem=task.prompt, plan=plan),
                _SYSTEM,
                task_id=task.task_id,
                method=self.METHOD_NAME,
                call_type="generation",
                attempt=attempt,
            )

            # Evaluate
            solution = extract_code(raw)
            ok, error, _ = self._evaluate(solution, task.task_id)

            if ok:
                break

        # Token totals from tracker
        in_tok, out_tok = self._token_totals(task.task_id)

        return SolveResult(
            task_id=task.task_id,
            method=self.METHOD_NAME,
            solution=solution,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            num_attempts=len(details) * 2,  # 2 calls per attempt
            num_retries=len(details) * 2 - 2,
            wall_time=time.time() - t0,
            attempt_details=details,
        )
```

**Key insights:**
1. `self.llm.generate()` handles API calls AND token tracking
2. `self._evaluate()` checks correctness (BCB server or quick_check)
3. `self._token_totals()` queries `TokenTracker` for this task+method
4. **On failure:** Loop restarts, regenerates plan + code (contrast with StateGen local backtracking)

---

## How to Extend

### Adding a new baseline:
1. Create `baselines/my_method.py`
2. Inherit from `BaseBaseline`
3. Implement `solve(task: Task) -> SolveResult`
4. Register in `baselines/__init__.py`: `ALL_BASELINES["my_method"] = MyBaseline`
5. Run: `python experiments/run_bigcodebench.py --method my_method`

### Changing default LLM:
1. Edit `config.py` lines 10–12
2. OR pass `--provider openai --model gpt-4o` to the CLI

### Tuning hyperparameters:
- `MAX_RETRIES` (config.py): max retries per state/attempt
- `TEMPERATURE` (config.py): LLM creativity (0.2 = deterministic)
- `TOP_P` (config.py): nucleus sampling threshold
- `MAX_NEW_TOKENS` (config.py): max output length (2048)

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Baselines** | 4: DirectGen, SelfPlanning, SelfDebugging, StateGen |
| **LLM Support** | OpenAI, Anthropic, DeepSeek (default), any OpenAI-compatible |
| **Token Tracking** | Per task/method/call, saved to JSONL |
| **Code Execution** | Sandboxed subprocess, 5-second timeout |
| **Memory Caching** | StateGen-specific, pattern lookup by (state_desc, context_vars) |
| **Evaluation** | BigCodeBench API server (real tests) or quick_check (syntax/imports) |
| **Results Format** | JSONL per method, metrics JSON, token events JSONL |
| **Dependencies** | openai, anthropic, sentence-transformers, datasets, pandas, numpy |
| **Python Version** | 3.10+ (uses type hints) |
| **Critical Setup** | API key in `.env`, BigCodeBench repo clone |

