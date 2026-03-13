# StateGen

StateGen is a multi-agent code generation framework that models library-oriented programming tasks as Markov Decision Processes. Instead of regenerating entire programs on failure, StateGen decomposes problems into verifiable sequential states and retries only the failing state — reducing token waste while improving correctness.

## Structure
- `agents/` — State Controller, LLM Agent, Execution Engine, Memory Manager
- `baselines/` — Direct Generation, Self-Debugging, Self-Planning, ReAct
- `evaluation/` — Metrics (pass@k, token efficiency, reuse rate) and experiment runner
- `data/` — Dataset loading scripts (DS-1000, BigCodeBench)
- `experiments/` — Run configs
- `results/` — Experiment output (gitignored)
- `report/` — LaTeX source

## Setup
```bash
pip install -r requirements.txt
```

## Running Tests

### Unit tests (BCB sanitizer)

```bash
# From the project root
.venv/bin/pytest bigcodebench/tests/
```

### Single-task smoke test (no BCB server needed)

Runs one task end-to-end using `quick_check` only (no real pass@1):

```bash
set -a && source .env && set +a
python experiments/debug_one.py --method stategen
```

Change `--method` to `direct_gen`, `self_planning`, or `self_debugging` to test other baselines.

### Full experiment (requires BCB evaluation server)

1. Start the BCB server:
   ```bash
   cd bigcodebench-new && uvicorn bcb_server.server:app --port 8000
   ```

2. Run all four methods on the full 148-task hard subset:
   ```bash
   set -a && source .env && set +a
   python experiments/run_bigcodebench.py \
     --method direct_gen self_debugging self_planning stategen \
     --bcb_server http://localhost:8000
   ```
   Outputs: `results/{method}_solutions.jsonl`

3. Inspect results:
   ```bash
   python experiments/inspect_results.py                           # summary table
   python experiments/inspect_results.py --task_id BigCodeBench/13 # single task
   ```