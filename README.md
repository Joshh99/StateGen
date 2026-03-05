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