# StateGen

MDP-based framework for LLM-driven code generation with local backtracking and cross-problem memory transfer.

**Paper:** StateGen: State-Based Code Generation with Local Backtracking and Transition Learning for Library-Oriented Programming — CMU-Africa MSAI, Spring 2025

---

## Overview

StateGen decomposes coding problems into verifiable sequential states and processes each state through a generate-verify cycle. When a state fails verification, only that state is regenerated (local backtracking), avoiding the full-program regeneration used by prior methods. Successful state implementations are stored in a semantic memory and retrieved for similar future states via embedding-based lookup. The system consists of four components: a **State Controller** that orchestrates the MDP loop, an **LLM Agent** that handles code generation and decomposition, an **Execution Engine** that verifies code in sandboxed subprocesses, and a **Memory Manager** that stores and retrieves successful code patterns.

---

## Repository Structure

```
agents/
  controller.py          State Controller — orchestrates the MDP feedback loop
  execution_engine.py    Sandboxed code execution and verification
  llm_agent.py           LiteLLM-based LLM wrapper with retry and token tracking
  memory_manager.py      Semantic memory store for code pattern reuse

baselines/
  base.py                Abstract base class and shared utilities for all methods
  direct_gen.py          Direct Generation baseline (single LLM call)
  self_planning.py       Self-Planning baseline (Jiang et al., 2024)
  self_debugging.py      Self-Debugging baseline (Chen et al., 2024)
  stategen_baseline.py   StateGen — MDP-based generation with local backtracking
  react.py               ReAct baseline (placeholder)

core/
  models.py              Shared data classes (State, ProgramContext, etc.)
  token_tracker.py       Per-call token accounting across all methods
  bcb_evaluator.py       HTTP client for the BigCodeBench evaluation server

data/
  loader.py              Unified data loading for BigCodeBench and DS-1000
  ds1000_loader.py       Standalone DS-1000 loader from HuggingFace

evaluation/
  metrics.py             Pass@1, token efficiency, and comparison table computation

experiments/
  run_bigcodebench.py    BigCodeBench experiment runner
  run_ds1000.py          DS-1000 experiment runner
  debug_one.py           Single-task debugging tool
  inspect_results.py     Human-readable result inspection

config.py                Central configuration (model, retries, thresholds)
main.py                  Single-problem entry point for local testing

notebooks/
  run_ds1000_colab.ipynb  Google Colab notebook for DS-1000 experiments
```

---

## Setup

### Requirements

- Python 3.10+
- openai >= 1.30.0
- anthropic >= 0.28.0
- sentence-transformers >= 2.7.0
- datasets >= 2.18.0
- numpy >= 1.26.0
- pandas >= 2.0.0
- tqdm >= 4.66.0

### Installation

```bash
git clone https://github.com/Joshh99/StateGen.git
cd StateGen
pip install -r requirements.txt
```

### API Keys

Create a `.env` file:

```
TOGETHER_API_KEY=your_key_here       # primary (Together.ai)
DEEPSEEK_API_KEY=your_key_here       # for DeepSeek-Coder experiments
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

---

## Reproducing Results

### DS-1000 (200 problems: 100 NumPy + 100 Pandas)

```bash
python experiments/run_ds1000.py --method direct_gen --max_tasks 200
python experiments/run_ds1000.py --method self_planning --max_tasks 200
python experiments/run_ds1000.py --method self_debugging --max_tasks 200
python experiments/run_ds1000.py --method stategen --max_tasks 200
```

### BigCodeBench (145 problems)

```bash
python experiments/run_bigcodebench.py --method direct_gen
python experiments/run_bigcodebench.py --method self_planning
python experiments/run_bigcodebench.py --method self_debugging
python experiments/run_bigcodebench.py --method stategen
```

Results are saved to `results/` as `.jsonl` files with a `metrics.json` summary.

---

## Key Results

| Method | DS-1000 Pass@1 | BCB Pass@1 | Avg Tokens (BCB) |
|---|---|---|---|
| Direct Generation | 95.5% | 23.4% | 604 |
| Self-Planning | 98.0% | 29.0% | 3,178 |
| Self-Debugging | 98.5% | 30.3% | 6,560 |
| StateGen (ours) | **99.0%** | **29.7%** | **3,093** |

---

## Running on Google Colab

See `notebooks/run_ds1000_colab.ipynb` for a ready-to-run Colab notebook that reproduces the DS-1000 experiments.

---

## Citation

```bibtex
@article{stategen2025,
  title={StateGen: State-Based Code Generation with Local Backtracking and Transition Learning},
  author={Maina, Joseph and Muriira, Patricia and Niyigena, Patrick and Momo, Joshua Wisdom and Chukwuma, Samuel},
  year={2025},
  institution={CMU-Africa}
}
```
