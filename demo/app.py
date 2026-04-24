"""
StateGen Live Demo
Run: .venv/bin/streamlit run demo/app.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="StateGen Demo", page_icon="🧩", layout="wide")

st.markdown("""
<style>
.pass-badge { background:#22c55e;color:white;padding:2px 10px;border-radius:12px;font-weight:600;font-size:13px; }
.fail-badge { background:#ef4444;color:white;padding:2px 10px;border-radius:12px;font-weight:600;font-size:13px; }
.step-box   { border-left:4px solid #6366f1;padding:8px 14px;margin:6px 0;background:#f8f8ff;border-radius:4px; }
.step-pass  { border-left:4px solid #22c55e;padding:8px 14px;margin:6px 0;background:#f0fdf4;border-radius:4px; }
.step-fail  { border-left:4px solid #ef4444;padding:8px 14px;margin:6px 0;background:#fef2f2;border-radius:4px; }
.step-warn  { border-left:4px solid #f59e0b;padding:8px 14px;margin:6px 0;background:#fffbeb;border-radius:4px; }
.problem-box{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:16px;font-size:14px;line-height:1.7; }
</style>
""", unsafe_allow_html=True)


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_tasks():
    from bigcodebench.data import get_bigcodebench
    return get_bigcodebench(subset="hard")


def badge(passed: bool) -> str:
    return '<span class="pass-badge">PASS ✓</span>' if passed else '<span class="fail-badge">FAIL ✗</span>'


# ── LLM setup ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("API_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("MODEL_NAME", "deepseek-coder")

def make_llm(tracker):
    from agents.llm_agent import LLMAgent
    return LLMAgent(
        model=DEEPSEEK_MODEL,
        provider="openai_compatible",
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        tracker=tracker,
        max_tokens=2048,
        temperature=0.2,
    )

def make_executor():
    from agents.execution_engine import ExecutionEngine
    return ExecutionEngine(timeout=10)

def make_tracker():
    from core.token_tracker import TokenTracker
    return TokenTracker()


# ── Method runners ────────────────────────────────────────────────────────────

def run_direct_gen(task, container):
    from baselines.base import extract_code
    tracker  = make_tracker()
    llm      = make_llm(tracker)
    executor = make_executor()

    with container:
        st.markdown("### Direct Generation")
        st.caption("One LLM call. No planning, no retry.")

        # Step 1: generate
        with st.status("Calling LLM…", expanded=True) as status:
            t0 = time.time()
            raw = llm.generate(task.prompt, task_id=task.task_id, method="direct_gen", call_type="generation")
            elapsed = time.time() - t0
            status.update(label=f"LLM responded in {elapsed:.1f}s", state="complete")

        code = extract_code(raw)

        st.markdown("**Generated code**")
        st.code(code, language="python")

        # Step 2: evaluate
        with st.status("Running BCB tests…", expanded=True) as status:
            result = executor.quick_run(code)
            passed = result.success
            status.update(
                label=f"Evaluation: {'PASS' if passed else 'FAIL'}",
                state="complete" if passed else "error",
            )

        st.markdown(f"**Result:** {badge(passed)}", unsafe_allow_html=True)
        if not passed and result.error:
            st.error(result.error[:400])

        _token_row(tracker, task.task_id, "direct_gen", elapsed)


def run_self_planning(task, container):
    from baselines.base import extract_code
    tracker  = make_tracker()
    llm      = make_llm(tracker)
    executor = make_executor()

    PLAN_PROMPT = (
        "You are an expert Python programmer.\n\n"
        "Write a numbered step-by-step plan (no code) for solving this problem:\n\n"
        "{problem}\n\n"
        "Output ONLY the numbered plan. No code, no explanation."
    )
    CODE_PROMPT = (
        "You are an expert Python programmer.\n\n"
        "Problem:\n{problem}\n\n"
        "Plan:\n{plan}\n\n"
        "Implement the plan in Python. Output ONLY the code in a ```python block."
    )

    with container:
        st.markdown("### Self-Planning")
        st.caption("Step 1: write a plan. Step 2: generate code from the plan.")

        t0 = time.time()

        # Step 1: plan
        with st.status("Step 1 — generating plan…", expanded=True) as status:
            plan = llm.generate(
                PLAN_PROMPT.format(problem=task.prompt),
                task_id=task.task_id, method="self_planning", call_type="planning",
            )
            status.update(label="Plan generated", state="complete")

        st.markdown("**Plan**")
        st.markdown(plan)

        # Step 2: code
        with st.status("Step 2 — generating code from plan…", expanded=True) as status:
            raw = llm.generate(
                CODE_PROMPT.format(problem=task.prompt, plan=plan),
                task_id=task.task_id, method="self_planning", call_type="generation",
            )
            status.update(label="Code generated", state="complete")

        code = extract_code(raw)
        st.markdown("**Generated code**")
        st.code(code, language="python")

        # Evaluate
        with st.status("Running BCB tests…", expanded=True) as status:
            result = executor.quick_run(code)
            passed = result.success
            status.update(label=f"Evaluation: {'PASS' if passed else 'FAIL'}", state="complete" if passed else "error")

        st.markdown(f"**Result:** {badge(passed)}", unsafe_allow_html=True)
        if not passed and result.error:
            st.error(result.error[:400])

        elapsed = time.time() - t0
        _token_row(tracker, task.task_id, "self_planning", elapsed)


def run_self_debugging(task, container):
    from baselines.base import extract_code
    tracker  = make_tracker()
    llm      = make_llm(tracker)
    executor = make_executor()

    GEN_PROMPT = (
        "You are an expert Python programmer. Solve this problem:\n\n{problem}\n\n"
        "Output ONLY the code in a ```python block."
    )
    EXPLAIN_PROMPT = (
        "Explain line-by-line what this Python code does:\n\n```python\n{code}\n```"
    )
    REPAIR_PROMPT = (
        "The code below has a bug.\n\nProblem:\n{problem}\n\n"
        "Code:\n```python\n{code}\n```\n\nError:\n{error}\n\n"
        "Explanation of the code:\n{explanation}\n\n"
        "Fix the bug. Output ONLY the corrected code in a ```python block."
    )
    MAX_RETRIES = 3

    with container:
        st.markdown("### Self-Debugging")
        st.caption("Generate → explain → repair (up to 3 rounds).")

        t0 = time.time()

        # Generate
        with st.status("Generating initial solution…", expanded=True) as s:
            raw  = llm.generate(GEN_PROMPT.format(problem=task.prompt),
                                task_id=task.task_id, method="self_debugging", call_type="generation")
            s.update(label="Initial solution generated", state="complete")

        code = extract_code(raw)
        st.markdown("**Initial code**")
        st.code(code, language="python")

        passed = False
        for attempt in range(MAX_RETRIES + 1):
            with st.status(f"Evaluating (attempt {attempt})…", expanded=True) as s:
                result  = executor.quick_run(code)
                passed  = result.success
                s.update(label=f"Attempt {attempt}: {'PASS' if passed else 'FAIL'}",
                         state="complete" if passed else "error")

            st.markdown(f"**Attempt {attempt}:** {badge(passed)}", unsafe_allow_html=True)

            if passed or attempt == MAX_RETRIES:
                break

            error = (result.error or "Unknown error")[:300]
            st.error(f"Error: {error}")

            with st.status(f"Explaining code (round {attempt+1})…", expanded=True) as s:
                explanation = llm.generate(
                    EXPLAIN_PROMPT.format(code=code),
                    task_id=task.task_id, method="self_debugging", call_type="explanation",
                )
                s.update(label="Explanation ready", state="complete")

            with st.expander(f"Explanation (round {attempt+1})", expanded=False):
                st.markdown(explanation)

            with st.status(f"Repairing (round {attempt+1})…", expanded=True) as s:
                raw = llm.generate(
                    REPAIR_PROMPT.format(problem=task.prompt, code=code,
                                        error=error, explanation=explanation),
                    task_id=task.task_id, method="self_debugging", call_type="repair",
                )
                s.update(label=f"Repair {attempt+1} complete", state="complete")

            code = extract_code(raw)
            st.markdown(f"**Repaired code (round {attempt+1})**")
            st.code(code, language="python")

        st.markdown(f"**Final result:** {badge(passed)}", unsafe_allow_html=True)
        elapsed = time.time() - t0
        _token_row(tracker, task.task_id, "self_debugging", elapsed)


def run_stategen(task, container, method_name="stategen", label="StateGen"):
    from baselines.stategen_baseline import _parse_decomposition, _DECOMPOSE_PROMPT, _REPAIR_PROMPT, _SYSTEM
    from baselines.base import extract_code
    from core.models import State, ProgramContext

    no_memory    = "no_memory"    in method_name
    no_decompose = "no_decompose" in method_name
    full_backtrack = "full_backtrack" in method_name
    MAX_STATE_RETRIES = 3

    tracker  = make_tracker()
    llm      = make_llm(tracker)
    executor = make_executor()

    # Memory manager (only for full stategen and full_backtrack)
    memory = None
    if not no_memory:
        try:
            from agents.memory_manager import MemoryManager
            memory = MemoryManager(threshold=0.85)
        except Exception:
            pass

    with container:
        st.markdown(f"### {label}")
        st.caption({
            "stategen":                "Decompose → per-state verify & repair → final BCB eval",
            "stategen_no_memory":      "Same as StateGen but without pattern memory cache",
            "stategen_no_decompose":   "Skip decomposition — treat the whole problem as one state",
            "stategen_full_backtrack": "On state failure, backtrack all the way to state 0",
        }.get(method_name, ""))

        t0 = time.time()
        context = ProgramContext()

        if no_decompose:
            # Single state covering the whole problem
            states_and_code = [(
                State(index=0, description="Complete solution", preconditions=[], postconditions=[]),
                ""
            )]
            st.info("Decomposition skipped — generating complete solution as one state.")
        else:
            # Decompose
            with st.status("Decomposing problem into states…", expanded=True) as s:
                raw_decomp = llm.generate(
                    _DECOMPOSE_PROMPT.format(problem=task.prompt), _SYSTEM,
                    task_id=task.task_id, method=method_name, call_type="decomposition",
                )
                states_and_code = _parse_decomposition(raw_decomp)
                s.update(label=f"Decomposed into {len(states_and_code)} states", state="complete")

            with st.expander("Decomposition output", expanded=False):
                st.code(raw_decomp[:2000], language="text")

            if not states_and_code:
                st.error("Decomposition failed — could not parse states. Falling back to single state.")
                states_and_code = [(
                    State(index=0, description="Complete solution", preconditions=[], postconditions=[]),
                    ""
                )]

        st.markdown("**Per-state execution**")

        total_retries = 0
        state_idx = 0

        while state_idx < len(states_and_code):
            state, init_code = states_and_code[state_idx]
            st.markdown(f"#### State {state.index}: _{state.description}_")

            fragment = init_code
            last_fragment = init_code
            state_passed = False

            for attempt in range(MAX_STATE_RETRIES):
                # Try memory first
                mem_hit = False
                if attempt == 0 and memory:
                    cached = memory.lookup(state, context)
                    if cached:
                        fragment = cached
                        mem_hit  = True

                if not fragment:
                    # Generate state code
                    with st.status(f"State {state.index} — generating (attempt {attempt})…") as s:
                        prompt = llm._build_state_prompt(state, context, None if attempt == 0 else error)
                        fragment = extract_code(llm.generate(
                            prompt, task_id=task.task_id,
                            method=method_name, call_type="generation", attempt=attempt,
                        ))
                        s.update(label=f"Code generated", state="complete")

                # Evaluate fragment
                full = context.full_code() + "\n" + fragment if context.full_code() else fragment
                with st.status(f"State {state.index} — verifying (attempt {attempt})…") as s:
                    result = executor.quick_run(full)
                    state_passed = result.success
                    src = " _(from memory)_" if mem_hit else ""
                    s.update(
                        label=f"State {state.index} attempt {attempt}{src}: {'PASS' if state_passed else 'FAIL'}",
                        state="complete" if state_passed else "error",
                    )

                col1, col2 = st.columns([1, 3])
                col1.markdown(f"{badge(state_passed)}", unsafe_allow_html=True)
                if mem_hit:
                    col2.caption("Loaded from memory cache")
                with st.expander(f"Code (attempt {attempt})", expanded=state_passed):
                    st.code(fragment, language="python")

                if state_passed:
                    if memory and not mem_hit:
                        memory.store(state, context, fragment)
                    break

                error = (result.error or "")[:300]
                st.error(f"Error: {error}")
                total_retries += 1
                last_fragment = fragment
                fragment = ""  # force regeneration on next attempt

            # If all attempts failed, keep the last generated code (best effort)
            if not state_passed and not fragment:
                fragment = last_fragment

            if state_passed:
                context.add(fragment)
                state_idx += 1
            else:
                st.warning(f"State {state.index} exhausted retries.")
                if full_backtrack and state_idx > 0:
                    st.info("Full backtrack — resetting to State 0.")
                    context = ProgramContext()
                    state_idx = 0
                else:
                    context.add(fragment)  # best effort
                    state_idx += 1

        # Final evaluation
        final_code = context.full_code()
        st.markdown("---")
        st.markdown("**Final code**")
        st.code(final_code, language="python")

        with st.status("Final evaluation…", expanded=True) as s:
            result = executor.quick_run(final_code)
            passed = result.success
            s.update(label=f"Final: {'PASS' if passed else 'FAIL'}", state="complete" if passed else "error")

        st.markdown(f"**Final result:** {badge(passed)}", unsafe_allow_html=True)
        if not passed and result.error:
            st.error(result.error[:400])

        elapsed = time.time() - t0
        _token_row(tracker, task.task_id, method_name, elapsed, extra_retries=total_retries)


def _token_row(tracker, task_id, method, elapsed, extra_retries=None):
    events = tracker.events_for_task(task_id)
    inp  = sum(e.input_tokens  for e in events)
    out  = sum(e.output_tokens for e in events)
    calls = len(events)
    st.divider()
    cols = st.columns(5 if extra_retries is not None else 4)
    cols[0].metric("Input tokens",  f"{inp:,}")
    cols[1].metric("Output tokens", f"{out:,}")
    cols[2].metric("Total tokens",  f"{inp+out:,}")
    cols[3].metric("Wall time",     f"{elapsed:.1f}s")
    if extra_retries is not None:
        cols[4].metric("State retries", extra_retries)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧩 StateGen Demo")
    st.caption("CMU 18-786 — Spring 2026")
    st.divider()

    tasks = load_tasks()
    task_ids = sorted(tasks.keys(), key=lambda x: int(x.split("/")[-1]))

    task_id = st.selectbox(
        "Select task",
        task_ids,
        format_func=lambda x: x.replace("BigCodeBench/", "BCB/"),
    )

    method = st.radio(
        "Method",
        ["direct_gen", "self_planning", "self_debugging",
         "stategen", "stategen_no_memory", "stategen_no_decompose", "stategen_full_backtrack"],
        format_func=lambda x: {
            "direct_gen":              "Direct Generation",
            "self_planning":           "Self-Planning",
            "self_debugging":          "Self-Debugging",
            "stategen":                "StateGen (full)",
            "stategen_no_memory":      "StateGen — no memory",
            "stategen_no_decompose":   "StateGen — no decompose",
            "stategen_full_backtrack": "StateGen — full backtrack",
        }[x],
    )

    run = st.button("▶ Run", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"Model: `{DEEPSEEK_MODEL}`")
    st.caption("API: DeepSeek (OpenAI-compatible)")


# ── Main ──────────────────────────────────────────────────────────────────────
task_obj = tasks[task_id]

# Wrap as a simple object the runners can access
class Task:
    def __init__(self, d):
        self.task_id = d["task_id"]
        self.prompt  = d["instruct_prompt"]

task = Task(task_obj)

st.title(f"Task: {task_id}")

with st.expander("Problem statement", expanded=True):
    st.markdown(task.prompt)

st.divider()

output_area = st.container()

if run:
    METHOD_LABELS = {
        "direct_gen":              "Direct Generation",
        "self_planning":           "Self-Planning",
        "self_debugging":          "Self-Debugging",
        "stategen":                "StateGen",
        "stategen_no_memory":      "StateGen (no memory)",
        "stategen_no_decompose":   "StateGen (no decompose)",
        "stategen_full_backtrack": "StateGen (full backtrack)",
    }
    with output_area:
        if method == "direct_gen":
            run_direct_gen(task, st.container())
        elif method == "self_planning":
            run_self_planning(task, st.container())
        elif method == "self_debugging":
            run_self_debugging(task, st.container())
        else:
            run_stategen(task, st.container(),
                         method_name=method, label=METHOD_LABELS[method])
else:
    with output_area:
        st.info("Select a task and method, then click **▶ Run** to see a live execution.")
