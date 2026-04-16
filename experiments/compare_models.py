#!/usr/bin/env python3
# experiments/compare_models.py
# Cross-model comparison table: pass@1 and token efficiency
# across multiple experiment directories (one directory per model).
#
# Each directory contains *_solutions.jsonl files produced by
# run_bigcodebench.py (with or without --experiment).
# A manifest.json in the directory labels the model (auto-created by
# run_bigcodebench.py --experiment).  If absent, the directory name is used.
#
# Usage:
#   python experiments/compare_models.py \
#       --experiments results/deepseek results/gpt4omini results/claude_sonnet \
#       --output results/cross_model.json
#
#   # Or with explicit labels:
#   python experiments/compare_models.py \
#       --experiments results/deepseek:DeepSeek-Coder \
#                     results/gpt4omini:GPT-4o-mini \
#                     results/claude_sonnet:"Claude Sonnet 4.6" \
#       --output results/cross_model.json
#
#   # Show only specific methods:
#   python experiments/compare_models.py \
#       --experiments results/deepseek results/gpt4omini \
#       --methods direct_gen stategen self_debugging \
#       --output results/cross_model.json

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import compute_metrics, load_results, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Display order for methods (unlisted methods appear at the end alphabetically)
_METHOD_ORDER = [
    "direct_gen",
    "self_planning",
    "self_debugging",
    "stategen",
    "stategen_no_memory",
    "stategen_no_decompose",
    "stategen_full_backtrack",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_experiment_arg(arg: str) -> tuple:
    """
    Parse 'path:label' or 'path' experiment argument.
    Returns (Path, label_str).
    """
    if ":" in arg:
        # Split on LAST colon to allow colons in paths (Windows drive letters etc.)
        idx = arg.rfind(":")
        path_str, label = arg[:idx], arg[idx + 1:]
    else:
        path_str, label = arg, None
    return Path(path_str), label


def _read_manifest(d: Path) -> dict:
    p = d / "manifest.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _label_for(d: Path, manifest: dict, explicit_label: Optional[str]) -> str:
    if explicit_label:
        return explicit_label
    if manifest.get("model"):
        return manifest["model"]
    return d.name


def _sort_methods(methods: list) -> list:
    ordered = [m for m in _METHOD_ORDER if m in methods]
    rest = sorted(m for m in methods if m not in _METHOD_ORDER)
    return ordered + rest


def _fmt(val, fmt: str) -> str:
    if val is None:
        return "—"
    try:
        return format(val, fmt)
    except (ValueError, TypeError):
        return str(val)


# ── Grid tables ───────────────────────────────────────────────────────────────

def _print_grid(
    title: str,
    metric_key: str,
    fmt: str,
    all_metrics: dict,      # {model_label: {method: {metric: value}}}
    methods: list,
    models: list,
    col_width: int = 16,
) -> None:
    method_col = 28
    print(f"\n{'─' * (method_col + col_width * len(models))}")
    print(f"  {title}")
    print(f"{'─' * (method_col + col_width * len(models))}")
    header = f"{'Method':<{method_col}}" + "".join(
        f"{m[:col_width - 2]:>{col_width}}" for m in models
    )
    print(header)
    print("─" * len(header))

    for method in methods:
        row = f"{method:<{method_col}}"
        for model in models:
            val = all_metrics.get(model, {}).get(method, {}).get(metric_key)
            row += f"{_fmt(val, fmt):>{col_width}}"
        print(row)

    print("─" * len(header))


def _print_delta_grid(
    title: str,
    metric_key: str,
    fmt: str,
    all_metrics: dict,
    methods: list,
    models: list,
    baseline_method: str = "direct_gen",
    col_width: int = 16,
) -> None:
    """Print a grid showing delta vs baseline_method for each model."""
    method_col = 28
    print(f"\n{'─' * (method_col + col_width * len(models))}")
    print(f"  {title}  (Δ vs {baseline_method})")
    print(f"{'─' * (method_col + col_width * len(models))}")
    header = f"{'Method':<{method_col}}" + "".join(
        f"{m[:col_width - 2]:>{col_width}}" for m in models
    )
    print(header)
    print("─" * len(header))

    for method in methods:
        if method == baseline_method:
            continue
        row = f"{method:<{method_col}}"
        for model in models:
            m_metrics = all_metrics.get(model, {})
            val = m_metrics.get(method, {}).get(metric_key)
            base = m_metrics.get(baseline_method, {}).get(metric_key)
            if val is not None and base is not None:
                delta = val - base
                sign = "+" if delta >= 0 else ""
                cell = f"{sign}{_fmt(delta, fmt)}"
            else:
                cell = "—"
            row += f"{cell:>{col_width}}"
        print(row)

    print("─" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-model comparison of pass@1 and token efficiency."
    )
    p.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        metavar="DIR[:LABEL]",
        help=(
            "Experiment directories, optionally with a display label after ':'. "
            "E.g. results/deepseek:DeepSeek  results/gpt4omini:GPT-4o-mini"
        ),
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=None,
        metavar="METHOD",
        help="Methods to include (default: all found). E.g. --methods direct_gen stategen",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Save combined metrics to this JSON file (optional).",
    )
    p.add_argument(
        "--baseline",
        default="direct_gen",
        metavar="METHOD",
        help="Baseline method for delta tables (default: direct_gen).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    all_metrics: dict = {}   # {model_label: {method: {metric: value}}}
    model_labels: list = []

    for exp_arg in args.experiments:
        d, explicit_label = _parse_experiment_arg(exp_arg)

        if not d.exists():
            logger.warning(f"Directory not found, skipping: {d}")
            continue

        manifest = _read_manifest(d)
        label = _label_for(d, manifest, explicit_label)

        logger.info(
            f"Loading [{label}] from {d}  "
            f"(model={manifest.get('model','?')}, "
            f"provider={manifest.get('provider','?')}, "
            f"date={manifest.get('date','?')})"
        )

        records = load_results(str(d), methods=args.methods)
        if not records:
            logger.warning(f"  No results found in {d} — skipping.")
            continue

        metrics = compute_metrics(records)
        all_metrics[label] = metrics
        model_labels.append(label)
        logger.info(
            f"  Found {len(metrics)} method(s): {sorted(metrics.keys())}"
        )

    if not all_metrics:
        logger.error("No results loaded from any experiment directory.")
        sys.exit(1)

    # Union of all methods across all experiments, in display order
    all_methods_seen = set()
    for m in all_metrics.values():
        all_methods_seen.update(m.keys())
    if args.methods:
        all_methods_seen &= set(args.methods)
    methods = _sort_methods(list(all_methods_seen))

    print("\n" + "=" * 72)
    print("  CROSS-MODEL COMPARISON")
    print("=" * 72)
    print(f"  Models: {model_labels}")
    print(f"  Methods: {methods}")
    print(f"  Tasks: {next(iter(next(iter(all_metrics.values())).values())).get('n_tasks','?')} per method")

    # ── pass@1 ────────────────────────────────────────────────────────────────
    _print_grid(
        "pass@1",
        "pass_at_1",
        ".1%",
        all_metrics,
        methods,
        model_labels,
    )

    _print_delta_grid(
        "pass@1 improvement",
        "pass_at_1",
        ".1%",
        all_metrics,
        methods,
        model_labels,
        baseline_method=args.baseline,
    )

    # ── avg tokens ────────────────────────────────────────────────────────────
    _print_grid(
        "avg tokens / task",
        "avg_total_tokens",
        ".0f",
        all_metrics,
        methods,
        model_labels,
    )

    # ── token overhead vs direct_gen ─────────────────────────────────────────
    _print_grid(
        "token overhead vs direct_gen",
        "token_overhead_vs_direct",
        "+.1%",
        all_metrics,
        methods,
        model_labels,
    )

    # ── avg retries ───────────────────────────────────────────────────────────
    _print_grid(
        "avg retries / task",
        "avg_retries",
        ".2f",
        all_metrics,
        methods,
        model_labels,
    )

    # ── wasted token rate ─────────────────────────────────────────────────────
    _print_grid(
        "wasted token rate (tokens on failed tasks / total)",
        "wasted_token_rate",
        ".1%",
        all_metrics,
        methods,
        model_labels,
    )

    # ── stategen gain summary ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STATEGEN EFFICIENCY SUMMARY")
    print("─" * 72)
    print(f"  {'Model':<24}  {'stategen pass@1':>16}  {'vs direct':>10}  {'vs self_debug':>14}  {'tok saving vs self_debug':>24}")
    print("─" * 72)
    for model in model_labels:
        m = all_metrics.get(model, {})
        sg = m.get("stategen", {})
        dg = m.get("direct_gen", {})
        sd = m.get("self_debugging", {})

        p1     = sg.get("pass_at_1")
        p1_dg  = dg.get("pass_at_1")
        p1_sd  = sd.get("pass_at_1")
        tok_sg = sg.get("avg_total_tokens")
        tok_sd = sd.get("avg_total_tokens")

        vs_dg  = f"{(p1 - p1_dg):+.1%}" if p1 is not None and p1_dg is not None else "—"
        vs_sd  = f"{(p1 - p1_sd):+.1%}" if p1 is not None and p1_sd is not None else "—"
        tok_save = f"{(1 - tok_sg / tok_sd):.1%} fewer" if tok_sg and tok_sd else "—"

        p1_str = _fmt(p1, ".1%")
        print(f"  {model:<24}  {p1_str:>16}  {vs_dg:>10}  {vs_sd:>14}  {tok_save:>24}")

    print("=" * 72)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined = {
            "model_labels": model_labels,
            "methods": methods,
            "metrics": all_metrics,
        }
        with open(out, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        logger.info(f"Combined metrics saved → {out}")


if __name__ == "__main__":
    main()
