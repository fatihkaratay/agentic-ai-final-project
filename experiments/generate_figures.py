"""
Generate the data-driven figures from Phase 5 results.

Reads per-episode JSONL emitted by run_experiments.py and writes:

  fig4_success_rate_by_variant.png   — grouped bar chart, 4 variants × 3 conditions
  fig5_learning_curve.png            — episode-over-episode success rolling avg
  fig6_cost_quality_scatter.png      — mean wall-clock latency vs. success rate
  fig7_sample_justification.png      — formatted text box of an actual LLM
                                       reflection summary

The proposal also requires three diagrammatic figures (LangGraph topology,
data flow, grid visualization). Those are produced separately:
  • Topology and data-flow are drawn manually (graphviz / draw.io).
  • Grid visualization comes from src/environment/visualizer.py / demo_grid.py.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER   = ["astar", "dstar_lite", "agent_noreflection", "agent_full"]
CONDITION_ORDER = ["static", "dynamic_low", "dynamic_high"]
VARIANT_LABEL   = {
    "astar":              "A*",
    "dstar_lite":         "D* Lite",
    "agent_noreflection": "Agent\n(NoReflection)",
    "agent_full":         "Full Agent",
}
CONDITION_LABEL = {
    "static":       "Static",
    "dynamic_low":  "Dynamic-Low",
    "dynamic_high": "Dynamic-High",
}
VARIANT_COLOR = {
    "astar":              "#999999",
    "dstar_lite":         "#4477aa",
    "agent_noreflection": "#ee8866",
    "agent_full":         "#228833",
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_records(input_dir: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".jsonl") or fname.startswith("_memstore__"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_cell[(rec["variant"], rec["condition"])].append(rec)
    return by_cell


# ── Figure 4: success rate grouped bar chart ─────────────────────────────────

def fig_success_rate(records, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    n_v = len(VARIANT_ORDER)
    n_c = len(CONDITION_ORDER)
    bar_w = 0.8 / n_v
    x_base = np.arange(n_c)

    for i, variant in enumerate(VARIANT_ORDER):
        rates = []
        for cond in CONDITION_ORDER:
            recs = records.get((variant, cond), [])
            if recs:
                rates.append(sum(1 for r in recs if r["outcome"] == "success") / len(recs))
            else:
                rates.append(0.0)
        offset = (i - (n_v - 1) / 2) * bar_w
        ax.bar(
            x_base + offset, rates, bar_w,
            label=VARIANT_LABEL[variant],
            color=VARIANT_COLOR[variant],
            edgecolor="black", linewidth=0.4,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([CONDITION_LABEL[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success Rate by Variant and Environment Condition", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── Figure 5: episode-over-episode learning curve ────────────────────────────

def _rolling_success(records: List[Dict[str, Any]], window: int = 5) -> List[float]:
    """Rolling success rate over a fixed window, ordered by episode_idx."""
    recs = sorted(records, key=lambda r: r["episode_idx"])
    flags = [1.0 if r["outcome"] == "success" else 0.0 for r in recs]
    out = []
    for i in range(len(flags)):
        lo = max(0, i - window + 1)
        out.append(sum(flags[lo : i + 1]) / (i - lo + 1))
    return out


def fig_learning_curve(records, out_path: str) -> None:
    """Rolling success rate per variant on Dynamic-Low (where memory should matter)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cond = "dynamic_low"
    for variant in VARIANT_ORDER:
        recs = records.get((variant, cond), [])
        if not recs:
            continue
        ys = _rolling_success(recs, window=5)
        xs = list(range(1, len(ys) + 1))
        ax.plot(
            xs, ys,
            label=VARIANT_LABEL[variant].replace("\n", " "),
            color=VARIANT_COLOR[variant],
            marker="o", markersize=3, linewidth=1.6,
        )
    ax.set_xlabel("Episode index (sequential)")
    ax.set_ylabel("Rolling success rate (window = 5)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Episode-over-Episode Learning Curve — Dynamic-Low", fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── Figure 6: cost-quality scatter ───────────────────────────────────────────

def fig_cost_quality(records, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"static": "o", "dynamic_low": "s", "dynamic_high": "^"}
    for variant in VARIANT_ORDER:
        for cond in CONDITION_ORDER:
            recs = records.get((variant, cond), [])
            if not recs:
                continue
            mean_latency = np.mean([r.get("wall_clock_ms", 0.0) for r in recs])
            success_rate = sum(1 for r in recs if r["outcome"] == "success") / len(recs)
            ax.scatter(
                mean_latency, success_rate,
                marker=markers[cond],
                s=140,
                color=VARIANT_COLOR[variant],
                edgecolor="black", linewidth=0.6,
                alpha=0.85,
                label=f"{VARIANT_LABEL[variant].replace(chr(10), ' ')} — {CONDITION_LABEL[cond]}",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Mean wall-clock per episode (ms, log scale)")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Cost vs. Quality across Variants and Conditions", fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7, ncol=2, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── Figure 7: sample LLM reflection ──────────────────────────────────────────

def _pick_sample_summary(records) -> Tuple[str, str]:
    """Return (header, summary) — prefer a successful agent_full episode."""
    for variant in ("agent_full", "agent_noreflection"):
        for cond in ("dynamic_low", "static", "dynamic_high"):
            for r in records.get((variant, cond), []):
                summary = r.get("reflection_summary") or ""
                if summary and "[no-reflection]" not in summary and "[LLM unavailable]" not in summary:
                    header = f"{VARIANT_LABEL[variant].replace(chr(10), ' ')}  ·  {CONDITION_LABEL[cond]}  ·  seed={r['seed']}  ·  outcome={r['outcome']}"
                    return header, summary
    return "no LLM summary captured", ""


def fig_sample_justification(records, out_path: str) -> None:
    header, summary = _pick_sample_summary(records)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    ax.text(
        0.02, 0.92, "Sample LLM Reflection Output",
        fontsize=12, fontweight="bold", transform=ax.transAxes,
    )
    ax.text(
        0.02, 0.82, header,
        fontsize=8.5, color="#444",
        transform=ax.transAxes, family="monospace",
    )
    # Wrapped body text
    import textwrap
    wrapped = "\n".join(textwrap.wrap(summary, width=95))
    ax.text(
        0.02, 0.05, wrapped or "(no LLM summary captured in this run)",
        fontsize=10.5, color="black",
        transform=ax.transAxes, family="serif", verticalalignment="bottom",
    )
    # Decorative border
    ax.add_patch(plt.Rectangle(
        (0.005, 0.01), 0.99, 0.98,
        transform=ax.transAxes, fill=False,
        edgecolor="black", linewidth=0.8,
    ))
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Phase 5 figures.")
    ap.add_argument("--input",  default="results/logs/pilot")
    ap.add_argument("--output", default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    records = _load_records(args.input)
    if not records:
        print(f"No records found under {args.input!r}.")
        return

    targets = [
        ("fig4_success_rate_by_variant.png", fig_success_rate),
        ("fig5_learning_curve.png",          fig_learning_curve),
        ("fig6_cost_quality_scatter.png",    fig_cost_quality),
        ("fig7_sample_justification.png",    fig_sample_justification),
    ]
    for fname, fn in targets:
        out_path = os.path.join(args.output, fname)
        fn(records, out_path)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
