"""
Generate the architecture/visualization figures referenced by the paper:

  fig1_langgraph_topology.png   — 7-node LangGraph topology with conditional edges
  fig2_data_flow.png            — agent reasoning pipeline + AgentState fields
  fig3_grid_world.png           — Grid world visualization with hazard overlay + path

All figures are produced from code (no external diagram tools) so the paper
remains reproducible from the repo.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ── Figure 1: LangGraph topology ─────────────────────────────────────────────

NODE_STYLES = {
    "reactive":     {"facecolor": "#cce4ff", "edgecolor": "#1f4e79"},
    "deliberative": {"facecolor": "#ffe1b3", "edgecolor": "#a05a00"},
    "llm":          {"facecolor": "#d9c2e8", "edgecolor": "#5d3a76"},
    "memory":       {"facecolor": "#cfe9d3", "edgecolor": "#2d6a36"},
    "reflective":   {"facecolor": "#f5d0d0", "edgecolor": "#923232"},
    "terminal":     {"facecolor": "#eeeeee", "edgecolor": "#444444"},
}

NODES = [
    # (id, label, kind, x, y)
    ("start",  "START",                "terminal",     5.5,  9.5),
    ("scan",   "Environment\nScanner", "reactive",     5.5,  8.3),
    ("plan",   "A* Planner",           "deliberative", 5.5,  7.0),
    ("mem_r",  "Memory\nManager (read)","memory",      5.5,  5.7),
    ("risk",   "Path Risk\nEvaluator",  "llm",         5.5,  4.4),
    ("heal",   "Path Healer",          "deliberative", 2.0,  4.4),
    ("exec",   "Execution\nMonitor",   "reactive",     5.5,  2.6),
    ("refl",   "Reflection",           "reflective",   9.0,  2.6),
    ("mem_w",  "Memory\nManager (write)","memory",     9.0,  1.2),
    ("end",    "END",                  "terminal",     5.5,  1.2),
]

# (src, dst, label, style, curve_rad)
# curve_rad — connectionstyle bend; 0.0 = straight, +/- = curved (radian-ish)
EDGES = [
    ("start", "scan",  "",                                     "solid",  0.0),
    ("scan",  "plan",  "",                                     "solid",  0.0),
    ("plan",  "mem_r", "",                                     "solid",  0.0),
    ("mem_r", "risk",  "",                                     "solid",  0.0),
    ("risk",  "heal",  "risk ≥ θ\n& heals < max",              "dashed", 0.0),
    ("risk",  "exec",  "risk < θ",                             "dashed", 0.0),
    ("heal",  "plan",  "replan",                               "solid",  0.4),
    ("exec",  "exec",  "",                                     "dashed", 0.0),
    ("exec",  "risk",  "obstacle\ndetected",                   "dashed", 0.5),
    ("exec",  "plan",  "stuck\n(empty path)",                  "dashed", -1.0),
    ("exec",  "refl",  "terminal",                             "dashed", 0.0),
    ("refl",  "mem_w", "",                                     "solid",  0.0),
    ("mem_w", "end",   "",                                     "solid",  0.0),
]


def _draw_node(ax, x, y, label, kind, w=2.0, h=0.8):
    style = NODE_STYLES[kind]
    box = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.10",
        facecolor=style["facecolor"], edgecolor=style["edgecolor"], linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold")


def _draw_edge(ax, src, dst, label, kind, rad=0.0):
    sx, sy = src
    dx, dy = dst
    if (sx, sy) == (dx, dy):                      # self-loop
        loop = mpatches.FancyArrowPatch(
            (sx + 1.05, sy + 0.25), (sx + 1.05, sy - 0.25),
            connectionstyle="arc3,rad=-1.6",
            arrowstyle="-|>", mutation_scale=11,
            color="#444", linewidth=1.0,
            linestyle="--",
        )
        ax.add_patch(loop)
        ax.text(sx + 2.1, sy, "running\n& clear", fontsize=7, color="#333",
                style="italic", ha="left", va="center")
        return
    style = "--" if kind == "dashed" else "-"
    cs    = f"arc3,rad={rad}" if rad else "arc3,rad=0"
    arrow = mpatches.FancyArrowPatch(
        (sx, sy), (dx, dy),
        connectionstyle=cs,
        arrowstyle="-|>", mutation_scale=11,
        color="#444", linewidth=1.0,
        linestyle=style,
        shrinkA=22, shrinkB=22,
    )
    ax.add_patch(arrow)
    if label:
        # For curved edges, push the label outward perpendicular to the chord
        # by an amount that scales with |rad| so the label clears any straight-
        # line nodes the chord crosses through.
        mx, my = (sx + dx) / 2, (sy + dy) / 2
        if rad:
            import math
            chord_dx, chord_dy = dx - sx, dy - sy
            length = max(1e-6, math.hypot(chord_dx, chord_dy))
            px, py = -chord_dy / length, chord_dx / length
            offset = max(1.0, abs(rad) * length * 0.35) * (1 if rad > 0 else -1)
            mx += px * offset
            my += py * offset
        ax.text(mx, my, label, fontsize=7, color="#333",
                style="italic", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bbb",
                          alpha=0.95, linewidth=0.4))


def fig_topology(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 11))
    pos = {nid: (x, y) for nid, _, _, x, y in NODES}

    for nid, label, kind, x, y in NODES:
        _draw_node(ax, x, y, label, kind)

    for src, dst, label, kind, rad in EDGES:
        _draw_edge(ax, pos[src], pos[dst], label, kind, rad)

    legend = [
        mpatches.Patch(facecolor=NODE_STYLES["reactive"]["facecolor"],     edgecolor=NODE_STYLES["reactive"]["edgecolor"],     label="Reactive"),
        mpatches.Patch(facecolor=NODE_STYLES["deliberative"]["facecolor"], edgecolor=NODE_STYLES["deliberative"]["edgecolor"], label="Deliberative"),
        mpatches.Patch(facecolor=NODE_STYLES["llm"]["facecolor"],          edgecolor=NODE_STYLES["llm"]["edgecolor"],          label="Deliberative / LLM"),
        mpatches.Patch(facecolor=NODE_STYLES["memory"]["facecolor"],       edgecolor=NODE_STYLES["memory"]["edgecolor"],       label="Memory"),
        mpatches.Patch(facecolor=NODE_STYLES["reflective"]["facecolor"],   edgecolor=NODE_STYLES["reflective"]["edgecolor"],   label="Reflective"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.95)

    ax.set_xlim(-0.5, 11.0)
    ax.set_ylim(-0.2, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("LangGraph Agent Topology\n"
                 "(dashed edges are conditional — routed by AgentState)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: data flow / pipeline ───────────────────────────────────────────

PIPELINE = [
    # (label, kind, fields populated)
    ("Environment\nScanner",     "reactive",     "hazard_map (blended)\nepisode_log"),
    ("A* Planner",               "deliberative", "planned_path\nreplan_count\nepisode_log"),
    ("Memory Manager\n(read)",   "memory",       "memory_context\nepisode_log"),
    ("Path Risk\nEvaluator (LLM)", "llm",        "risk_score\nrisk_justification\nepisode_log"),
    ("Path Healer\n(if risky)",  "deliberative", "hazard_map (bumped)\nheal_count\nepisode_log"),
    ("Execution\nMonitor",       "reactive",     "current_pos\nstep_count\nstuck_count\nepisode_status"),
    ("Reflection\n(LLM)",        "reflective",   "reflection_summary\ngrid.hazard_map\nepisode_log"),
    ("Memory Manager\n(write)",  "memory",       "(persists episode\n to JSONL store)"),
]


def fig_data_flow(out_path: str) -> None:
    n = len(PIPELINE)
    fig, ax = plt.subplots(figsize=(13, 5.4))

    for i, (label, kind, fields) in enumerate(PIPELINE):
        x = 1.5 + i * 1.55
        # Node box
        style = NODE_STYLES[kind]
        box = mpatches.FancyBboxPatch(
            (x - 0.65, 2.6), 1.30, 1.0,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            facecolor=style["facecolor"], edgecolor=style["edgecolor"], linewidth=1.3,
        )
        ax.add_patch(box)
        ax.text(x, 3.1, label, ha="center", va="center",
                fontsize=8, fontweight="bold")

        # Fields populated (below node)
        ax.text(x, 1.2, fields, ha="center", va="top",
                fontsize=7, color="#222",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fbfbfb",
                          ec="#bbb", linewidth=0.6))

        # Arrow to next
        if i < n - 1:
            nx = 1.5 + (i + 1) * 1.55
            ax.add_patch(mpatches.FancyArrowPatch(
                (x + 0.65, 3.1), (nx - 0.65, 3.1),
                arrowstyle="-|>", mutation_scale=11,
                color="#444", linewidth=1.0,
            ))

    # AgentState shared bar (across the top)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.6, 4.2), n * 1.55 - 0.4, 0.7,
        boxstyle="round,pad=0.1",
        facecolor="#fff8d6", edgecolor="#9c8800", linewidth=1.2,
    ))
    ax.text((0.6 + n * 1.55 - 0.4) / 2 + 0.4, 4.55,
            "Shared AgentState (TypedDict) — passed to and updated by every node",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#5b4d00")

    ax.set_xlim(0, n * 1.55 + 1.5)
    ax.set_ylim(-0.4, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Agent Reasoning Pipeline — Per-Step Data Flow",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: grid world visualization (real Grid + visualizer) ──────────────

def fig_grid_world(out_path: str) -> None:
    """Render a populated grid using the actual Grid + visualizer modules."""
    from src.environment.grid import Grid, GridConfig
    from src.environment.visualizer import render_grid

    cfg = GridConfig(
        size             = 20,
        obstacle_density = 0.15,
        dynamic_ratio    = 0.5,
        motion_model     = "random_walk",
        motion_speed     = 1.0,
        start            = (0, 0),
        goal             = (19, 19),
    )
    grid = Grid(cfg, seed=42)

    # Simulate ~15 steps so a meaningful hazard map develops.
    for _ in range(15):
        grid.step()
        grid.update_hazard_map(grid._dynamic_positions, increment=0.12, decay=0.97)

    # An illustrative diagonal path for the visual.
    path = [(i, i) for i in range(20)]

    fig = render_grid(
        grid,
        path      = path,
        title     = "Grid World — historical hazard heatmap, dynamic obstacles, planned path",
        save_path = out_path,
    )
    plt.close(fig)


# ── Driver ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Phase-6 architecture figures.")
    ap.add_argument("--output", default="results/figures/run50")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    targets = [
        ("fig1_langgraph_topology.png", fig_topology),
        ("fig2_data_flow.png",          fig_data_flow),
        ("fig3_grid_world.png",         fig_grid_world),
    ]
    for fname, fn in targets:
        out = os.path.join(args.output, fname)
        fn(out)
        print(f"  → {out}")


if __name__ == "__main__":
    main()
