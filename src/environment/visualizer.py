"""
Grid World Visualizer
Renders the grid state with hazard overlay, planned path, and agent position.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from typing import Optional, List, Tuple

from src.environment.grid import Grid, FREE, STATIC, DYNAMIC


def render_grid(
    grid_obj: Grid,
    path: Optional[List[Tuple[int,int]]] = None,
    title: str = "Grid World",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Render the current grid state.

    Layers (bottom to top):
      1. Hazard map heatmap (red intensity = historical risk)
      2. Static walls (dark grey)
      3. Dynamic obstacles (orange)
      4. Planned path (blue dashed line)
      5. Agent position (green circle)
      6. Goal position (gold star)

    Args:
        grid_obj:  The Grid instance to render.
        path:      Optional list of (row, col) waypoints to draw.
        title:     Figure title.
        save_path: If provided, save the figure to this path.
        show:      If True, call plt.show() (use False for headless runs).

    Returns:
        The matplotlib Figure object.
    """
    size    = grid_obj.size
    grid    = grid_obj.get_grid()
    hazards = grid_obj.hazard_map

    fig, ax = plt.subplots(figsize=(7, 7))

    # ── Layer 1: hazard heatmap background ────────────────────────────────
    hazard_rgba = plt.cm.Reds(hazards)
    hazard_rgba[..., 3] = np.clip(hazards * 1.5, 0, 0.6)   # alpha proportional to risk
    ax.imshow(hazard_rgba, origin="upper", zorder=1)

    # ── Layer 2: static walls ─────────────────────────────────────────────
    wall_mask = np.ma.masked_where(grid != STATIC, grid)
    ax.imshow(
        np.where(grid == STATIC, 0.2, np.nan),
        cmap="Greys", vmin=0, vmax=1,
        origin="upper", zorder=2
    )

    # ── Layer 3: dynamic obstacles ────────────────────────────────────────
    for r, c in grid_obj._dynamic_positions:
        ax.add_patch(plt.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            color="darkorange", alpha=0.85, zorder=3
        ))

    # ── Layer 4: planned path ─────────────────────────────────────────────
    if path and len(path) > 1:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color="royalblue", linewidth=2,
                linestyle="--", marker=".", markersize=4,
                zorder=4, label="Planned path")

    # ── Layer 5: agent position ───────────────────────────────────────────
    ar, ac = grid_obj.agent_pos
    ax.add_patch(plt.Circle(
        (ac, ar), 0.4,
        color="limegreen", zorder=5, label="Agent"
    ))

    # ── Layer 6: goal position ────────────────────────────────────────────
    gr, gc = grid_obj.goal
    ax.plot(gc, gr, marker="*", markersize=18,
            color="gold", markeredgecolor="black",
            zorder=5, label="Goal")

    # ── Formatting ────────────────────────────────────────────────────────
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="lightgrey", linewidth=0.4, zorder=0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="dimgrey",    label="Static wall"),
        mpatches.Patch(facecolor="darkorange", label="Dynamic obstacle"),
        mpatches.Patch(facecolor="salmon",     alpha=0.6, label="Hazard (memory)"),
        mpatches.Patch(facecolor="limegreen",  label="Agent"),
        mpatches.Patch(facecolor="gold",       label="Goal"),
    ]
    if path:
        legend_elements.append(
            mpatches.Patch(facecolor="royalblue", label="Planned path")
        )
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()

    return fig
