"""
Environment Scanner Node (reactive)

Blends two information sources into a single per-step uncertainty overlay
that the planner uses as its cost-map weights:

    1. Persistent hazard map (cross-episode memory: grid.hazard_map)
    2. Real-time dynamic-obstacle proximity (this step's grid snapshot)

The blend keeps cross-episode learning in play while letting the agent
react to obstacles it can currently see. Static walls are NOT bumped here:
the planner's cost function already treats them as non-traversable.
"""

from typing import Any, Dict

import numpy as np

from src.agent.state import AgentState
from src.environment.grid import Grid


NEIGHBOUR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def compute_uncertainty(
    grid: Grid,
    proximity_bump: float = 0.3,
) -> np.ndarray:
    """
    Build the per-step uncertainty overlay.

        uncertainty[r, c] =
            1.0                          if cell currently holds a dynamic obstacle
            max(historical, bump)        if cell is 4-adjacent to a dynamic obstacle
            grid.hazard_map[r, c]        otherwise

    Returns a fresh array; grid.hazard_map is NOT mutated.
    """
    uncertainty = grid.hazard_map.copy()

    for r, c in grid._dynamic_positions:
        uncertainty[r, c] = 1.0
        for dr, dc in NEIGHBOUR_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.size and 0 <= nc < grid.size:
                if uncertainty[nr, nc] < proximity_bump:
                    uncertainty[nr, nc] = proximity_bump

    return np.clip(uncertainty, 0.0, 1.0)


def environment_scanner_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node entry point. Returns a partial AgentState update.

    Updates:
      hazard_map   — replaced with the blended uncertainty overlay
      episode_log  — appended with a scanner trace
    """
    grid        = state["grid"]
    uncertainty = compute_uncertainty(grid)

    log_entry = {
        "step":             state.get("step_count", 0),
        "node":             "environment_scanner",
        "n_dynamic":        len(grid._dynamic_positions),
        "max_uncertainty":  float(uncertainty.max()),
        "mean_uncertainty": float(uncertainty.mean()),
    }

    return {
        "hazard_map":  uncertainty,
        "episode_log": state.get("episode_log", []) + [log_entry],
    }
