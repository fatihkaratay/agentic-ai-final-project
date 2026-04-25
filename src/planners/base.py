"""
Planner Base Interface
Shared contract for all path planners (A*, D* Lite, agent-wrapped variants).
Ensures every variant produces an identical PlanResult so the experiment
runner in Phase 5 can aggregate metrics without per-planner branching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.environment.grid import Grid


Position = Tuple[int, int]


@dataclass
class PlanResult:
    """
    Uniform planner output. All variants emit this for fair comparison.

    Fields:
      path             — ordered (row, col) waypoints from start to goal.
                         Empty list when no path is found.
      success          — True iff a complete path to the goal was returned.
      nodes_expanded   — number of grid cells the search expanded
                         (proxy for search effort, comparable across planners).
      plan_latency_ms  — wall-clock time for the planning call, in ms.
    """
    path:            List[Position] = field(default_factory=list)
    success:         bool           = False
    nodes_expanded:  int            = 0
    plan_latency_ms: float          = 0.0


class Planner(ABC):
    """
    Abstract base for all path planners.

    The two-method interface accommodates both:
      • One-shot planners (A*): replan() restarts search from current_pos.
      • Incremental planners (D* Lite): plan() seeds internal state;
        replan() reuses it to update only the affected sub-tree.

    Cost convention:
      • Base traversal cost per cell = 1.0.
      • Hazard cost = hazard_map[r, c] * hazard_weight (subclass-defined).
      • Blocked cells (grid.is_blocked) are non-traversable, regardless of hazard.
    """

    name: str = "BasePlanner"

    @abstractmethod
    def plan(
        self,
        start: Position,
        goal: Position,
        grid: Grid,
        hazard_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """
        Compute an initial path from start to goal.

        hazard_map: NxN float array (0–1) overlaid as soft cost.
                    If None, uses grid.hazard_map.
                    Pass np.zeros((N, N)) to plan as if memory were disabled.
        """
        ...

    @abstractmethod
    def replan(
        self,
        current_pos: Position,
        grid: Grid,
        hazard_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        """
        Update the path after the environment changed mid-episode.
        Reuses the goal from the most recent plan() call.
        """
        ...

    # ── Shared helpers ────────────────────────────────────────────────────

    @staticmethod
    def resolve_hazard_map(grid: Grid, hazard_map: Optional[np.ndarray]) -> np.ndarray:
        """Return the hazard_map to use, defaulting to the grid's own."""
        return grid.hazard_map if hazard_map is None else hazard_map
