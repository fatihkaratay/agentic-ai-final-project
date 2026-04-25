"""
A* Pathfinding
Optimal search with a hazard-aware cost function and Manhattan heuristic.

Movement model: 4-connected (up/down/left/right) — matches the obstacle
motion model in src/environment/grid.py.

Cost per traversed cell: 1.0 + hazard_weight * hazard_map[r, c]
  • hazard_weight = 0  →  pure distance-optimal A* (ignores memory)
  • hazard_weight > 0  →  soft avoidance of historically risky cells
"""

import heapq
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment.grid import Grid
from src.planners.base import Planner, PlanResult, Position


NEIGHBOUR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class AStarPlanner(Planner):
    """One-shot A* search. replan() restarts from the agent's current cell."""

    name = "AStar"

    def __init__(self, hazard_weight: float = 2.0):
        """
        hazard_weight: multiplier applied to hazard_map[r, c] when forming
                       per-cell cost. The default of 2.0 makes a maximally
                       risky cell (hazard=1.0) cost 3× a clear cell.
        """
        self.hazard_weight = hazard_weight
        self._last_goal: Optional[Position] = None

    # ── Public API ────────────────────────────────────────────────────────

    def plan(
        self,
        start: Position,
        goal: Position,
        grid: Grid,
        hazard_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        self._last_goal = goal
        return self._search(start, goal, grid, self.resolve_hazard_map(grid, hazard_map))

    def replan(
        self,
        current_pos: Position,
        grid: Grid,
        hazard_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        if self._last_goal is None:
            raise RuntimeError("replan() called before plan(); no goal cached.")
        return self._search(
            current_pos, self._last_goal, grid,
            self.resolve_hazard_map(grid, hazard_map),
        )

    # ── Search core ───────────────────────────────────────────────────────

    def _search(
        self,
        start: Position,
        goal:  Position,
        grid:  Grid,
        hazards: np.ndarray,
    ) -> PlanResult:
        t0 = time.perf_counter()

        # Edge case: start blocked or out of bounds → no path.
        # (Goal blocked is also unreachable, but we let the search discover it
        # so nodes_expanded is reported honestly.)
        if grid.is_blocked(start):
            return PlanResult(
                path=[], success=False, nodes_expanded=0,
                plan_latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        if start == goal:
            return PlanResult(
                path=[start], success=True, nodes_expanded=1,
                plan_latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        open_heap: List[Tuple[float, int, Position]] = []
        # heap entries: (f_score, tiebreak_counter, position)
        # tiebreak_counter prevents heapq from comparing tuples with positions
        # when f_scores tie — keeps ordering deterministic.
        counter = 0
        heapq.heappush(open_heap, (self._h(start, goal), counter, start))

        came_from:  Dict[Position, Position] = {}
        g_score:    Dict[Position, float]    = {start: 0.0}
        closed:     set                      = set()
        expanded                              = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)
            expanded += 1

            if current == goal:
                return PlanResult(
                    path=self._reconstruct(came_from, current),
                    success=True,
                    nodes_expanded=expanded,
                    plan_latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

            for dr, dc in NEIGHBOUR_OFFSETS:
                nbr = (current[0] + dr, current[1] + dc)
                if grid.is_blocked(nbr) or nbr in closed:
                    continue

                step_cost     = 1.0 + self.hazard_weight * float(hazards[nbr[0], nbr[1]])
                tentative_g   = g_score[current] + step_cost

                if tentative_g < g_score.get(nbr, float("inf")):
                    came_from[nbr] = current
                    g_score[nbr]   = tentative_g
                    counter       += 1
                    heapq.heappush(
                        open_heap,
                        (tentative_g + self._h(nbr, goal), counter, nbr),
                    )

        # Open set exhausted without reaching goal.
        return PlanResult(
            path=[], success=False, nodes_expanded=expanded,
            plan_latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _h(a: Position, b: Position) -> float:
        """Manhattan distance — admissible for 4-connected unit-cost grids."""
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    @staticmethod
    def _reconstruct(came_from: Dict[Position, Position], end: Position) -> List[Position]:
        path = [end]
        while end in came_from:
            end = came_from[end]
            path.append(end)
        path.reverse()
        return path
