"""
D* Lite (Koenig & Likhachev, 2002)
Incremental backward search that reuses prior search effort when the
environment changes mid-episode.

Movement model: 4-connected (matches A* and the grid obstacle motion).

Cost convention (identical to A* for fair comparison):
    c(u, v) = inf                             if v is blocked or out of bounds
    c(u, v) = 1.0 + hazard_weight * H[v]      otherwise

Implementation notes:
  • Lazy heap deletion: stale entries are re-keyed on pop when km has
    advanced; consistent vertices (g == rhs) are skipped if popped.
  • Obstacle changes are detected by diffing against a cached snapshot
    of the previous grid — caller need not enumerate changes manually.
  • Hazard map changes are NOT tracked across replans; if the agent
    needs to apply a new hazard map, call plan() to reset state.
    (Within an episode the hazard map is static, so this matches use.)
"""

import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.environment.grid import Grid
from src.planners.base import Planner, PlanResult, Position


NEIGHBOUR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
INF = float("inf")


class DStarLitePlanner(Planner):
    """Incremental backward search from goal; reuses state across replan calls."""

    name = "DStarLite"

    def __init__(self, hazard_weight: float = 2.0):
        self.hazard_weight = hazard_weight
        self._reset_state()

    # ── Public API ────────────────────────────────────────────────────────

    def plan(
        self,
        start: Position,
        goal:  Position,
        grid:  Grid,
        hazard_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        t0 = time.perf_counter()
        self._reset_state()
        self._grid     = grid
        self._size     = grid.size
        self._goal     = goal
        self._s_last   = start
        self._hazards  = self.resolve_hazard_map(grid, hazard_map)

        self._rhs[goal] = 0.0
        self._counter  += 1
        # Initial key for goal: [h(start, goal), 0]
        heapq.heappush(self._U, (self._h(start, goal), 0.0, self._counter, goal))

        self._cached_obstacles = self._snapshot_obstacles()

        self._compute_shortest_path(start)
        path    = self._extract_path(start)
        success = len(path) > 0 and path[-1] == goal

        return PlanResult(
            path=path, success=success,
            nodes_expanded=self._expanded_this_call,
            plan_latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def replan(
        self,
        current_pos: Position,
        grid: Grid,
        hazard_map: Optional[np.ndarray] = None,
    ) -> PlanResult:
        if self._goal is None:
            raise RuntimeError("replan() called before plan(); no state to reuse.")

        t0 = time.perf_counter()
        self._expanded_this_call = 0
        self._grid    = grid
        self._hazards = self.resolve_hazard_map(grid, hazard_map)

        # Detect obstacle changes since last call.
        new_obstacles = self._snapshot_obstacles()
        changed = new_obstacles.symmetric_difference(self._cached_obstacles)
        self._cached_obstacles = new_obstacles

        if changed:
            # km accumulator absorbs the heuristic shift caused by the moving
            # start, so existing heap keys remain valid for ordering.
            self._km    += self._h(self._s_last, current_pos)
            self._s_last = current_pos

            # Re-evaluate every cell touching a changed obstacle.
            affected: Set[Position] = set()
            for cell in changed:
                affected.add(cell)
                for n in self._neighbors(cell):
                    affected.add(n)
            for v in affected:
                self._update_vertex(v, current_pos)

        self._compute_shortest_path(current_pos)
        path    = self._extract_path(current_pos)
        success = len(path) > 0 and path[-1] == self._goal

        return PlanResult(
            path=path, success=success,
            nodes_expanded=self._expanded_this_call,
            plan_latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ── Core algorithm ────────────────────────────────────────────────────

    def _compute_shortest_path(self, current_start: Position) -> None:
        while self._U:
            top      = self._U[0]
            k_old    = (top[0], top[1])
            u        = top[3]

            start_key       = self._calculate_key(current_start, current_start)
            start_consistent = self._g_val(current_start) == self._rhs_val(current_start)
            if not (self._key_lt(k_old, start_key) or not start_consistent):
                break

            heapq.heappop(self._U)
            self._expanded_this_call += 1

            current_key = self._calculate_key(u, current_start)

            if self._key_lt(k_old, current_key):
                # Stale: km has grown since this entry was pushed; re-insert.
                self._push(u, current_start)
                continue

            if self._g_val(u) == self._rhs_val(u):
                # Already-consistent leftover — lazy delete.
                continue

            if self._g_val(u) > self._rhs_val(u):
                # Overconsistent → tighten g, propagate to predecessors.
                self._g[u] = self._rhs_val(u)
                for s in self._neighbors(u):
                    self._update_vertex(s, current_start)
            else:
                # Underconsistent → invalidate g, re-evaluate u and predecessors.
                self._g[u] = INF
                self._update_vertex(u, current_start)
                for s in self._neighbors(u):
                    self._update_vertex(s, current_start)

    def _update_vertex(self, u: Position, current_start: Position) -> None:
        if u != self._goal:
            best = INF
            for v in self._neighbors(u):
                c = self._cost(u, v)
                gv = self._g_val(v)
                if c < INF and gv < INF:
                    candidate = c + gv
                    if candidate < best:
                        best = candidate
            self._rhs[u] = best
        # Lazy removal — we never explicitly remove from the heap.
        if self._g_val(u) != self._rhs_val(u):
            self._push(u, current_start)

    # ── Path extraction ───────────────────────────────────────────────────

    def _extract_path(self, start: Position) -> List[Position]:
        if self._g_val(start) >= INF:
            return []
        path: List[Position] = [start]
        current = start
        cap = self._size * self._size      # safety against accidental cycles
        while current != self._goal and len(path) < cap:
            best_next: Optional[Position] = None
            best_cost = INF
            for n in self._neighbors(current):
                c  = self._cost(current, n)
                gn = self._g_val(n)
                if c < INF and gn < INF:
                    total = c + gn
                    if total < best_cost:
                        best_cost = total
                        best_next = n
            if best_next is None:
                return []
            path.append(best_next)
            current = best_next
        return path if current == self._goal else []

    # ── Helpers ───────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        self._goal:    Optional[Position] = None
        self._s_last:  Optional[Position] = None
        self._g:       Dict[Position, float] = {}
        self._rhs:     Dict[Position, float] = {}
        self._U:       List[Tuple[float, float, int, Position]] = []
        self._counter: int   = 0
        self._km:      float = 0.0
        self._cached_obstacles: Set[Position] = set()
        self._hazards: Optional[np.ndarray] = None
        self._grid:    Optional[Grid] = None
        self._size:    int = 0
        self._expanded_this_call: int = 0

    def _g_val(self, s: Position) -> float:
        return self._g.get(s, INF)

    def _rhs_val(self, s: Position) -> float:
        return self._rhs.get(s, INF)

    @staticmethod
    def _h(a: Position, b: Position) -> float:
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    @staticmethod
    def _key_lt(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        """Lexicographic key comparison (D* Lite ordering)."""
        return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])

    def _calculate_key(self, s: Position, current_start: Position) -> Tuple[float, float]:
        m = min(self._g_val(s), self._rhs_val(s))
        if m >= INF:
            return (INF, INF)
        return (m + self._h(current_start, s) + self._km, m)

    def _push(self, s: Position, current_start: Position) -> None:
        k = self._calculate_key(s, current_start)
        self._counter += 1
        heapq.heappush(self._U, (k[0], k[1], self._counter, s))

    def _neighbors(self, pos: Position):
        for dr, dc in NEIGHBOUR_OFFSETS:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self._size and 0 <= nc < self._size:
                yield (nr, nc)

    def _cost(self, u: Position, v: Position) -> float:
        if self._grid.is_blocked(u) or self._grid.is_blocked(v):
            return INF
        return 1.0 + self.hazard_weight * float(self._hazards[v[0], v[1]])

    def _snapshot_obstacles(self) -> Set[Position]:
        """Return the set of currently blocked cells in the grid."""
        blocked: Set[Position] = set()
        for r in range(self._size):
            for c in range(self._size):
                if self._grid.is_blocked((r, c)):
                    blocked.add((r, c))
        return blocked
