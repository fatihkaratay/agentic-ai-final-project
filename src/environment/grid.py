"""
Grid World Simulation Engine
Discrete NxN grid environment with static and dynamic obstacles.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Cell type constants
FREE    = 0
STATIC  = 1
DYNAMIC = 2


@dataclass
class GridConfig:
    size: int   = 20       # grid is size x size
    obstacle_density: float = 0.15   # fraction of cells that are obstacles
    dynamic_ratio:    float = 0.5    # fraction of obstacles that are dynamic
    motion_model:     str   = "random_walk"  # "random_walk" | "directional" | "static"
    motion_speed:     float = 1.0    # probability an obstacle moves each step
                                     # 1.0 = every step, 0.2 = ~1 in 5 steps
    start: Tuple[int,int] = (0, 0)
    goal:  Optional[Tuple[int,int]] = None   # defaults to (size-1, size-1)


class Grid:
    """
    Simulation environment for the agentic pathfinding project.

    Responsibilities:
      - Maintain static wall layout and dynamic obstacle positions
      - Advance the simulation one step at a time (obstacles move)
      - Maintain a persistent hazard map updated across episodes
      - Provide a compact LLM-friendly state summary

    Cell values in get_grid():
      0 (FREE)    — passable
      1 (STATIC)  — permanent wall
      2 (DYNAMIC) — moving obstacle (dangerous this step)
    """

    def __init__(self, config: GridConfig, seed: Optional[int] = None):
        self.config   = config
        self.size     = config.size
        self.start    = config.start
        self.goal     = config.goal if config.goal else (config.size - 1, config.size - 1)

        # Hazard map persists across episodes (updated by Memory Manager)
        self.hazard_map: np.ndarray = np.zeros((self.size, self.size), dtype=float)

        self.reset(seed)

    # ── Episode lifecycle ──────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-initialise the grid for a new episode."""
        self._rng               = np.random.default_rng(seed)
        self._static_grid       = np.zeros((self.size, self.size), dtype=np.int8)
        self._dynamic_positions: List[Tuple[int,int]] = []
        self.agent_pos          = list(self.start)
        self.step_count         = 0
        self._place_obstacles()

    def step(self) -> None:
        """Advance the simulation by one time step (move dynamic obstacles)."""
        self.step_count += 1
        if self.config.motion_model == "static":
            return
        self._dynamic_positions = [
            self._move_obstacle(pos) if self._rng.random() < self.config.motion_speed else pos
            for pos in self._dynamic_positions
        ]

    # ── Query methods ──────────────────────────────────────────────────────

    def get_grid(self) -> np.ndarray:
        """Return a fresh NxN snapshot with static walls + current dynamic positions."""
        grid = self._static_grid.copy()
        for r, c in self._dynamic_positions:
            grid[r, c] = DYNAMIC
        return grid

    def is_blocked(self, pos: Tuple[int,int]) -> bool:
        """Return True if the cell is out of bounds or occupied by any obstacle."""
        r, c = pos
        if not (0 <= r < self.size and 0 <= c < self.size):
            return True
        if self._static_grid[r, c] == STATIC:
            return True
        return (r, c) in set(self._dynamic_positions)

    def get_state_summary(self, path: Optional[List[Tuple[int,int]]] = None) -> str:
        """
        Compact LLM-friendly text description of the current environment state.
        Intentionally concise to keep token costs low.
        """
        n_static  = int(np.sum(self._static_grid == STATIC))
        n_dynamic = len(self._dynamic_positions)

        # Top 5 highest-risk cells with non-zero score
        flat      = self.hazard_map.ravel()
        top_idx   = np.argsort(flat)[::-1][:5]
        top_risks = [
            f"({int(i//self.size)},{int(i%self.size)})={flat[i]:.2f}"
            for i in top_idx if flat[i] > 0.05
        ]

        # Which planned waypoints cross high-risk cells
        risky_waypoints: List[str] = []
        if path:
            for r, c in path[:15]:
                if self.hazard_map[r, c] > 0.2:
                    risky_waypoints.append(f"({r},{c})={self.hazard_map[r,c]:.2f}")

        return (
            f"Grid {self.size}x{self.size} | Step {self.step_count} | "
            f"Static walls: {n_static} | Dynamic obstacles: {n_dynamic} | "
            f"Agent: {tuple(self.agent_pos)} | Goal: {self.goal}\n"
            f"Top hazard cells: {top_risks or ['none yet']}\n"
            f"High-risk waypoints on planned path: {risky_waypoints or ['none']}"
        )

    # ── Hazard map ─────────────────────────────────────────────────────────

    def update_hazard_map(
        self,
        hazard_positions: List[Tuple[int,int]],
        increment: float = 0.15,
        decay:     float = 0.97,
    ) -> None:
        """
        Decay all cells then boost cells where obstacles were observed.
        Called by Memory Manager after each episode step or episode end.
        """
        self.hazard_map *= decay
        for r, c in hazard_positions:
            self.hazard_map[r, c] = min(1.0, self.hazard_map[r, c] + increment)

    def reset_hazard_map(self) -> None:
        """Clear the hazard map (used when starting a completely fresh experiment)."""
        self.hazard_map[:] = 0.0

    # ── Internal helpers ───────────────────────────────────────────────────

    def _place_obstacles(self) -> None:
        protected = {self.start, self.goal}
        all_cells = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in protected
        ]
        n_total   = int(len(all_cells) * self.config.obstacle_density)
        n_dynamic = int(n_total * self.config.dynamic_ratio)
        n_static  = n_total - n_dynamic

        indices = self._rng.choice(len(all_cells), size=n_total, replace=False)
        chosen  = [all_cells[i] for i in indices]

        for r, c in chosen[:n_static]:
            self._static_grid[r, c] = STATIC

        self._dynamic_positions = [tuple(p) for p in chosen[n_static:]]

    def _move_obstacle(self, pos: Tuple[int,int]) -> Tuple[int,int]:
        """Move obstacle to a random free adjacent cell, or stay if none available."""
        r, c = pos
        candidates = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.size and 0 <= nc < self.size
                    and self._static_grid[nr, nc] == FREE):
                candidates.append((nr, nc))
        if not candidates:
            return pos
        return candidates[int(self._rng.integers(len(candidates)))]
