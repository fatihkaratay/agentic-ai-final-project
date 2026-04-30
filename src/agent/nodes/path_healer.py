"""
Path Healer Node (deliberative)

Strategy: bump-and-replan. The healer never overrides A*'s geometry — it
nudges the cost map upward at risky waypoints and lets the planner re-derive
the path. This preserves A*'s optimality guarantees within the modified
cost space ("memory as soft priors", per the proposal's design decisions).

Graph routing the healer expects:
    Risk Evaluator → (risk ≥ threshold AND heal_count < max) → Path Healer
    Path Healer    → A* Planner (replan)

Heuristic:
  1. Walk planned_path[1:-1] (skip current cell and goal — both are fixed).
  2. Cells with hazard_map[r,c] ≥ heal_threshold get raised to ≥ bump_floor.
  3. If no waypoint crosses heal_threshold but the LLM still flagged risk,
     fall back to bumping the single highest-hazard cell on the path so the
     replan has at least one degree of freedom.

state["hazard_map"] is *replaced* with a fresh array. grid.hazard_map (the
persistent cross-episode memory) is NOT touched — only the per-step overlay.
"""

from typing import Any, Dict, List, Tuple

from src.agent.state import AgentState


Position = Tuple[int, int]


class PathHealerNode:
    """Bumps cost-map cells on the current planned path to force a re-route."""

    def __init__(
        self,
        heal_threshold: float = 0.4,
        bump_floor:     float = 0.95,
        fallback_bump:  float = 0.30,
    ):
        """
        heal_threshold — minimum hazard score for a waypoint to be auto-flagged.
        bump_floor      — the score every flagged cell is raised to (clipped at 1.0).
        fallback_bump   — additive bump for the single highest-hazard waypoint
                          when no cell crosses heal_threshold.
        """
        self.heal_threshold = heal_threshold
        self.bump_floor     = bump_floor
        self.fallback_bump  = fallback_bump

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        log          = state.get("episode_log", [])
        planned_path = state.get("planned_path", [])
        hazard_map   = state["hazard_map"]
        heal_count   = state.get("heal_count", 0)
        step         = state.get("step_count", 0)

        # Nothing actionable: keep state untouched, log a no-op.
        if len(planned_path) < 3:
            return {
                "episode_log": log + [{
                    "step":   step,
                    "node":   "path_healer",
                    "action": "skipped",
                    "reason": "path too short",
                }],
            }

        new_map      = hazard_map.copy()
        bumped: List[Position] = []

        # Pass 1 — auto-bump every waypoint at or above heal_threshold.
        for r, c in planned_path[1:-1]:
            if new_map[r, c] >= self.heal_threshold:
                target = max(new_map[r, c], self.bump_floor)
                if target > new_map[r, c]:
                    new_map[r, c] = min(target, 1.0)
                bumped.append((r, c))

        # Pass 2 — fallback when LLM flagged risk but no map cell crossed
        # threshold. Bump the riskiest waypoint on the path so the replan has
        # *some* signal to deviate.
        if not bumped:
            interior = planned_path[1:-1]
            if interior:
                worst_idx   = max(range(len(interior)),
                                  key=lambda i: hazard_map[interior[i][0], interior[i][1]])
                r, c        = interior[worst_idx]
                old         = float(hazard_map[r, c])
                new_map[r, c] = min(1.0, max(old + self.fallback_bump, 0.7))
                bumped.append((r, c))

        return {
            "hazard_map":  new_map,
            "heal_count":  heal_count + 1,
            "episode_log": log + [{
                "step":         step,
                "node":         "path_healer",
                "action":       "bumped",
                "n_bumped":     len(bumped),
                "bumped_cells": [list(p) for p in bumped],
                "heal_count":   heal_count + 1,
            }],
        }
