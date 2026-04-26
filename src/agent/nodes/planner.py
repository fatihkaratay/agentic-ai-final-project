"""
A* Planner Node (deliberative)

Wraps the AStarPlanner from Phase 3. Reads the scanner's per-step uncertainty
overlay from state["hazard_map"] as the cost weights, then:

  • Initial invocation: AStarPlanner.plan(start, goal, grid, hazards)
  • Subsequent invocations: AStarPlanner.replan(current_pos, grid, hazards)

Lifecycle: instantiate a fresh PlannerNode at the start of each episode.
The wrapped AStarPlanner caches the goal between replans; re-using one
node across episodes would mix state.
"""

from typing import Any, Dict

from src.agent.state import AgentState
from src.planners.a_star import AStarPlanner


class PlannerNode:
    """Callable LangGraph node owning an A* planner instance."""

    def __init__(self, hazard_weight: float = 2.0):
        self._planner: AStarPlanner = AStarPlanner(hazard_weight=hazard_weight)
        self._has_planned: bool     = False

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        grid    = state["grid"]
        hazards = state["hazard_map"]

        if not self._has_planned:
            result = self._planner.plan(
                state["start"], state["goal"], grid, hazards
            )
            self._has_planned = True
            new_replan_count  = state.get("replan_count", 0)
            kind = "initial"
        else:
            result = self._planner.replan(
                state["current_pos"], grid, hazards
            )
            new_replan_count = state.get("replan_count", 0) + 1
            kind = "replan"

        log_entry = {
            "step":            state.get("step_count", 0),
            "node":            "planner",
            "kind":            kind,
            "success":         result.success,
            "path_length":     len(result.path),
            "nodes_expanded":  result.nodes_expanded,
            "plan_latency_ms": round(result.plan_latency_ms, 3),
        }

        return {
            "planned_path": result.path,
            "replan_count": new_replan_count,
            "episode_log":  state.get("episode_log", []) + [log_entry],
        }
