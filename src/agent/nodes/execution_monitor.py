"""
Execution Monitor Node (reactive)

Advances the agent one step along the planned path and signals when
mid-episode replanning is required.

Operation order per call (confirmed design):
    1. Detect terminal state (already at goal).
    2. Try to move to planned_path[1] if that cell is free RIGHT NOW.
       • Free  → move, action="moved"
       • Blocked → stay, action="blocked", obstacle_detected=True
       • No usable path → stay, action="stuck"
    3. Advance the world: grid.step() (dynamic obstacles move).
    4. Re-evaluate terminal status against the post-step grid:
       • At goal                       → "success"
       • Agent's cell is now blocked   → "collision" (obstacle moved onto us)
       • step_count ≥ max_steps        → "timeout"
       • Cannot make progress (stuck)  → "timeout"
       • Otherwise                     → "running"

step_count increments every call regardless of move success — the world's
clock advances whether the agent moves or pauses to replan.
"""

from typing import Any, Dict

from src.agent.state import AgentState, EpisodeStatus


def execution_monitor_node(state: AgentState) -> Dict[str, Any]:
    grid          = state["grid"]
    current_pos   = state["current_pos"]
    goal          = state["goal"]
    planned_path  = state.get("planned_path", [])
    step_count    = state.get("step_count", 0)
    max_steps     = state.get("max_steps", 1000)
    log           = state.get("episode_log", [])

    # ── 1. Already at goal — rare, but guard so we don't waste a step ────
    if current_pos == goal:
        return {
            "episode_status":    "success",
            "obstacle_detected": False,
            "episode_log": log + [{
                "step":        step_count,
                "node":        "execution_monitor",
                "action":      "at_goal",
                "current_pos": list(current_pos),
            }],
        }

    # ── 2. Decide this step's action ─────────────────────────────────────
    path_valid = (
        len(planned_path) >= 2
        and planned_path[0] == current_pos
    )

    if path_valid:
        next_pos = planned_path[1]
        if grid.is_blocked(next_pos):
            new_pos          = current_pos
            new_planned_path = planned_path
            action           = "blocked"
        else:
            new_pos          = next_pos
            new_planned_path = planned_path[1:]   # drop the cell we just left
            action           = "moved"
    else:
        new_pos          = current_pos
        new_planned_path = planned_path
        action           = "stuck"

    new_step = step_count + 1

    # ── 3. Advance the world ─────────────────────────────────────────────
    grid.step()
    grid.agent_pos = list(new_pos)   # keep visualizer mirror in sync

    # ── 4. Terminal status against post-step grid ────────────────────────
    status: EpisodeStatus
    if new_pos == goal:
        status = "success"
    elif grid.is_blocked(new_pos):
        status = "collision"
    elif new_step >= max_steps:
        status = "timeout"
    elif action == "stuck":
        status = "timeout"
    else:
        status = "running"

    log_entry = {
        "step":   new_step,
        "node":   "execution_monitor",
        "action": action,
        "from":   list(current_pos),
        "to":     list(new_pos),
        "status": status,
    }

    return {
        "current_pos":       new_pos,
        "planned_path":      new_planned_path,
        "step_count":        new_step,
        "obstacle_detected": (action == "blocked"),
        "episode_status":    status,
        "episode_log":       log + [log_entry],
    }
