"""
AgentState — shared LangGraph state for the pathfinding agent.

A TypedDict carried between every node. `total=False` lets nodes populate
fields progressively (e.g. risk_score only exists after the evaluator runs);
all required defaults are seeded by ``init_state()``.

The proposal's schema is preserved field-for-field (grid, current_pos,
planned_path, hazard_map, risk_score, risk_justification, memory_context,
episode_log, replan_count), with a few additional fields needed to wire
LangGraph's conditional edges and the experiment runner:

  start, goal           — episode endpoints (constants)
  step_count            — for timeout detection
  obstacle_detected     — flag set by Execution Monitor → routes to Evaluator
  episode_status        — terminal state for the runner
  risk_threshold        — gate for Path Healer invocation
  max_steps             — timeout cap (typically 3× optimal path length)
  enable_memory         — ablation flag (Agent-NoMemory variant)
  enable_reflection     — ablation flag (Agent-NoReflection variant)
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

from src.environment.grid import Grid


Position       = Tuple[int, int]
EpisodeStatus  = Literal["running", "success", "collision", "timeout"]


class AgentState(TypedDict, total=False):
    """Shared state carried across every LangGraph node."""

    # ── Environment ────────────────────────────────────────────────────────
    grid:        Grid              # live Grid object; nodes call is_blocked(), step(), etc.
    start:       Position
    goal:        Position
    hazard_map:  np.ndarray        # shared reference to grid.hazard_map for explicit visibility

    # ── Episode progress ──────────────────────────────────────────────────
    current_pos:   Position
    step_count:    int
    planned_path:  List[Position]  # waypoints from the most recent plan/replan

    # ── Risk reasoning (LLM-driven) ───────────────────────────────────────
    risk_score:          float    # 0.0 (safe) → 1.0 (catastrophic)
    risk_justification:  str      # NL explanation produced by the evaluator

    # ── Memory ────────────────────────────────────────────────────────────
    memory_context:  List[Dict[str, Any]]   # retrieved past-episode summaries

    # ── Execution monitoring ──────────────────────────────────────────────
    obstacle_detected:  bool
    episode_status:     EpisodeStatus

    # ── Cost & logging ────────────────────────────────────────────────────
    replan_count:  int
    episode_log:   List[Dict[str, Any]]     # ordered, timestamped decision records
    episode_id:    str                       # set by the experiment runner per episode

    # ── Reflection output ─────────────────────────────────────────────────
    reflection_summary:  str    # NL synthesis written by the Reflection Node

    # ── Per-episode configuration ─────────────────────────────────────────
    risk_threshold:     float    # Path Healer activates above this score
    max_steps:          int
    enable_memory:      bool
    enable_reflection:  bool


# ── Construction helper ───────────────────────────────────────────────────────

DEFAULT_RISK_THRESHOLD = 0.5


def init_state(
    grid:              Grid,
    start:             Optional[Position] = None,
    goal:              Optional[Position] = None,
    risk_threshold:    float = DEFAULT_RISK_THRESHOLD,
    max_steps:         Optional[int]  = None,
    enable_memory:     bool = True,
    enable_reflection: bool = True,
) -> AgentState:
    """
    Create a fresh AgentState for a new episode with sensible defaults.

    Args:
        grid:              The live Grid object (mutated as the episode runs).
        start, goal:       Override for grid.start / grid.goal if needed.
        risk_threshold:    Path Healer activation threshold (0–1).
        max_steps:         Timeout cap. Defaults to 3 × Manhattan(start→goal).
        enable_memory:     False disables memory reads/writes (NoMemory variant).
        enable_reflection: False skips reflection synthesis (NoReflection variant).
    """
    s = start if start is not None else grid.start
    g = goal  if goal  is not None else grid.goal

    if max_steps is None:
        manhattan = abs(s[0] - g[0]) + abs(s[1] - g[1])
        max_steps = max(1, manhattan) * 3

    return AgentState(
        grid               = grid,
        start              = tuple(s),
        goal               = tuple(g),
        hazard_map         = grid.hazard_map,
        current_pos        = tuple(s),
        step_count         = 0,
        planned_path       = [],
        risk_score         = 0.0,
        risk_justification = "",
        memory_context     = [],
        obstacle_detected  = False,
        episode_status     = "running",
        replan_count       = 0,
        episode_log        = [],
        episode_id         = "",
        reflection_summary = "",
        risk_threshold     = risk_threshold,
        max_steps          = max_steps,
        enable_memory      = enable_memory,
        enable_reflection  = enable_reflection,
    )
