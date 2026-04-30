"""
Reflection Node (reflective)

Runs once at episode end. When enabled, asks the LLM to:
  1. Diagnose what went right or wrong this episode
  2. Identify cells the agent should remember as persistently risky

Outputs:
  state["reflection_summary"]   — NL diagnosis (consumed by MemoryWriteNode)
  grid.hazard_map (mutated)     — cross-episode memory updated with
                                  observed + LLM-flagged hazards

Ablation behaviour:
  enable_memory     = False  → skip entirely (NoMemory variant)
  enable_reflection = False  → no LLM call; static template summary; hazard
                               map still updated from observed events only
                               (NoReflection — distinguishes raw memory
                               accumulation from synthesised reflection)
"""

from typing import Any, Dict, List, Optional, Tuple

from src.agent.state import AgentState
from src.environment.grid import Grid
from src.llm.client import LLMClient


Position = Tuple[int, int]


SYSTEM_PROMPT = (
    "You are a reflective analyst for an autonomous pathfinding agent. "
    "Given a completed episode's outcome and decision history, produce a "
    "JSON object with two fields:\n"
    '  "summary":      a 2-3 sentence diagnosis of what went right or wrong\n'
    '  "hazard_cells": a list of [row, col] coordinates the agent should '
    "treat as persistently risky in future episodes\n\n"
    "Base hazard_cells on visible patterns (repeated blocking, collision "
    "sites, near-misses). Output JSON only, no prose."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _observed_hazards(state: AgentState) -> List[Position]:
    """Cells that should be bumped from raw episode events.

    For MVP, the final dynamic-obstacle snapshot is the primary signal.
    A collision adds the agent's final cell.
    """
    grid: Grid = state["grid"]
    cells: List[Position] = list(grid._dynamic_positions)
    if state.get("episode_status") == "collision":
        cells.append(tuple(state["current_pos"]))
    return cells


def _build_user_prompt(state: AgentState) -> str:
    grid: Grid = state["grid"]

    risk_history = [
        round(float(e["risk_score"]), 2)
        for e in state.get("episode_log", [])
        if e.get("node") == "risk_evaluator" and "risk_score" in e
    ]
    risk_str = ", ".join(map(str, risk_history)) if risk_history else "(none)"

    return (
        f"EPISODE SUMMARY\n"
        f"  outcome:      {state.get('episode_status', '?')}\n"
        f"  steps_taken:  {state.get('step_count', 0)}\n"
        f"  replan_count: {state.get('replan_count', 0)}\n"
        f"  heal_count:   {state.get('heal_count', 0)}\n"
        f"  start:        {state.get('start')}\n"
        f"  goal:         {state.get('goal')}\n"
        f"  final_pos:    {state.get('current_pos')}\n\n"
        f"RISK SCORE HISTORY (predictions over time)\n  {risk_str}\n\n"
        f"FINAL GRID STATE\n{grid.get_state_summary()}\n\n"
        f"Diagnose this episode and list cells to remember as risky."
    )


def _parse_llm_cells(raw: Any, grid_size: int) -> List[Position]:
    """Defensive parsing of [[r, c], ...] from LLM output."""
    if not isinstance(raw, list):
        return []
    out: List[Position] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            r, c = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if 0 <= r < grid_size and 0 <= c < grid_size:
            out.append((r, c))
    return out


# ── Node ─────────────────────────────────────────────────────────────────────

class ReflectionNode:
    """End-of-episode synthesis and hazard-map update."""

    def __init__(self, llm: Optional[Any] = None, llm_bump: float = 0.20):
        self._llm     = llm if llm is not None else LLMClient()
        self.llm_bump = llm_bump   # extra increment for LLM-suggested cells

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        log  = state.get("episode_log", [])
        step = state.get("step_count", 0)

        # ── NoMemory ablation: skip entirely ──────────────────────────────
        if not state.get("enable_memory", True):
            return {
                "reflection_summary": "",
                "episode_log": log + [{
                    "step":    step,
                    "node":    "reflection",
                    "skipped": True,
                    "reason":  "enable_memory=False",
                }],
            }

        grid: Grid = state["grid"]
        observed   = _observed_hazards(state)

        # ── NoReflection ablation: static template, still update map ─────
        if not state.get("enable_reflection", True):
            grid.update_hazard_map(observed)
            outcome  = state.get("episode_status", "?")
            template = (
                f"[no-reflection] Episode ended in {outcome} after "
                f"{step} steps. Observed {len(observed)} hazard cells."
            )
            return {
                "reflection_summary": template,
                "episode_log": log + [{
                    "step":               step,
                    "node":               "reflection",
                    "mode":               "template",
                    "n_observed_hazards": len(observed),
                    "summary":            template,
                }],
            }

        # ── Full reflection: LLM call ─────────────────────────────────────
        fallback = {
            "summary":      f"[LLM unavailable] Episode {state.get('episode_status', '?')} "
                            f"after {step} steps.",
            "hazard_cells": [],
        }
        result = self._llm.call_json(
            system      = SYSTEM_PROMPT,
            user        = _build_user_prompt(state),
            fallback    = fallback,
            max_tokens  = 400,
            temperature = 0.3,
        )

        summary = result.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary)

        llm_cells = _parse_llm_cells(result.get("hazard_cells"), grid.size)

        # Bake into the persistent hazard map (cross-episode memory).
        grid.update_hazard_map(observed)
        if llm_cells:
            grid.update_hazard_map(llm_cells, increment=self.llm_bump, decay=1.0)

        log_entry: Dict[str, Any] = {
            "step":               step,
            "node":               "reflection",
            "mode":               "llm",
            "n_observed_hazards": len(observed),
            "n_llm_hazards":      len(llm_cells),
            "summary":            summary[:240],
        }
        if "__error__" in result:
            log_entry["error"] = result["__error__"]

        return {
            "reflection_summary": summary,
            "episode_log":        log + [log_entry],
        }
