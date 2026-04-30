"""
Path Risk Evaluator Node (deliberative / LLM)

Asks an LLM to inspect the planned path against the per-step uncertainty
overlay and prior-episode memory. Output (parsed JSON):

    risk_score    — float in [0.0, 1.0]
    justification — natural-language explanation of the score

These two values populate state["risk_score"] and state["risk_justification"]
respectively. The justification is the project's primary interpretability
artifact — every routing decision the agent makes downstream of this node
has a human-readable rationale captured in the episode log.

Fail-open semantics: on any LLM failure (no API key, malformed JSON, network
error), risk_score defaults to 0.0 with justification "[LLM unavailable]",
allowing the agent to degrade gracefully to A* + hazard-map behavior.
"""

from typing import Any, Dict, Optional

from src.agent.state import AgentState
from src.environment.grid import Grid
from src.llm.client import LLMClient


SYSTEM_PROMPT = (
    "You are a strategic risk assessor for an autonomous pathfinding agent in "
    "a discrete grid environment. You will receive:\n"
    "  - a compressed description of the current environment\n"
    "  - the agent's planned path (waypoints from current position to goal)\n"
    "  - high-hazard waypoints flagged on that path (with risk scores)\n"
    "  - optional summaries of past episodes for context\n\n"
    "Evaluate whether the proposed path is safe to execute. Output a JSON object with:\n"
    '  "risk_score":    a float in [0.0, 1.0]  (0.0 = safe, 1.0 = catastrophic)\n'
    '  "justification": a concise natural-language explanation (1-3 sentences)\n\n'
    "Be specific: cite waypoint coordinates, hazard values, or past patterns. "
    "Output JSON only, no prose."
)


def _build_user_prompt(state: AgentState) -> str:
    grid:    Grid = state["grid"]
    path          = state.get("planned_path", [])
    hazards       = state["hazard_map"]

    env_summary = grid.get_state_summary(path=path)

    risky_lines = []
    for r, c in path[:25]:                          # bound prompt size
        score = float(hazards[r, c])
        if score > 0.05:
            risky_lines.append(f"  ({r},{c}): hazard={score:.2f}")
    risky_str = "\n".join(risky_lines) if risky_lines else "  (none above 0.05)"

    memory_lines = []
    for rec in (state.get("memory_context") or [])[:3]:
        summary = (rec.get("summary") or "")[:120]
        memory_lines.append(
            f"  - {rec.get('episode_id', '?')}: outcome={rec.get('outcome', '?')}, "
            f"steps={rec.get('steps_taken', '?')}, summary=\"{summary}\""
        )
    memory_str = "\n".join(memory_lines) if memory_lines else "  (no prior episodes)"

    head = path[:8]
    tail = path[-3:] if len(path) > 8 else []

    return (
        f"ENVIRONMENT\n{env_summary}\n\n"
        f"PLANNED PATH ({len(path)} waypoints)\n"
        f"  head: {head}\n"
        f"  tail: {tail}\n\n"
        f"HIGH-HAZARD WAYPOINTS ON PATH\n{risky_str}\n\n"
        f"PRIOR EPISODES\n{memory_str}\n\n"
        f"Evaluate this path."
    )


class RiskEvaluatorNode:
    """LangGraph node — invokes the LLM to score the current planned path."""

    def __init__(self, llm: Optional[Any] = None):
        # `llm` may be LLMClient (real) or MockLLMClient (tests).
        # Both must expose `call_json(system, user, fallback, ...) -> dict`.
        self._llm = llm if llm is not None else LLMClient()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        log = state.get("episode_log", [])

        # Nothing to evaluate.
        if not state.get("planned_path"):
            return {
                "risk_score":         0.0,
                "risk_justification": "No planned path to evaluate.",
                "episode_log": log + [{
                    "step":    state.get("step_count", 0),
                    "node":    "risk_evaluator",
                    "skipped": True,
                    "reason":  "empty_path",
                }],
            }

        user_prompt = _build_user_prompt(state)
        fallback    = {"risk_score": 0.0, "justification": "[LLM unavailable]"}

        result = self._llm.call_json(
            system      = SYSTEM_PROMPT,
            user        = user_prompt,
            fallback    = fallback,
            max_tokens  = 400,
            temperature = 0.2,
        )

        # ── Validate & clamp ───────────────────────────────────────────
        try:
            score = float(result.get("risk_score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))

        justification = result.get("justification", "")
        if not isinstance(justification, str):
            justification = str(justification)

        log_entry = {
            "step":          state.get("step_count", 0),
            "node":          "risk_evaluator",
            "risk_score":    score,
            "justification": justification[:240],
        }
        if "__error__" in result:
            log_entry["error"] = result["__error__"]

        return {
            "risk_score":         score,
            "risk_justification": justification,
            "episode_log":        log + [log_entry],
        }
