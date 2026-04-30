"""
LangGraph wiring for the agentic pathfinder.

Topology (matches the proposal's flow chart):

    START → scanner → planner → memory_read → risk_evaluator
                ▲                                    │
                │                              ┌─────┴─────┐
                │                              │           │
                │                       path_healer   execution_monitor
                │                              │           │
                └──────────────────────────────┘           │
                                                  ┌────────┼─────────┐
                                                  │        │         │
                                              reflection   │  risk_evaluator
                                                  │   (self-loop)    │
                                            memory_write              │
                                                  │
                                                 END

Conditional routing:
  • after risk_evaluator: heal if (risk ≥ threshold AND heal_count < max_heals),
                          otherwise execute
  • after execution_monitor:
        terminal status         → reflection
        running + obstacle      → risk_evaluator (mid-episode oversight)
        running + clear         → execution_monitor (advance one more step)

Per-episode lifecycle:
  PlannerNode and PathHealerNode hold per-episode state, so a fresh graph
  must be built for each episode. ``RECURSION_LIMIT`` allows the EM
  self-loop to run for the full step budget plus replans.
"""

from typing import Any, Optional

from langgraph.graph import END, StateGraph

from src.agent.nodes.execution_monitor import execution_monitor_node
from src.agent.nodes.memory_manager import MemoryReadNode, MemoryWriteNode
from src.agent.nodes.path_healer import PathHealerNode
from src.agent.nodes.planner import PlannerNode
from src.agent.nodes.reflection import ReflectionNode
from src.agent.nodes.risk_evaluator import RiskEvaluatorNode
from src.agent.nodes.scanner import environment_scanner_node
from src.agent.state import AgentState
from src.memory.store import EpisodicMemoryStore


# Episodes can take hundreds of graph steps (one per agent move + replan/heal
# overhead). LangGraph's default 25 is far too low.
RECURSION_LIMIT = 500


def build_agent_graph(
    llm_client:    Optional[Any] = None,
    hazard_weight: float = 2.0,
    max_heals:     int   = 3,
    store:         Optional[EpisodicMemoryStore] = None,
):
    """
    Construct and compile a fresh agent graph. Build one per episode.

    Args:
        llm_client     — must expose ``call_json(...)`` (LLMClient or MockLLMClient).
                          Shared by RiskEvaluator and Reflection nodes.
        hazard_weight  — A* cost multiplier on the per-cell hazard score.
        max_heals      — circuit breaker on heal-then-replan cycles per episode.
        store          — EpisodicMemoryStore shared across episodes.
    """
    sb = StateGraph(AgentState)

    sb.add_node("scanner",            environment_scanner_node)
    sb.add_node("planner",            PlannerNode(hazard_weight=hazard_weight))
    sb.add_node("memory_read",        MemoryReadNode(store=store))
    sb.add_node("risk_evaluator",     RiskEvaluatorNode(llm=llm_client))
    sb.add_node("path_healer",        PathHealerNode())
    sb.add_node("execution_monitor",  execution_monitor_node)
    sb.add_node("reflection",         ReflectionNode(llm=llm_client))
    sb.add_node("memory_write",       MemoryWriteNode(store=store))

    sb.set_entry_point("scanner")

    # ── Linear core ───────────────────────────────────────────────────────
    sb.add_edge("scanner",      "planner")
    sb.add_edge("planner",      "memory_read")
    sb.add_edge("memory_read",  "risk_evaluator")

    # ── Branch: heal vs execute ───────────────────────────────────────────
    def route_after_risk(state: AgentState) -> str:
        if (
            state.get("risk_score", 0.0) >= state.get("risk_threshold", 0.5)
            and state.get("heal_count", 0)  < max_heals
        ):
            return "path_healer"
        return "execution_monitor"

    sb.add_conditional_edges(
        "risk_evaluator",
        route_after_risk,
        {
            "path_healer":       "path_healer",
            "execution_monitor": "execution_monitor",
        },
    )

    # Heal loops back to the planner for a replan with bumped costs.
    sb.add_edge("path_healer", "planner")

    # ── Branch: continue / reroute / terminate ────────────────────────────
    def route_after_execution(state: AgentState) -> str:
        if state.get("episode_status", "running") != "running":
            return "reflection"
        # Stuck (empty path) — try to replan now that the world has stepped.
        # The world advanced inside EM, so dynamic obstacles may have cleared.
        if state.get("stuck_count", 0) > 0:
            return "planner"
        if state.get("obstacle_detected", False):
            return "risk_evaluator"
        return "execution_monitor"

    sb.add_conditional_edges(
        "execution_monitor",
        route_after_execution,
        {
            "reflection":        "reflection",
            "planner":           "planner",
            "risk_evaluator":    "risk_evaluator",
            "execution_monitor": "execution_monitor",
        },
    )

    # ── Closeout ──────────────────────────────────────────────────────────
    sb.add_edge("reflection",    "memory_write")
    sb.add_edge("memory_write",  END)

    return sb.compile()
