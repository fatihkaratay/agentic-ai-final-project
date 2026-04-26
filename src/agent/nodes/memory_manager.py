"""
Memory Manager Nodes (Memory)

Two LangGraph entry points share an EpisodicMemoryStore:

  • MemoryReadNode   — populates state["memory_context"] before the
                       Risk Evaluator so the LLM has prior-episode context.
  • MemoryWriteNode  — persists an episode record after the Reflection Node
                       produces its NL summary.

Hazard-map updates (the persistent cross-episode memory baked into
grid.hazard_map) are NOT performed here — that is the Reflection Node's
responsibility. This split keeps the writer purely about persistence.

Both nodes honor the `enable_memory` ablation flag — when False, read
returns empty context and write is a no-op (Agent-NoMemory variant).
"""

from typing import Any, Dict, List, Optional

import numpy as np

from src.agent.state import AgentState
from src.memory.store import EpisodicMemoryStore


# ── Default singleton store (overridable for tests / experiments) ────────────

_default_store: Optional[EpisodicMemoryStore] = None


def get_default_store() -> EpisodicMemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = EpisodicMemoryStore()
    return _default_store


# ── Read ──────────────────────────────────────────────────────────────────────

class MemoryReadNode:
    """Retrieves similar past-episode summaries into state.memory_context."""

    def __init__(
        self,
        store: Optional[EpisodicMemoryStore] = None,
        limit: int = 5,
    ):
        self._store = store if store is not None else get_default_store()
        self._limit = limit

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        log = state.get("episode_log", [])

        if not state.get("enable_memory", True):
            return {
                "memory_context": [],
                "episode_log": log + [{
                    "step": state.get("step_count", 0),
                    "node": "memory_read",
                    "skipped": True,
                }],
            }

        grid = state["grid"]
        records = self._store.query_similar(
            grid_size        = grid.size,
            obstacle_density = grid.config.obstacle_density,
            limit            = self._limit,
        )

        return {
            "memory_context": records,
            "episode_log": log + [{
                "step":        state.get("step_count", 0),
                "node":        "memory_read",
                "n_retrieved": len(records),
            }],
        }


# ── Write ─────────────────────────────────────────────────────────────────────

class MemoryWriteNode:
    """Appends one episode record to the store at episode end."""

    def __init__(self, store: Optional[EpisodicMemoryStore] = None):
        self._store = store if store is not None else get_default_store()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        log = state.get("episode_log", [])

        if not state.get("enable_memory", True):
            return {
                "episode_log": log + [{
                    "step": state.get("step_count", 0),
                    "node": "memory_write",
                    "skipped": True,
                }],
            }

        grid = state["grid"]
        record = {
            "episode_id":       state.get("episode_id", ""),
            "grid_size":        grid.size,
            "obstacle_density": grid.config.obstacle_density,
            "outcome":          state.get("episode_status", "running"),
            "steps_taken":      state.get("step_count", 0),
            "replan_count":     state.get("replan_count", 0),
            "top_hazards":      _top_hazard_cells(grid.hazard_map, grid.size, k=10),
            "summary":          state.get("reflection_summary", ""),
        }
        self._store.append(record)

        return {
            "episode_log": log + [{
                "step":     state.get("step_count", 0),
                "node":     "memory_write",
                "outcome":  record["outcome"],
                "n_top_hazards": len(record["top_hazards"]),
            }],
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _top_hazard_cells(
    hazard_map: np.ndarray,
    size:       int,
    k:          int = 10,
    min_score:  float = 0.05,
) -> List[List[Any]]:
    """
    Return the k highest-scoring cells from the persistent hazard map as
    [row, col, score] triples — list-form so JSON can encode them.
    """
    flat       = hazard_map.ravel()
    top_idx    = np.argsort(flat)[::-1][:k]
    return [
        [int(i // size), int(i % size), round(float(flat[i]), 3)]
        for i in top_idx if flat[i] > min_score
    ]
