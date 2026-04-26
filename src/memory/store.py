"""
Episodic Memory Store
JSON-Lines backed log of past episodes — append-only, line-streamable.

One JSON object per line. Schema (set by the writer in memory_manager):

    episode_id        — str (set by the experiment runner)
    timestamp         — ISO-8601 UTC, stamped at write time if absent
    grid_size         — int
    obstacle_density  — float
    outcome           — "success" | "collision" | "timeout"
    steps_taken       — int
    replan_count      — int
    top_hazards       — list of [row, col, score] for high-risk cells
    summary           — NL string produced by the Reflection Node

Retrieval is intentionally simple for the MVP: filter by grid_size and
return the most recent K records. Vector / semantic retrieval can be
swapped in for the conference extension.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DEFAULT_STORE_PATH = "results/logs/episode_memory.jsonl"


class EpisodicMemoryStore:
    """Append-only JSONL store with simple recency-based retrieval."""

    def __init__(self, path: str = DEFAULT_STORE_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ── Mutations ─────────────────────────────────────────────────────────

    def append(self, record: Dict[str, Any]) -> None:
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def clear(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)

    # ── Reads ─────────────────────────────────────────────────────────────

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue                    # skip malformed lines silently
        return out

    def query_similar(
        self,
        grid_size:        int,
        obstacle_density: Optional[float] = None,
        density_tol:      float = 0.05,
        limit:            int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return up to `limit` most-recent records matching the criteria.

        Records are ordered most-recent-first in the returned list so the
        risk evaluator sees fresh context at the top of its prompt.
        """
        records = [r for r in self.read_all() if r.get("grid_size") == grid_size]
        if obstacle_density is not None:
            records = [
                r for r in records
                if abs(r.get("obstacle_density", -1.0) - obstacle_density) <= density_tol
            ]
        return records[-limit:][::-1]

    def __len__(self) -> int:
        return len(self.read_all())
