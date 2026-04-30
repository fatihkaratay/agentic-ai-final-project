"""
MockLLMClient — deterministic substitute for LLMClient in tests.

Records every call in ``self.calls`` so tests can assert on prompt content.
Two ways to control responses:
  • Constructor ``default_response`` — returned for every call
  • ``queue(response)`` — FIFO override per call (next call pops one)
"""

from typing import Any, Dict, List, Optional


class MockLLMClient:
    def __init__(self, default_response: Optional[Dict[str, Any]] = None):
        self.default: Dict[str, Any] = (
            default_response
            if default_response is not None
            else {"risk_score": 0.0, "justification": "[mock] default response"}
        )
        self._queued: List[Dict[str, Any]] = []
        self.calls:    List[Dict[str, Any]] = []
        # Same surface as LLMClient so the runner is provider-agnostic.
        self.call_count:   int = 0
        self.total_tokens: int = 0

    def reset_counters(self) -> None:
        self.call_count   = 0
        self.total_tokens = 0

    def queue(self, response: Dict[str, Any]) -> None:
        """Append a response to be returned on the next call."""
        self._queued.append(response)

    def call_json(
        self,
        system:      str,
        user:        str,
        fallback:    Dict[str, Any],
        max_tokens:  int   = 400,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        self.call_count   += 1
        # Mock token usage so the runner has non-zero data to aggregate in tests.
        self.total_tokens += len(system) + len(user)
        if self._queued:
            return self._queued.pop(0)
        return dict(self.default)
