"""
OpenAI-backed LLM client with structured-JSON output.

Reads from environment:
  LLM_PROVIDER    — must be "openai" (kept as a guard for future flexibility)
  LLM_MODEL       — default "gpt-4o-mini"
  OPENAI_API_KEY  — required for live calls

Fail mode (graceful degradation):
  Any error — missing key, network, parse failure, malformed JSON — causes
  call_json() to return the caller-supplied `fallback` dict, augmented with
  an "__error__" key so the audit trail records what went wrong.
"""

import json
import os
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

from src.llm._ssl import get_merged_ca_bundle


DEFAULT_MODEL = "gpt-4o-mini"


class LLMClient:
    """Thin wrapper around OpenAI Chat Completions with JSON-mode output."""

    def __init__(self, model: Optional[str] = None):
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider != "openai":
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. MVP only supports 'openai'."
            )
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._client: Optional[OpenAI] = None
        else:
            ca_bundle = get_merged_ca_bundle()
            if ca_bundle:
                http_client = httpx.Client(verify=ca_bundle)
                self._client = OpenAI(api_key=api_key, http_client=http_client)
            else:
                self._client = OpenAI(api_key=api_key)

        # Per-instance counters — runner reads these between episodes.
        self.call_count:   int = 0
        self.total_tokens: int = 0

    def reset_counters(self) -> None:
        self.call_count   = 0
        self.total_tokens = 0

    def call_json(
        self,
        system:      str,
        user:        str,
        fallback:    Dict[str, Any],
        max_tokens:  int   = 400,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Issue one chat completion in JSON-object mode and return the parsed dict.

        On any failure, returns ``fallback`` merged with ``__error__`` so the
        node can keep running and the failure is visible in the episode log.
        """
        if self._client is None:
            return {**fallback, "__error__": "no_api_key"}

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self.call_count += 1
            if resp.usage is not None:
                self.total_tokens += int(resp.usage.total_tokens)
            content = resp.choices[0].message.content or "{}"
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {**fallback, "__error__": f"json_decode: {e}"}
        except Exception as e:
            return {**fallback, "__error__": f"api_error: {type(e).__name__}: {e}"}
