"""
No-GPU inference harness that delegates to an LLM provider API.

Drop-in replacement for ToolCallHarness when:
  - No GPU / local model is available (quickstart mode)
  - An adapter has not yet been trained for a new domain

The system prompt is intentionally identical to ToolCallHarness so that
responses from the provider match what the trained adapters will eventually
produce, making AITL feedback and critic labels consistent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .harness import _SYSTEM_PROMPT
from .parser import parse_tool_call, validate_against_spec

if TYPE_CHECKING:
    from pipeline.providers import LLMProvider
    from .schema import DomainSpec


class ProviderHarness:
    """
    Implements the same run(domain, query) interface as ToolCallHarness.

    load() is a no-op — the provider is always ready.
    """

    def __init__(self, provider: "LLMProvider") -> None:
        self._provider = provider

    def load(self) -> None:
        pass

    def run(
        self,
        domain: "DomainSpec",
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Generate a tool call via the provider API.

        Returns the same dict as ToolCallHarness.run():
          name, parameters, raw, parse_error, schema_errors
        """
        tools_json = json.dumps(domain.to_prompt_list(), indent=2)
        system_content = _SYSTEM_PROMPT.format(tools_json=tools_json)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        try:
            raw = self._provider.complete(
                messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                response_format="json_object",
            )
        except Exception as exc:
            return {
                "name": None,
                "parameters": {},
                "raw": f"<provider_error: {exc}>",
                "parse_error": True,
                "schema_errors": [],
            }

        tool_call = parse_tool_call(raw)
        parse_error = tool_call is None

        if parse_error:
            return {
                "name": None,
                "parameters": {},
                "raw": raw,
                "parse_error": True,
                "schema_errors": [],
            }

        schema_errors = validate_against_spec(tool_call, domain.tool_map())
        return {
            "name": tool_call.get("name"),
            "parameters": tool_call.get("parameters", {}),
            "raw": raw,
            "parse_error": False,
            "schema_errors": schema_errors,
        }
