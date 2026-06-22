from __future__ import annotations

import json
import re
from typing import Any


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """
    Extract a JSON tool call from raw model output.

    Tries, in order:
      1. Fenced code block: ```json { ... } ```
      2. First {...} JSON object in the text
    Returns None if no valid tool call dict (with a "name" key) is found.
    """
    text = text.strip()

    # 1. Fenced code block (```json or ``` alone)
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block:
        result = _try_parse(block.group(1))
        if result is not None:
            return result

    # 2. First JSON object anywhere in the output
    obj = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    if obj:
        result = _try_parse(obj.group(0))
        if result is not None:
            return result

    return None


def _try_parse(s: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(s.strip())
        if isinstance(parsed, dict) and "name" in parsed:
            parsed.setdefault("parameters", {})
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def validate_against_spec(
    tool_call: dict[str, Any],
    tool_map: dict[str, Any],
) -> list[str]:
    """
    Return a list of validation errors for a parsed tool call against a DomainSpec.
    Empty list means the call is valid.
    Used by the critic agent in Phase 3.
    """
    errors: list[str] = []
    name = tool_call.get("name")
    if name not in tool_map:
        errors.append(f"unknown tool '{name}'")
        return errors

    tool = tool_map[name]
    params = tool_call.get("parameters", {})

    for req in tool.required_params():
        if req not in params:
            errors.append(f"missing required parameter '{req}'")

    for key in params:
        if key not in tool.parameters:
            errors.append(f"unknown parameter '{key}' (hallucination risk)")

    return errors
