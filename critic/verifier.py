"""
Phase 3 — Critic / verifier agent.

Three-layer pipeline per architecture.md §2.3:

  Layer 1  Schema validation  (pure Python, no LLM — fast, deterministic)
  Layer 2  Semantic plausibility  (LLM call via provider)
  Layer 3  Optional escalation  (second-opinion from a larger model for FLAGs)

All prompt templates are pinned constants at module level.  They are NEVER
dynamically assembled — even small phrasing changes cause verdict flip rates
of 0.4–0.99 (JudgeSense 2025) and corrupt training-signal labels.

Usage:
    from critic.verifier import CriticVerifier, CriticResult
    from feedback.store import CriticVerdict

    critic = CriticVerifier(provider=my_provider)
    result = critic.verify(
        sub_query="reserve 5 units of SKU-1",
        tool_call={"name": "reserve_stock", "parameters": {"product_id": "SKU-1", "quantity": 5}},
        domain=domain_spec,
    )
    # result.verdict in (CriticVerdict.PASS, CriticVerdict.FLAG, CriticVerdict.BLOCK)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from feedback.store import CriticVerdict

if TYPE_CHECKING:
    from pipeline.providers import LLMProvider
    from specialists.runtime.schema import DomainSpec, ToolSpec

# Never modify these strings without re-running calibration.  Each template
# is versioned by its position in this file; CI will fail if the hash changes
# without a corresponding calibration re-run (enforced in Phase 4 CI step).

_CRITIC_SEMANTIC_SYSTEM = """\
You are a precision tool-call verifier. Evaluate whether a structured tool
call correctly satisfies a user query given the tool's JSON schema.

Rules (apply them uniformly — do not let output length, tone, or style
influence your verdict):
  1. Judge the tool NAME first. If the wrong tool was selected, verdict is
     "block" regardless of parameter quality.
  2. For required parameters: missing → "block", wrong type → "flag".
  3. For optional parameters: unexpected extra params that are not in the
     schema at all → "flag" (hallucination).
  4. If parameters are plausible but you are uncertain, prefer "flag" over
     "block".
  5. Verbosity does not affect quality. A short correct response equals a long
     correct response.

Respond with JSON only — no prose outside the JSON object:
{
  "verdict": "pass" | "flag" | "block",
  "reason": "<one sentence>",
  "corrected_parameters": { ... }   // optional: only include when fixable
}\
"""

_CRITIC_ESCALATION_SYSTEM = """\
You are a senior tool-call auditor providing a second opinion. A junior
verifier flagged the following tool call. Your job is to independently
determine whether the flag was justified.

If you agree the output is problematic, respond with "block".
If you believe the output is actually acceptable, respond with "pass".
Do NOT respond with "flag" — you must resolve the ambiguity.

Respond with JSON only:
{
  "verdict": "pass" | "block",
  "reason": "<one sentence>"
}\
"""


import re as _re

_PII_PATTERNS: list[tuple[_re.Pattern[str], str]] = [
    (_re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", _re.IGNORECASE), "[EMAIL]"),
    (_re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (_re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (_re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"), "[CC]"),
    (_re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
]


def _scrub_pii(text: str) -> str:
    """Replace common PII patterns with redaction tokens before sending to external LLM."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text



@dataclass
class CriticResult:
    verdict: CriticVerdict
    reason: str
    layer: str                        # "schema" | "semantic" | "escalation"
    corrected_output: dict | None = None
    confidence: float = 1.0           # 0.0–1.0; lower for escalation cases

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reason": self.reason,
            "layer": self.layer,
            "corrected_output": self.corrected_output,
            "confidence": round(self.confidence, 3),
        }



class CriticVerifier:
    """
    Three-layer critic.  Layers short-circuit: BLOCK from schema validation
    never reaches the LLM.  Escalation only fires on FLAG verdicts when an
    escalation_provider is configured.
    """

    def __init__(
        self,
        provider: "LLMProvider | None" = None,
        escalation_provider: "LLMProvider | None" = None,
    ) -> None:
        self._provider = provider
        self._escalation_provider = escalation_provider

    def verify(
        self,
        sub_query: str,
        tool_call: dict[str, Any],
        domain: "DomainSpec",
        parse_error: bool = False,
    ) -> CriticResult:
        """
        Run all applicable critic layers and return the final CriticResult.

        Args:
            sub_query: The original user sub-query sent to the specialist.
            tool_call: The specialist's structured output
                       {"name": ..., "parameters": {...}}.
            domain: The domain spec (tools.yaml) for the specialist.
            parse_error: True if the specialist's output couldn't be parsed
                         as JSON (fast-path BLOCK without LLM).
        """
        # Layer 1: schema
        schema_result = _check_schema(tool_call, domain, parse_error)
        if schema_result.verdict == CriticVerdict.BLOCK:
            return schema_result

        # Layer 2: semantic (requires provider)
        if self._provider is None:
            # No LLM configured — trust schema validation result
            return schema_result

        semantic_result = self._check_semantic(sub_query, tool_call, domain)

        # Layer 3: escalation (only for FLAG)
        if (
            semantic_result.verdict == CriticVerdict.FLAG
            and self._escalation_provider is not None
        ):
            return self._escalate(sub_query, tool_call, domain, semantic_result)

        return semantic_result


    def _check_semantic(
        self,
        sub_query: str,
        tool_call: dict[str, Any],
        domain: "DomainSpec",
    ) -> CriticResult:
        tool_name = tool_call.get("name", "")
        tool_spec = domain.tool_map().get(tool_name)
        schema_json = json.dumps(tool_spec.to_prompt_dict(), indent=2) if tool_spec else "{}"

        scrubbed_query = _scrub_pii(sub_query)
        call_json = json.dumps(tool_call, indent=2)

        user_msg = (
            f"Query: {scrubbed_query}\n\n"
            f"Tool schema:\n{schema_json}\n\n"
            f"Specialist output:\n{call_json}\n\n"
            "Verdict?"
        )

        try:
            raw = self._provider.complete(  # type: ignore[union-attr]
                messages=[
                    {"role": "system", "content": _CRITIC_SEMANTIC_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            parsed = json.loads(raw)
        except Exception as exc:
            # Fail open — if LLM or parse error, defer to schema result
            return CriticResult(
                verdict=CriticVerdict.FLAG,
                reason=f"semantic check error (fail-open): {exc}",
                layer="semantic",
                confidence=0.0,
            )

        verdict_str = parsed.get("verdict", "flag").lower()
        verdict = CriticVerdict(verdict_str) if verdict_str in ("pass", "flag", "block") else CriticVerdict.FLAG
        corrected = parsed.get("corrected_parameters")

        return CriticResult(
            verdict=verdict,
            reason=parsed.get("reason", ""),
            layer="semantic",
            corrected_output={"name": tool_name, "parameters": corrected} if corrected else None,
            confidence=0.85,
        )

    def _escalate(
        self,
        sub_query: str,
        tool_call: dict[str, Any],
        domain: "DomainSpec",
        initial_result: CriticResult,
    ) -> CriticResult:
        tool_name = tool_call.get("name", "")
        tool_spec = domain.tool_map().get(tool_name)
        schema_json = json.dumps(tool_spec.to_prompt_dict(), indent=2) if tool_spec else "{}"

        user_msg = (
            f"Query: {_scrub_pii(sub_query)}\n\n"
            f"Tool schema:\n{schema_json}\n\n"
            f"Specialist output:\n{json.dumps(tool_call, indent=2)}\n\n"
            f"Junior verifier flagged this with reason: {initial_result.reason!r}\n\n"
            "Do you agree? Resolve to pass or block."
        )

        try:
            raw = self._escalation_provider.complete(  # type: ignore[union-attr]
                messages=[
                    {"role": "system", "content": _CRITIC_ESCALATION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=128,
            )
            parsed = json.loads(raw)
            verdict_str = parsed.get("verdict", "block").lower()
            verdict = CriticVerdict.BLOCK if verdict_str == "block" else CriticVerdict.PASS
            reason = parsed.get("reason", initial_result.reason)
        except Exception:
            # Escalation error — keep original FLAG
            return CriticResult(
                verdict=CriticVerdict.FLAG,
                reason=initial_result.reason,
                layer="escalation",
                confidence=0.5,
            )

        return CriticResult(
            verdict=verdict,
            reason=reason,
            layer="escalation",
            corrected_output=initial_result.corrected_output,
            confidence=0.95,
        )



_TYPE_VALIDATORS: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": (int, float),  # type: ignore[dict-item]
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _check_schema(
    tool_call: dict[str, Any],
    domain: "DomainSpec",
    parse_error: bool,
) -> CriticResult:
    """
    Layer 1: deterministic schema validation.

    Returns BLOCK on fatal structural errors, FLAG on type/enum mismatches,
    PASS when everything checks out.
    """
    if parse_error:
        return CriticResult(
            verdict=CriticVerdict.BLOCK,
            reason="specialist output could not be parsed as JSON",
            layer="schema",
        )

    tool_name = tool_call.get("name")
    if not tool_name:
        return CriticResult(
            verdict=CriticVerdict.BLOCK,
            reason="tool_call missing 'name' field",
            layer="schema",
        )

    tool_map = domain.tool_map()
    if tool_name == "no_tool":
        # no_tool is always structurally valid
        return CriticResult(verdict=CriticVerdict.PASS, reason="no_tool selected", layer="schema")

    if tool_name not in tool_map:
        return CriticResult(
            verdict=CriticVerdict.BLOCK,
            reason=f"unknown tool name: {tool_name!r}",
            layer="schema",
        )

    spec: "ToolSpec" = tool_map[tool_name]
    params: dict[str, Any] = tool_call.get("parameters") or {}

    # Check required params
    missing = [k for k in spec.required_params() if k not in params]
    if missing:
        return CriticResult(
            verdict=CriticVerdict.BLOCK,
            reason=f"required parameter(s) missing: {', '.join(missing)}",
            layer="schema",
        )

    # Check types and enum constraints
    flags: list[str] = []
    for param_name, value in params.items():
        param_spec = spec.parameters.get(param_name)
        if param_spec is None:
            flags.append(f"unknown parameter: {param_name!r} (possible hallucination)")
            continue
        expected_type = param_spec.type
        validator = _TYPE_VALIDATORS.get(expected_type)
        if validator and not isinstance(value, validator):  # type: ignore[arg-type]
            flags.append(f"parameter {param_name!r}: expected {expected_type}, got {type(value).__name__}")
        if param_spec.enum and value not in param_spec.enum:
            flags.append(f"parameter {param_name!r}: value {value!r} not in enum {param_spec.enum}")

    if flags:
        return CriticResult(
            verdict=CriticVerdict.FLAG,
            reason="; ".join(flags),
            layer="schema",
        )

    return CriticResult(verdict=CriticVerdict.PASS, reason="schema valid", layer="schema")
