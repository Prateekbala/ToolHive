"""
Phase 2/3 — Dispatch plan executor with critic integration.

Walks a DispatchPlan step-by-step, loads the specialist harness for each
step, executes the tool call, runs the critic verifier (Phase 3), and writes
to the feedback store.

Harnesses are cached in-process by specialist_id so repeated calls to the
same specialist within one session don't reload the model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from specialists.registry import SpecialistRegistry
    from specialists.runtime.harness import ToolCallHarness
    from specialists.runtime.schema import DomainSpec
    from router.router import DispatchPlan, DispatchStep
    from critic.verifier import CriticVerifier, CriticResult
    from feedback.store import FeedbackStore


@dataclass
class StepResult:
    specialist_id: str
    domain: str
    sub_query: str
    tool_name: str | None
    parameters: dict[str, Any]
    raw_output: str
    parse_error: bool
    schema_errors: list[str]
    latency_ms: float
    critic_verdict: str | None = None    # "pass" | "flag" | "block" | None (no critic)
    critic_reason: str = ""
    critic_layer: str = ""               # "schema" | "semantic" | "escalation"


@dataclass
class OrchestrationResult:
    plan: "DispatchPlan"
    results: list[StepResult]
    success: bool          # True iff every step parsed without error

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "plan": self.plan.to_dict(),
            "results": [
                {
                    "specialist_id": r.specialist_id,
                    "domain": r.domain,
                    "sub_query": r.sub_query,
                    "tool_name": r.tool_name,
                    "parameters": r.parameters,
                    "parse_error": r.parse_error,
                    "schema_errors": r.schema_errors,
                    "latency_ms": round(r.latency_ms, 1),
                    "critic_verdict": r.critic_verdict,
                    "critic_reason": r.critic_reason,
                    "critic_layer": r.critic_layer,
                }
                for r in self.results
            ],
        }


class Orchestrator:
    """
    Execute a DispatchPlan produced by the Router.

    Optional Phase 3 integrations:
      - critic: CriticVerifier run after each specialist step
      - feedback_store: FeedbackStore where every step result is logged

    Harnesses are loaded lazily and cached by specialist_id.
    To free GPU memory between requests, call clear_cache().
    """

    def __init__(
        self,
        registry: "SpecialistRegistry",
        base_model: str,
        critic: "CriticVerifier | None" = None,
        feedback_store: "FeedbackStore | None" = None,
        harness_factory: "Any | None" = None,
    ) -> None:
        self._registry = registry
        self._base_model = base_model
        self._critic = critic
        self._feedback_store = feedback_store
        self._harness_factory = harness_factory  # callable(entry) -> harness; None = ToolCallHarness
        self._harness_cache: dict[str, "ToolCallHarness"] = {}
        self._domain_cache: dict[str, "DomainSpec"] = {}

    def execute(self, plan: "DispatchPlan") -> OrchestrationResult:
        """
        Execute all steps in the plan sequentially.
        Steps with specialist_id="no_specialist" are returned with parse_error=True.
        If a critic is configured, each step is verified and logged.
        """
        results: list[StepResult] = []

        for step in plan.steps:
            result = self._execute_step(step)
            results.append(result)

        success = all(
            not r.parse_error and r.critic_verdict not in ("flag", "block")
            for r in results
        )
        return OrchestrationResult(plan=plan, results=results, success=success)

    def _execute_step(self, step: "DispatchStep") -> StepResult:
        if step.specialist_id == "no_specialist":
            return StepResult(
                specialist_id="no_specialist",
                domain=step.domain,
                sub_query=step.sub_query,
                tool_name=None,
                parameters={},
                raw_output="",
                parse_error=True,
                schema_errors=["no specialist found for this domain"],
                latency_ms=0.0,
            )

        entry = self._registry.get(step.specialist_id)
        if entry is None:
            return StepResult(
                specialist_id=step.specialist_id,
                domain=step.domain,
                sub_query=step.sub_query,
                tool_name=None,
                parameters={},
                raw_output="",
                parse_error=True,
                schema_errors=[f"specialist_id not found in registry: {step.specialist_id!r}"],
                latency_ms=0.0,
            )

        harness = self._get_harness(entry)
        domain = self._get_domain(entry)

        t0 = time.monotonic()
        try:
            result = harness.run(domain, step.sub_query)
        except Exception as exc:
            return StepResult(
                specialist_id=step.specialist_id,
                domain=entry.domain,
                sub_query=step.sub_query,
                tool_name=None,
                parameters={},
                raw_output="",
                parse_error=True,
                schema_errors=[f"harness error: {exc}"],
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        latency_ms = (time.monotonic() - t0) * 1000

        step_result = StepResult(
            specialist_id=step.specialist_id,
            domain=entry.domain,
            sub_query=step.sub_query,
            tool_name=result["name"],
            parameters=result["parameters"],
            raw_output=result["raw"],
            parse_error=result["parse_error"],
            schema_errors=result["schema_errors"],
            latency_ms=latency_ms,
        )

        # Phase 3: critic verification + feedback logging
        if self._critic is not None:
            domain_spec = self._get_domain(entry)
            tool_call = {"name": result["name"], "parameters": result["parameters"]}
            critic_result = self._critic.verify(
                sub_query=step.sub_query,
                tool_call=tool_call,
                domain=domain_spec,
                parse_error=result["parse_error"],
            )
            step_result.critic_verdict = critic_result.verdict.value
            step_result.critic_reason = critic_result.reason
            step_result.critic_layer = critic_result.layer

            if self._feedback_store is not None:
                self._log_feedback(step_result, critic_result)

        return step_result

    def _log_feedback(self, step_result: StepResult, critic_result: "CriticResult") -> None:
        from feedback.store import FeedbackEntry, CriticVerdict, SignalType

        # Map critic verdict + parse_error to signal type
        if step_result.parse_error:
            signal = SignalType.MISSING_KNOWLEDGE
        elif critic_result.verdict == CriticVerdict.PASS:
            signal = SignalType.ADOPTION_DECISION
        elif critic_result.layer == "schema" and "unknown parameter" in critic_result.reason:
            signal = SignalType.KNOWLEDGE_RELEVANCE
        elif critic_result.verdict == CriticVerdict.BLOCK:
            signal = SignalType.MISSING_KNOWLEDGE
        else:
            signal = SignalType.PAIRWISE_PREFERENCE

        entry = FeedbackEntry(
            specialist_id=step_result.specialist_id,
            sub_query=step_result.sub_query,
            model_output={
                "name": step_result.tool_name,
                "parameters": step_result.parameters,
            },
            critic_verdict=critic_result.verdict,
            critic_reason=critic_result.reason,
            signal_type=signal,
        )
        try:
            self._feedback_store.append(entry)  # type: ignore[union-attr]
        except Exception:
            pass  # never let logging failures break the request path

    def _get_harness(self, entry: Any) -> "ToolCallHarness":
        if entry.specialist_id not in self._harness_cache:
            if self._harness_factory is not None:
                h = self._harness_factory(entry)
            else:
                from specialists.runtime.harness import ToolCallHarness
                h = ToolCallHarness(
                    model_name_or_path=self._base_model,
                    adapter_path=entry.adapter_path,
                )
            h.load()
            self._harness_cache[entry.specialist_id] = h
        return self._harness_cache[entry.specialist_id]

    def _get_domain(self, entry: Any) -> "DomainSpec":
        if entry.domain not in self._domain_cache:
            from specialists.runtime.harness import load_domain
            self._domain_cache[entry.domain] = load_domain(entry.tools_yaml_path)
        return self._domain_cache[entry.domain]

    def clear_cache(self) -> None:
        """Release all cached harnesses (frees GPU memory)."""
        self._harness_cache.clear()
        self._domain_cache.clear()
