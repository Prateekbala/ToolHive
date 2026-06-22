"""
Phase 1 — Evaluation scorer.

Runs a ToolCallHarness against a held-out test set and computes accuracy
metrics in BFCL format (Berkeley Function-Calling Leaderboard v4).

BFCL categories used in Phase 1:
  simple              — single tool call with clear parameters
  irrelevance_detection — request that should map to no_tool

Metrics:
  tool_selection_accuracy  % where the correct tool was chosen
  param_exact_match        % where all params correct + no extras (tool-correct examples only)
  hallucination_rate       % with at least one hallucinated parameter (any example)
  invalid_json_rate        % where model output could not be parsed
  exact_match              tool correct + params correct + no hallucination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from specialists.runtime.harness import ToolCallHarness
    from specialists.runtime.schema import DomainSpec
    from pipeline.datagen import TrainingExample


@dataclass
class EvalResult:
    tool_selection_accuracy: float
    param_exact_match: float
    hallucination_rate: float
    invalid_json_rate: float
    exact_match: float
    n_examples: int
    by_category: dict[str, float] = field(default_factory=dict)

    def passes_goal(self, goal: dict[str, Any]) -> bool:
        return (
            self.tool_selection_accuracy >= goal.get("tool_selection_accuracy", 0.95)
            and self.hallucination_rate < goal.get("hallucination_rate", 0.02)
        )

    def summary(self) -> str:
        return (
            f"exact_match={self.exact_match:.1%}  "
            f"tool_acc={self.tool_selection_accuracy:.1%}  "
            f"halluc={self.hallucination_rate:.1%}  "
            f"invalid_json={self.invalid_json_rate:.1%}  "
            f"n={self.n_examples}"
        )


def _assign_bfcl_category(example: "TrainingExample") -> str:
    """
    Phase 1 uses two BFCL categories:
      irrelevance_detection — expected tool is no_tool
      simple                — all other single-call examples
    """
    if example.expected_tool_call.get("name") == "no_tool":
        return "irrelevance_detection"
    return "simple"


def _score_one(
    expected: dict[str, Any],
    actual: dict[str, Any] | None,
    parse_error: bool,
    tool_map: dict[str, Any],
) -> dict[str, bool]:
    """
    Compute binary scores for a single example.

    Returns:
      tool_correct   : expected tool name == actual tool name
      params_exact   : all expected params present and matching, no extras
      hallucinated   : any parameter in actual that is not in the tool spec
      invalid_json   : parse_error is True
      exact_match    : tool_correct and params_exact and not hallucinated
    """
    if parse_error or actual is None:
        return {
            "tool_correct": False,
            "params_exact": False,
            "hallucinated": False,
            "invalid_json": True,
            "exact_match": False,
        }

    tool_correct = expected["name"] == actual.get("name")
    actual_params = actual.get("parameters", {})
    expected_params = expected.get("parameters", {})

    # Hallucination: any param in output that doesn't exist in the tool spec
    hallucinated = False
    if tool_correct and actual.get("name") in tool_map:
        tool_spec = tool_map[actual["name"]]
        hallucinated = any(k not in tool_spec.parameters for k in actual_params)

    # Param exact match: correct values for all expected params, no extras
    if tool_correct and not hallucinated:
        expected_set = set(expected_params.keys())
        actual_set = set(actual_params.keys())
        values_match = all(
            str(actual_params.get(k, "")).strip() == str(v).strip()
            for k, v in expected_params.items()
        )
        params_exact = (expected_set == actual_set) and values_match
    else:
        params_exact = False

    exact_match = tool_correct and params_exact and not hallucinated
    return {
        "tool_correct": tool_correct,
        "params_exact": params_exact,
        "hallucinated": hallucinated,
        "invalid_json": False,
        "exact_match": exact_match,
    }


def evaluate(
    harness: "ToolCallHarness",
    domain: "DomainSpec",
    test_examples: list["TrainingExample"],
) -> EvalResult:
    """
    Run harness.run() on every test example and compute aggregate metrics.
    harness.load() must have been called before this function.
    """
    if not test_examples:
        return EvalResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, {})

    tool_map = domain.tool_map()
    per_category: dict[str, list[dict[str, bool]]] = {}
    aggregated: list[dict[str, bool]] = []

    for ex in test_examples:
        query = ex.messages[-1]["content"]
        result = harness.run(domain, query)

        actual = (
            None if result["parse_error"]
            else {"name": result["name"], "parameters": result["parameters"]}
        )
        scores = _score_one(
            expected=ex.expected_tool_call,
            actual=actual,
            parse_error=result["parse_error"],
            tool_map=tool_map,
        )
        aggregated.append(scores)

        cat = _assign_bfcl_category(ex)
        per_category.setdefault(cat, []).append(scores)

    n = len(aggregated)

    def _rate(key: str) -> float:
        return sum(1 for s in aggregated if s[key]) / n

    tool_correct_scores = [s for s in aggregated if s["tool_correct"]]
    param_exact = (
        sum(1 for s in tool_correct_scores if s["params_exact"]) / len(tool_correct_scores)
        if tool_correct_scores else 0.0
    )

    by_category = {
        cat: sum(1 for s in scores if s["exact_match"]) / len(scores)
        for cat, scores in per_category.items()
    }

    return EvalResult(
        tool_selection_accuracy=_rate("tool_correct"),
        param_exact_match=param_exact,
        hallucination_rate=_rate("hallucinated"),
        invalid_json_rate=_rate("invalid_json"),
        exact_match=_rate("exact_match"),
        n_examples=n,
        by_category=by_category,
    )
