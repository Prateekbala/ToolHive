"""
Phase 3 — Critic bias calibration.

Architecture.md §2.3 mandates that the critic be validated against a labeled
gold set (≥100 known-good and known-bad outputs) BEFORE its verdicts feed
the retrain pipeline.  Minimum bar: precision ≥ 85%.

Calibration also checks for systematic bias across 14+ documented LLM-judge
failure modes (JudgeSense benchmark, 2025).  Any bias category whose
error_rate exceeds _BIAS_WARN_THRESHOLD triggers a warning in the report.

Usage:
    from critic.calibration import GoldExample, calibrate

    gold = [
        GoldExample(sub_query="...", tool_call={...}, expected_verdict=CriticVerdict.PASS),
        GoldExample(sub_query="...", tool_call={...}, expected_verdict=CriticVerdict.BLOCK,
                    bias_categories=["hallucination"]),
        ...
    ]
    report = calibrate(critic, domain, gold_examples=gold)
    if not report.passes_threshold:
        raise RuntimeError("Critic precision below 85% — do not enable auto-retraining")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from feedback.store import CriticVerdict

if TYPE_CHECKING:
    from critic.verifier import CriticVerifier
    from specialists.runtime.schema import DomainSpec

# Precision threshold from architecture.md §2.3
_PRECISION_THRESHOLD = 0.85

# Bias categories documented by JudgeSense (2025) and related work
KNOWN_BIAS_CATEGORIES: list[str] = [
    "verbosity_bias",          # longer outputs rated better regardless of quality
    "position_bias",           # earlier options in a comparison rated higher
    "self_enhancement_bias",   # model rates its own output style higher
    "authority_bias",          # confident-sounding outputs rated higher
    "sentiment_bias",          # positive-sentiment outputs rated higher
    "format_bias",             # markdown/formatted outputs rated higher
    "length_penalty_bias",     # very short outputs penalised regardless of correctness
    "name_bias",               # outputs attributed to prestigious names rated higher
    "familiarity_bias",        # outputs matching training distribution rated higher
    "hallucination_tolerance", # plausible-sounding hallucinations not caught
    "required_param_miss",     # missing required params not consistently caught
    "type_mismatch_miss",      # type errors not consistently caught
    "enum_violation_miss",     # enum violations not consistently caught
    "wrong_tool_miss",         # wrong tool selection not consistently caught
]

_BIAS_WARN_THRESHOLD = 0.20  # warn if a bias category error rate exceeds 20%


@dataclass
class GoldExample:
    """A labeled example for calibration."""
    sub_query: str
    tool_call: dict                      # {"name": ..., "parameters": {...}}
    expected_verdict: CriticVerdict      # PASS or FLAG/BLOCK (BLOCK treated as "bad")
    parse_error: bool = False
    bias_categories: list[str] = field(default_factory=list)
    description: str = ""                # human note for debugging


@dataclass
class CalibrationReport:
    """
    Precision/recall of the critic on a labeled gold set.

    A "positive" is defined as an example whose expected_verdict is FLAG
    or BLOCK (i.e. a bad output the critic should catch).

    precision = TP / (TP + FP)  — of critic's flags, how many were real problems
    recall    = TP / (TP + FN)  — of real problems, how many did the critic catch
    """
    n_examples: int
    n_true_positives: int     # expected bad, critic flagged/blocked
    n_false_positives: int    # expected good, critic flagged/blocked
    n_false_negatives: int    # expected bad, critic passed
    n_true_negatives: int     # expected good, critic passed
    precision: float
    recall: float
    f1: float
    passes_threshold: bool    # precision >= _PRECISION_THRESHOLD
    bias_breakdown: dict[str, float]   # bias_category -> error_rate (0.0–1.0)
    errors: list[dict]                 # mis-predicted examples (for debugging)

    def summary(self) -> str:
        lines = [
            f"Calibration report — {self.n_examples} examples",
            f"  Precision : {self.precision:.1%}  {'✓' if self.passes_threshold else '✗ (below 85%)'}",
            f"  Recall    : {self.recall:.1%}",
            f"  F1        : {self.f1:.1%}",
            f"  TP/FP/FN/TN: {self.n_true_positives}/{self.n_false_positives}"
            f"/{self.n_false_negatives}/{self.n_true_negatives}",
        ]
        if self.bias_breakdown:
            high = [(cat, rate) for cat, rate in self.bias_breakdown.items()
                    if rate > _BIAS_WARN_THRESHOLD]
            if high:
                lines.append("  High-error bias categories (>20%):")
                for cat, rate in sorted(high, key=lambda x: -x[1]):
                    lines.append(f"    {cat}: {rate:.1%}")
        return "\n".join(lines)


def calibrate(
    critic: "CriticVerifier",
    domain: "DomainSpec",
    gold_examples: list[GoldExample],
) -> CalibrationReport:
    """
    Run the critic on every gold example and compute precision/recall.

    Args:
        critic: A fully initialised CriticVerifier (with or without provider).
        domain: The DomainSpec for the specialist being calibrated.
        gold_examples: Labeled examples with expected verdicts.

    Returns:
        CalibrationReport.  Call report.passes_threshold before enabling
        auto-retraining.
    """
    if len(gold_examples) < 10:
        warnings.warn(
            f"[calibration] gold set has only {len(gold_examples)} examples "
            f"(≥100 recommended); results may be unreliable",
            UserWarning,
            stacklevel=2,
        )

    tp = fp = fn = tn = 0
    errors: list[dict] = []
    bias_counts: dict[str, dict[str, int]] = {
        cat: {"total": 0, "errors": 0} for cat in KNOWN_BIAS_CATEGORIES
    }

    for ex in gold_examples:
        result = critic.verify(
            sub_query=ex.sub_query,
            tool_call=ex.tool_call,
            domain=domain,
            parse_error=ex.parse_error,
        )

        expected_bad = ex.expected_verdict in (CriticVerdict.FLAG, CriticVerdict.BLOCK)
        got_bad = result.verdict in (CriticVerdict.FLAG, CriticVerdict.BLOCK)

        if expected_bad and got_bad:
            tp += 1
        elif expected_bad and not got_bad:
            fn += 1
            errors.append({
                "type": "false_negative",
                "sub_query": ex.sub_query,
                "expected": ex.expected_verdict.value,
                "got": result.verdict.value,
                "critic_reason": result.reason,
                "description": ex.description,
            })
        elif not expected_bad and got_bad:
            fp += 1
            errors.append({
                "type": "false_positive",
                "sub_query": ex.sub_query,
                "expected": ex.expected_verdict.value,
                "got": result.verdict.value,
                "critic_reason": result.reason,
                "description": ex.description,
            })
        else:  # not expected_bad and not got_bad
            tn += 1

        # Bias breakdown: count errors per annotated bias category
        for cat in ex.bias_categories:
            if cat in bias_counts:
                bias_counts[cat]["total"] += 1
                is_error = (expected_bad and not got_bad) or (not expected_bad and got_bad)
                if is_error:
                    bias_counts[cat]["errors"] += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    passes = precision >= _PRECISION_THRESHOLD

    bias_breakdown = {
        cat: counts["errors"] / counts["total"]
        for cat, counts in bias_counts.items()
        if counts["total"] > 0
    }

    if not passes:
        warnings.warn(
            f"[calibration] Critic precision {precision:.1%} is below the 85% threshold "
            f"— do not enable auto-retraining until this is resolved. "
            f"Review false-positive examples in the report.",
            UserWarning,
            stacklevel=2,
        )

    return CalibrationReport(
        n_examples=len(gold_examples),
        n_true_positives=tp,
        n_false_positives=fp,
        n_false_negatives=fn,
        n_true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        passes_threshold=passes,
        bias_breakdown=bias_breakdown,
        errors=errors,
    )


def build_injection_gold_set(domain: "DomainSpec") -> list[GoldExample]:
    """
    Build a synthetic gold set by injecting known defects into the domain's
    tools.  Used for the exit-criteria test harness (≥90% recall target).

    Each tool gets three injected bad examples:
      - missing required parameter  (→ BLOCK)
      - wrong type on a param       (→ FLAG)
      - completely wrong tool       (→ BLOCK)
    And one good example            (→ PASS)
    """
    gold: list[GoldExample] = []
    tool_names = [t.name for t in domain.tools if t.name != "no_tool"]

    for tool in domain.tools:
        if tool.name == "no_tool":
            continue

        # Build a valid call
        valid_params: dict = {}
        for param_name, param_spec in tool.parameters.items():
            if param_spec.type == "string":
                valid_params[param_name] = (
                    param_spec.enum[0] if param_spec.enum else "test_value"
                )
            elif param_spec.type == "integer":
                valid_params[param_name] = 1
            elif param_spec.type == "number":
                valid_params[param_name] = 1.0
            elif param_spec.type == "boolean":
                valid_params[param_name] = True
            elif param_spec.type == "array":
                valid_params[param_name] = []
            elif param_spec.type == "object":
                valid_params[param_name] = {}

        # Good example
        gold.append(GoldExample(
            sub_query=f"use {tool.name}",
            tool_call={"name": tool.name, "parameters": valid_params},
            expected_verdict=CriticVerdict.PASS,
            description=f"valid call to {tool.name}",
        ))

        # Bad: missing required param
        required = tool.required_params()
        if required:
            missing_params = {k: v for k, v in valid_params.items() if k != required[0]}
            gold.append(GoldExample(
                sub_query=f"use {tool.name} without {required[0]}",
                tool_call={"name": tool.name, "parameters": missing_params},
                expected_verdict=CriticVerdict.BLOCK,
                bias_categories=["required_param_miss"],
                description=f"missing required param {required[0]!r} for {tool.name}",
            ))

        # Bad: type mismatch
        if tool.parameters:
            first_param_name = next(iter(tool.parameters))
            first_param = tool.parameters[first_param_name]
            if first_param.type == "string":
                bad_value: Any = 42
            elif first_param.type in ("integer", "number"):
                bad_value = "not_a_number"
            elif first_param.type == "boolean":
                bad_value = "true"  # string instead of bool
            else:
                bad_value = None  # null

            bad_type_params = {**valid_params, first_param_name: bad_value}
            gold.append(GoldExample(
                sub_query=f"use {tool.name} with bad type",
                tool_call={"name": tool.name, "parameters": bad_type_params},
                expected_verdict=CriticVerdict.FLAG,
                bias_categories=["type_mismatch_miss"],
                description=f"type mismatch on {first_param_name!r} for {tool.name}",
            ))

        # Bad: parse error
        gold.append(GoldExample(
            sub_query=f"use {tool.name} (garbled output)",
            tool_call={},
            expected_verdict=CriticVerdict.BLOCK,
            parse_error=True,
            description=f"parse error for {tool.name}",
        ))

        # Bad: wrong tool — pick any other tool
        other_tools = [n for n in tool_names if n != tool.name]
        if other_tools:
            gold.append(GoldExample(
                sub_query=f"use {tool.name}",
                tool_call={"name": other_tools[0], "parameters": {}},
                expected_verdict=CriticVerdict.BLOCK,
                bias_categories=["wrong_tool_miss"],
                description=f"wrong tool: {other_tools[0]!r} instead of {tool.name!r}",
            ))

    return gold
