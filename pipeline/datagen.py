"""
Phase 1 — Synthetic training data generator + virtual-judge quality filter.

generate()         → raw examples (unfiltered)
vj_filter()        → (kept, discarded)
generate_patch()   → raw patch examples for a specific failure cluster
save_jsonl()       → write to disk
load_jsonl()       → read from disk
split_examples()   → (train_set, eval_set) stratified by tool name
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from specialists.runtime.schema import DomainSpec
    from pipeline.providers import LLMProvider
    from pipeline.cluster import FailureCluster

# Imported at call time to keep module importable without heavy deps.
# This same template is used in harness.py so train/inference are aligned.
_INFERENCE_SYSTEM_PROMPT_TEMPLATE = """\
You are a precise tool-calling assistant. Given a user request, output ONLY a \
single JSON object representing the best tool call. No explanation, no markdown — \
just the raw JSON on one line.

Available tools:
{tools_json}

Output format:
{{"name": "<tool_name>", "parameters": {{<param>: <value>, ...}}}}

Rules:
- Only include parameters explicitly mentioned or clearly implied by the request.
- Never invent or guess parameter values.
- Omit optional parameters that are not mentioned.
- If no tool matches, output: {{"name": "no_tool", "parameters": {{}}}}
"""

# Never assembled dynamically from user content. Only {tool_json}, {n}, and
# {focus_description} slots are filled at generation time.

_GENERATION_SYSTEM = """\
You are a training-data generator for a tool-calling AI assistant.

Given a set of tool definitions, produce exactly {n} realistic user queries.
Each query must map to exactly one specific tool call.

Tools:
{tools_json}

For each query, output exactly one JSON object per line (no array wrapper):
{{"query": "<natural language request>", "tool_name": "<exact tool name>", "parameters": {{<param>: <value>}}}}

Rules:
- Only include parameters that are explicitly mentioned or clearly implied by the query.
- Never invent parameter values.
- Vary phrasing, vocabulary, and sentence structure significantly across examples.
- Focus on: {focus_description}
"""

_NOTOOL_GENERATION_SYSTEM = """\
You are a training-data generator for a tool-calling AI assistant.

Given a set of tool definitions, produce exactly {n} user queries that do NOT
match any of the available tools. These queries are out-of-domain or too
ambiguous to safely resolve to a single tool call.

Tools:
{tools_json}

For each query, output exactly one JSON object per line:
{{"query": "<request no tool can handle>", "tool_name": "no_tool", "parameters": {{}}}}

Good no_tool examples:
- Requests for domains this specialist does not cover
- Requests for information none of the tools return
- Ambiguous requests that could map to multiple tools
"""

_VJ_SYSTEM = """\
You are a quality judge for tool-calling training examples.

Available tools:
{tools_json}

Score the example below from 0.0 to 1.0:
  1.0 — Perfect: natural query, correct tool, no hallucinated or missing required parameters
  0.7 — Acceptable: minor phrasing issue but correct
  0.5 — Marginal: correct tool, borderline parameters
  0.0 — Reject: wrong tool, hallucinated parameters, or incoherent query

Training example:
Query: {query}
Tool call: {tool_call_json}

Respond with ONLY a JSON object — no preamble:
{{"score": <0.0–1.0>, "reason": "<one sentence>"}}
"""

_PATCH_SYSTEM = """\
You are a training-data generator fixing a specific failure pattern in a \
tool-calling AI assistant.

Failure pattern: {cluster_description}

Tools:
{tools_json}

Produce exactly {n} training examples that directly address this failure.
Each example should be a query the model currently gets wrong, paired with
the correct answer.

For each example, output exactly one JSON object per line:
{{"query": "<query that exercises this failure>", "tool_name": "<correct tool>", "parameters": {{<correct params>}}}}
"""



@dataclass
class TrainingExample:
    messages: list[dict[str, str]]
    expected_tool_call: dict[str, Any]
    source: str = "synthetic"
    cluster_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "expected_tool_call": self.expected_tool_call,
            "source": self.source,
            "cluster_id": self.cluster_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingExample":
        return cls(
            messages=d["messages"],
            expected_tool_call=d["expected_tool_call"],
            source=d.get("source", "synthetic"),
            cluster_id=d.get("cluster_id"),
        )



def save_jsonl(examples: list[TrainingExample], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def load_jsonl(path: str | Path) -> list[TrainingExample]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    examples = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(TrainingExample.from_dict(json.loads(line)))
    return examples



def split_examples(
    examples: list[TrainingExample],
    eval_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Stratified split by tool name to ensure every tool appears in both sets.
    Returns (train_set, eval_set).
    """
    rng = random.Random(seed)

    by_tool: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        key = ex.expected_tool_call["name"]
        by_tool.setdefault(key, []).append(ex)

    train: list[TrainingExample] = []
    eval_: list[TrainingExample] = []

    for tool_examples in by_tool.values():
        shuffled = list(tool_examples)
        rng.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * eval_fraction))
        eval_.extend(shuffled[:n_eval])
        train.extend(shuffled[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_



def vj_filter(
    examples: list[TrainingExample],
    domain: "DomainSpec",
    provider: "LLMProvider",
    threshold: float = 0.7,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """
    Virtual-judge quality filter (AITL, EMNLP 2025).
    Expects to discard 15–35% of examples. Fail-safe: malformed VJ response
    discards the example rather than silently keeping it.

    Returns (kept, discarded).
    """
    tools_json = json.dumps(domain.to_prompt_list(), indent=2)
    kept: list[TrainingExample] = []
    discarded: list[TrainingExample] = []

    for ex in examples:
        query = ex.messages[-1]["content"]
        tool_call_json = json.dumps(ex.expected_tool_call)
        prompt = _VJ_SYSTEM.format(
            tools_json=tools_json,
            query=query,
            tool_call_json=tool_call_json,
        )
        try:
            raw = provider.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
                response_format="json_object",
            )
            result = json.loads(raw)
            score = float(result.get("score", 0.0))
        except Exception:
            score = 0.0  # fail-safe: discard on any error

        if score >= threshold:
            kept.append(ex)
        else:
            discarded.append(ex)

    return kept, discarded



def _build_system_message(domain: "DomainSpec") -> dict[str, str]:
    tools_json = json.dumps(domain.to_prompt_list(), indent=2)
    return {
        "role": "system",
        "content": _INFERENCE_SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_json),
    }


def _parse_generation_response(raw: str) -> list[dict[str, Any]]:
    """
    Parse provider output into a list of row dicts.
    Handles: one-JSON-per-line (preferred), JSON array fallback.
    Silently skips unparseable lines.
    """
    rows: list[dict[str, Any]] = []

    # Try JSON array first
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
            if isinstance(items, list):
                return [r for r in items if isinstance(r, dict) and "query" in r and "tool_name" in r]
        except json.JSONDecodeError:
            pass

    # Fall back to one-JSON-per-line
    for line in stripped.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict) and "query" in row and "tool_name" in row:
                rows.append(row)
        except json.JSONDecodeError:
            continue

    return rows


def _row_to_example(
    row: dict[str, Any],
    domain: "DomainSpec",
    source: str = "synthetic",
    cluster_id: str | None = None,
) -> TrainingExample:
    tools_json = json.dumps(domain.to_prompt_list(), indent=2)
    system_content = _INFERENCE_SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_json)
    return TrainingExample(
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": row["query"]},
        ],
        expected_tool_call={
            "name": row["tool_name"],
            "parameters": row.get("parameters", {}),
        },
        source=source,
        cluster_id=cluster_id,
    )


def _generate_for_tool(
    tool_name: str,
    domain: "DomainSpec",
    provider: "LLMProvider",
    n: int,
    focus: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    tools_json = json.dumps(domain.to_prompt_list(), indent=2)
    prompt = _GENERATION_SYSTEM.format(
        n=n,
        tools_json=tools_json,
        focus_description=focus,
    )
    user_msg = f"Generate {n} examples specifically for the '{tool_name}' tool."
    raw = provider.complete(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    rows = _parse_generation_response(raw)
    # Keep only rows for the requested tool
    return [r for r in rows if r.get("tool_name") == tool_name]


def _generate_no_tool(
    domain: "DomainSpec",
    provider: "LLMProvider",
    n: int,
) -> list[dict[str, Any]]:
    tools_json = json.dumps(domain.to_prompt_list(), indent=2)
    prompt = _NOTOOL_GENERATION_SYSTEM.format(n=n, tools_json=tools_json)
    raw = provider.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=2048,
    )
    rows = _parse_generation_response(raw)
    return [r for r in rows if r.get("tool_name") == "no_tool"]



def generate(
    domain: "DomainSpec",
    provider: "LLMProvider",
    n_examples: int = 200,
    seed: int = 42,
) -> list[TrainingExample]:
    """
    Generate n_examples total distributed roughly:
      60% standard calls (evenly across non-no_tool tools)
      25% edge cases
      15% no_tool negatives

    Returns raw (unfiltered) examples — caller applies vj_filter().
    """
    rng = random.Random(seed)
    real_tools = [t.name for t in domain.tools if t.name != "no_tool"]

    n_negatives = max(1, int(n_examples * 0.15))
    n_positives = n_examples - n_negatives
    n_standard = int(n_positives * 0.70)
    n_edge = n_positives - n_standard

    per_tool_standard = max(1, n_standard // len(real_tools))
    per_tool_edge = max(1, n_edge // len(real_tools))

    all_rows: list[dict[str, Any]] = []

    for tool_name in real_tools:
        all_rows += _generate_for_tool(
            tool_name, domain, provider, per_tool_standard,
            focus="clear and unambiguous requests with all required parameters explicitly stated",
            rng=rng,
        )
        all_rows += _generate_for_tool(
            tool_name, domain, provider, per_tool_edge,
            focus="indirect mentions, synonyms, and partially specified requests — "
                  "optional parameters should only appear when the query implies them",
            rng=rng,
        )

    all_rows += _generate_no_tool(domain, provider, n_negatives)

    rng.shuffle(all_rows)
    return [_row_to_example(r, domain, source="synthetic") for r in all_rows]


def generate_patch(
    cluster: "FailureCluster",
    domain: "DomainSpec",
    provider: "LLMProvider",
    n_per_cluster: int = 20,
    seed: int = 42,
) -> list[TrainingExample]:
    """
    Generate targeted examples for a specific failure cluster.
    Returns raw (unfiltered) examples — caller applies vj_filter().
    """
    tools_json = json.dumps(domain.to_prompt_list(), indent=2)
    prompt = _PATCH_SYSTEM.format(
        cluster_description=cluster.description,
        tools_json=tools_json,
        n=n_per_cluster,
    )
    raw = provider.complete(
        [{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4096,
    )
    rows = _parse_generation_response(raw)
    return [
        _row_to_example(r, domain, source="patch", cluster_id=cluster.cluster_id)
        for r in rows
    ]
