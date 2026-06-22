"""
Phase 1 — Training loop orchestrator.

Full cycle: datagen → VJ filter → split → finetune → eval → [cluster → patch → finetune]* → done

Entry point:
  python -m pipeline.loop \\
    --domain specialists/domains/inventory \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --output specialists/domains/inventory/adapters

Provider credentials read from env:
  TOOLHIVE_PROVIDER_API_KEY, TOOLHIVE_PROVIDER_MODEL, TOOLHIVE_PROVIDER_BASE_URL

The same loop handles both cold-start training (synthetic data) and
ongoing retraining (production failures) — the only difference is the data source,
which is the feedback store's responsibility in Phase 4.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from specialists.runtime.schema import DomainSpec
    from specialists.runtime.harness import ToolCallHarness
    from pipeline.providers import LLMProvider
    from pipeline.datagen import TrainingExample


@dataclass
class LoopResult:
    promoted: bool
    final_eval_score: float
    iterations: int
    adapter_path: Path | None
    history: list[dict[str, Any]] = field(default_factory=list)



# (pattern, default, kind)  kind="pct" divides by 100, kind="int" keeps as int
_GOAL_PATTERNS: dict[str, tuple[str, Any, str]] = {
    "tool_selection_accuracy": (r"(?:accuracy|target).*?(\d+\.?\d*)%", 0.95, "pct"),
    "hallucination_rate": (r"hallucin.*?(\d+\.?\d*)%", 0.02, "pct"),
    "max_iterations": (r"max.*?iter.*?(\d+)", 5, "int"),
    "adapter_rank": (r"rank.*?(\d+)", 16, "int"),
}


def load_goal(goal_path: str | Path) -> dict[str, Any]:
    """
    Parse goal.md into a config dict using regex.
    Logs a warning and uses defaults for any key not found.
    """
    text = Path(goal_path).read_text()
    result: dict[str, Any] = {}
    for key, (pattern, default, kind) in _GOAL_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            result[key] = float(raw) / 100 if kind == "pct" else int(raw)
        else:
            warnings.warn(f"[goal] '{key}' not found in goal.md — using default {default}")
            result[key] = default
    return result



def _is_correct(
    example: "TrainingExample",
    harness: "ToolCallHarness",
    domain: "DomainSpec",
) -> bool:
    query = example.messages[-1]["content"]
    result = harness.run(domain, query)
    if result["parse_error"]:
        return False
    return (
        result["name"] == example.expected_tool_call["name"]
        and result["parameters"] == example.expected_tool_call.get("parameters", {})
    )


def _load_harness(base_model: str, adapter_path: Path | None) -> "ToolCallHarness":
    from specialists.runtime.harness import ToolCallHarness
    adapter = str(adapter_path) if adapter_path else None
    h = ToolCallHarness(base_model, adapter_path=adapter)
    h.load()
    return h



def run_loop(
    domain: "DomainSpec",
    goal: dict[str, Any],
    base_model: str,
    provider: "LLMProvider",
    output_dir: str | Path,
    n_examples: int = 200,
    n_patch_per_cluster: int = 20,
    eval_fraction: float = 0.15,
    seed: int = 42,
) -> LoopResult:
    """
    Full datagen → finetune → eval → patch loop.

    CPU-only mode: if finetune raises RuntimeError("requires GPU"), training
    is skipped but datagen and eval still run against the base model.
    """
    from pipeline.datagen import generate, vj_filter, split_examples, save_jsonl
    from pipeline.finetune import train
    from pipeline.eval import evaluate
    from pipeline.cluster import cluster_failures
    from pipeline.datagen import generate_patch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_iterations = goal.get("max_iterations", 5)

    print(f"[loop] generating {n_examples} examples for domain '{domain.domain}'")
    raw = generate(domain, provider, n_examples=n_examples, seed=seed)
    kept, discarded = vj_filter(raw, domain, provider)
    print(f"[loop] VJ filter: kept {len(kept)}/{len(raw)} ({len(discarded)} discarded)")

    save_jsonl(kept, output_dir / "data" / f"{domain.domain}_train_v0.jsonl")

    train_set, eval_set = split_examples(kept, eval_fraction=eval_fraction, seed=seed)
    print(f"[loop] split: {len(train_set)} train / {len(eval_set)} eval")

    best_score = 0.0
    best_adapter: Path | None = None
    history: list[dict[str, Any]] = []

    gpu_available = True
    adapter_path: Path | None = None

    for iteration in range(max_iterations):
        print(f"\n[loop] ── iteration {iteration} ──────────────────────────────")

        if gpu_available:
            try:
                adapter_path = train(domain, train_set, base_model, output_dir / "adapters", goal)
                print(f"[loop] trained adapter: {adapter_path}")
            except RuntimeError as e:
                if "requires GPU" in str(e):
                    print(f"[loop] WARNING: {e} — running eval on base model only")
                    gpu_available = False
                    adapter_path = None
                else:
                    raise

        harness = _load_harness(base_model, adapter_path)
        result = evaluate(harness, domain, eval_set)
        print(f"[loop] eval: {result.summary()}")

        entry: dict[str, Any] = {
            "iteration": iteration,
            "n_train": len(train_set),
            "n_eval": len(eval_set),
            "adapter_path": str(adapter_path) if adapter_path else None,
            "eval": {
                "tool_selection_accuracy": result.tool_selection_accuracy,
                "param_exact_match": result.param_exact_match,
                "hallucination_rate": result.hallucination_rate,
                "invalid_json_rate": result.invalid_json_rate,
                "exact_match": result.exact_match,
                "by_category": result.by_category,
            },
        }

        if result.exact_match > best_score:
            best_score = result.exact_match
            best_adapter = adapter_path

        if result.passes_goal(goal):
            print(f"[loop] goal reached at iteration {iteration}!")
            entry["goal_reached"] = True
            history.append(entry)
            _write_history(output_dir, history)
            return LoopResult(
                promoted=True,
                final_eval_score=result.exact_match,
                iterations=iteration + 1,
                adapter_path=best_adapter,
                history=history,
            )

        # Find failures and generate patch data
        failures = [ex for ex in eval_set if not _is_correct(ex, harness, domain)]
        print(f"[loop] {len(failures)} failures — clustering")

        clusters = cluster_failures(failures, provider)
        print(f"[loop] {len(clusters)} cluster(s): {[c.cluster_id for c in clusters]}")

        n_added = 0
        for cluster in clusters:
            raw_patch = generate_patch(cluster, domain, provider, n_patch_per_cluster)
            kept_patch, _ = vj_filter(raw_patch, domain, provider)
            train_set.extend(kept_patch)
            n_added += len(kept_patch)
            print(f"[loop]   {cluster.cluster_id} ({cluster.description!r}): +{len(kept_patch)} examples")

        entry["n_clusters"] = len(clusters)
        entry["n_patch_added"] = n_added
        history.append(entry)

        if not gpu_available:
            print("[loop] no GPU — stopping after first eval iteration")
            break

    _write_history(output_dir, history)
    return LoopResult(
        promoted=False,
        final_eval_score=best_score,
        iterations=len(history),
        adapter_path=best_adapter,
        history=history,
    )


def _write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    path = output_dir / "loop_history.json"
    path.write_text(json.dumps(history, indent=2))



def main() -> None:
    parser = argparse.ArgumentParser(
        description="ToolHive Phase 1 — autonomous specialist training loop"
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Path to domain directory containing tools.yaml and goal.md",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace base model name or local path (e.g. Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for adapters and data (default: <domain>/output)",
    )
    parser.add_argument("--examples", type=int, default=200, help="Number of training examples to generate")
    parser.add_argument("--patch-per-cluster", type=int, default=20, help="Patch examples per failure cluster")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from specialists.runtime.schema import DomainSpec
    from pipeline.providers import provider_from_env
    import yaml

    domain_dir = Path(args.domain)
    tools_path = domain_dir / "tools.yaml"
    goal_path = domain_dir / "goal.md"

    if not tools_path.exists():
        print(f"[error] tools.yaml not found at {tools_path}", file=sys.stderr)
        sys.exit(1)
    if not goal_path.exists():
        print(f"[error] goal.md not found at {goal_path}", file=sys.stderr)
        sys.exit(1)

    with tools_path.open() as f:
        domain = DomainSpec.model_validate(yaml.safe_load(f))

    goal = load_goal(goal_path)
    output_dir = Path(args.output) if args.output else domain_dir / "output"
    provider = provider_from_env()

    print(f"[loop] domain={domain.domain}  model={args.model}  output={output_dir}")
    print(f"[loop] goal: {goal}")

    result = run_loop(
        domain=domain,
        goal=goal,
        base_model=args.model,
        provider=provider,
        output_dir=output_dir,
        n_examples=args.examples,
        n_patch_per_cluster=args.patch_per_cluster,
        seed=args.seed,
    )

    print("\n[loop] ── result ─────────────────────────────────────────────")
    print(f"  promoted:    {result.promoted}")
    print(f"  best score:  {result.final_eval_score:.1%}")
    print(f"  iterations:  {result.iterations}")
    print(f"  adapter:     {result.adapter_path}")
    sys.exit(0 if result.promoted else 1)


if __name__ == "__main__":
    main()
