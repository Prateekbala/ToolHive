"""
Phase 4 — Retrain scheduler.

One RetrainScheduler instance drives the continuous improvement flywheel for
all specialists.  run_cycle(specialist_id) executes a single retrain attempt:

  1. Pull new failures (critic FLAG/BLOCK) from FeedbackStore since last run.
  2. Gate on min_failures_to_trigger — skip if not enough new evidence.
  3. Cluster failures → generate targeted patch examples → VJ-filter.
  4. Combine patch with existing training set; retrain LoRA adapter.
  5. Evaluate on the FULL stable held-out set (not just patch data).
  6. Promote new adapter only if eval_score >= previous eval_score.
  7. Persist scheduler state (timestamp, new score, retrain count).

train_fn and evaluate_fn are injected so the scheduler can be unit-tested
without GPU.  Defaults to pipeline.finetune.train and pipeline.eval.evaluate.

CLI:
    python -m scheduler.retrain --specialist inventory-v1 \
        --registry registry.db --feedback feedback.db \
        --state scheduler_state.db
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from specialists.registry import SpecialistEntry, SpecialistRegistry
from feedback.store import FeedbackStore, FeedbackEntry
from scheduler.state import SchedulerState, SchedulerStateStore

# Module-level imports so tests can patch scheduler.retrain.<name>
from pipeline.cluster import cluster_failures
from pipeline.datagen import generate_patch, vj_filter, load_jsonl, TrainingExample
from pipeline.eval import evaluate as _pipeline_evaluate
from specialists.runtime.harness import ToolCallHarness

if TYPE_CHECKING:
    from pipeline.providers import LLMProvider
    from pipeline.eval import EvalResult
    from specialists.runtime.schema import DomainSpec



@dataclass
class RetrainConfig:
    min_failures_to_trigger: int = 20
    n_patch_per_cluster: int = 20
    output_dir_base: str = "adapters"
    goal_overrides: dict[str, Any] = field(default_factory=dict)



@dataclass
class CycleResult:
    specialist_id: str
    triggered: bool
    n_new_failures: int = 0
    n_clusters: int = 0
    n_patch_examples: int = 0
    previous_score: float | None = None
    new_score: float | None = None
    promoted: bool = False
    new_specialist_id: str | None = None
    adapter_path: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialist_id": self.specialist_id,
            "triggered": self.triggered,
            "n_new_failures": self.n_new_failures,
            "n_clusters": self.n_clusters,
            "n_patch_examples": self.n_patch_examples,
            "previous_score": self.previous_score,
            "new_score": self.new_score,
            "promoted": self.promoted,
            "new_specialist_id": self.new_specialist_id,
            "adapter_path": self.adapter_path,
            "reason": self.reason,
        }



class RetrainScheduler:
    """
    Drives the retrain flywheel for all active specialists.

    Injected callables:
      train_fn(domain, examples, base_model, output_dir, goal) -> Path
      evaluate_fn(harness, domain, test_examples) -> EvalResult

    Pass mocks in tests; leave as None to use the real pipeline functions.
    """

    def __init__(
        self,
        registry: SpecialistRegistry,
        feedback_store: FeedbackStore,
        provider: "LLMProvider",
        config: RetrainConfig | None = None,
        state_store: SchedulerStateStore | None = None,
        train_fn: Callable | None = None,
        evaluate_fn: Callable | None = None,
    ) -> None:
        self._registry = registry
        self._feedback_store = feedback_store
        self._provider = provider
        self._config = config or RetrainConfig()
        self._state_store = state_store
        self._train_fn = train_fn
        self._evaluate_fn = evaluate_fn


    def run_cycle(
        self,
        specialist_id: str,
        domain: "DomainSpec | None" = None,
        eval_examples: "list[TrainingExample] | None" = None,
        train_examples: "list[TrainingExample] | None" = None,
    ) -> CycleResult:
        """
        Run one retrain cycle for specialist_id.

        domain, eval_examples, train_examples can be injected for testing;
        otherwise they are loaded from registry / state_store paths.
        """
        entry = self._registry.get(specialist_id)
        if entry is None:
            raise KeyError(f"specialist not found in registry: {specialist_id!r}")

        state = self._get_or_default_state(entry)

        failures = self._feedback_store.failures_since(
            specialist_id, state.last_run_timestamp
        )
        n_failures = len(failures)

        if n_failures < self._config.min_failures_to_trigger:
            return CycleResult(
                specialist_id=specialist_id,
                triggered=False,
                n_new_failures=n_failures,
                reason=(
                    f"insufficient failures: {n_failures} < "
                    f"{self._config.min_failures_to_trigger} required"
                ),
            )

        if domain is None:
            from specialists.runtime.harness import load_domain
            domain = load_domain(entry.tools_yaml_path)

        failure_examples = [_feedback_to_training_example(f) for f in failures]
        clusters = cluster_failures(failure_examples, self._provider)

        patch_examples: list[TrainingExample] = []
        for cluster in clusters:
            raw = generate_patch(cluster, domain, self._provider,
                                 n_per_cluster=self._config.n_patch_per_cluster)
            kept, _ = vj_filter(raw, domain, self._provider)
            patch_examples.extend(kept)

        if train_examples is None and state.train_data_path:
            try:
                train_examples = load_jsonl(state.train_data_path)
            except Exception:
                train_examples = []  # type: ignore[assignment]
        combined = list(train_examples or []) + patch_examples

        if not combined:
            return CycleResult(
                specialist_id=specialist_id,
                triggered=True,
                n_new_failures=n_failures,
                n_clusters=len(clusters),
                n_patch_examples=0,
                reason="no patch examples generated after VJ filter",
            )

        train_fn = self._train_fn or _default_train
        output_dir = Path(self._config.output_dir_base) / entry.domain
        output_dir.mkdir(parents=True, exist_ok=True)

        goal = {
            "num_epochs": 3,
            "seed": 42,
            **self._config.goal_overrides,
        }

        try:
            candidate_path = train_fn(domain, combined, entry.base_model, output_dir, goal)
        except RuntimeError as exc:
            if "training requires GPU" in str(exc):
                return CycleResult(
                    specialist_id=specialist_id,
                    triggered=True,
                    n_new_failures=n_failures,
                    n_clusters=len(clusters),
                    n_patch_examples=len(patch_examples),
                    reason="skipped: training requires GPU",
                )
            raise

        if eval_examples is None and state.eval_data_path:
            try:
                eval_examples = load_jsonl(state.eval_data_path)
            except Exception:
                eval_examples = []  # type: ignore[assignment]

        if not eval_examples:
            warnings.warn(
                f"[scheduler] no eval examples for {specialist_id} — "
                "registering candidate but skipping promotion",
                UserWarning,
                stacklevel=2,
            )
            self._save_state(state, increment=True)
            return CycleResult(
                specialist_id=specialist_id,
                triggered=True,
                n_new_failures=n_failures,
                n_clusters=len(clusters),
                n_patch_examples=len(patch_examples),
                adapter_path=str(candidate_path),
                reason="no eval examples — candidate registered but not promoted",
            )

        evaluate_fn = self._evaluate_fn or _default_evaluate
        harness = ToolCallHarness(
            model_name_or_path=entry.base_model,
            adapter_path=str(candidate_path),
        )
        harness.load()
        eval_result = evaluate_fn(harness, domain, eval_examples)
        new_score = eval_result.exact_match

        previous_score = state.last_eval_score
        promoted = new_score >= previous_score
        new_specialist_id: str | None = None

        if promoted:
            new_specialist_id = self._registry.next_version(entry.domain)
            now_ts = datetime.now(timezone.utc).isoformat()
            new_entry = SpecialistEntry(
                specialist_id=new_specialist_id,
                domain=entry.domain,
                base_model=entry.base_model,
                adapter_path=str(candidate_path),
                tools_yaml_path=entry.tools_yaml_path,
                eval_score=new_score,
                trained_at=now_ts,
                status="candidate",
            )
            self._registry.register(new_entry)
            self._registry.promote(new_specialist_id)
            state.last_eval_score = new_score
        else:
            warnings.warn(
                f"[scheduler] new score {new_score:.3f} < previous {previous_score:.3f} "
                f"for {specialist_id} — candidate not promoted",
                UserWarning,
                stacklevel=2,
            )

        self._save_state(state, increment=True)

        return CycleResult(
            specialist_id=specialist_id,
            triggered=True,
            n_new_failures=n_failures,
            n_clusters=len(clusters),
            n_patch_examples=len(patch_examples),
            previous_score=previous_score,
            new_score=new_score,
            promoted=promoted,
            new_specialist_id=new_specialist_id,
            adapter_path=str(candidate_path),
            reason="promoted" if promoted else "not promoted (score did not improve)",
        )

    def run_all(self, **cycle_kwargs: Any) -> list[CycleResult]:
        """Run one cycle for every active specialist in the registry."""
        results = []
        for entry in self._registry.list_active():
            try:
                result = self.run_cycle(entry.specialist_id, **cycle_kwargs)
            except Exception as exc:
                result = CycleResult(
                    specialist_id=entry.specialist_id,
                    triggered=False,
                    reason=f"error: {exc}",
                )
            results.append(result)
        return results


    def _get_or_default_state(self, entry: SpecialistEntry) -> SchedulerState:
        if self._state_store:
            state = self._state_store.get(entry.specialist_id)
            if state is not None:
                return state
        return SchedulerState(
            specialist_id=entry.specialist_id,
            last_run_timestamp="1970-01-01T00:00:00Z",
            last_eval_score=entry.eval_score,
        )

    def _save_state(self, state: SchedulerState, increment: bool = False) -> None:
        state.last_run_timestamp = datetime.now(timezone.utc).isoformat()
        if increment:
            state.retrain_count += 1
        if self._state_store:
            self._state_store.save(state)



def _default_train(domain, examples, base_model, output_dir, goal):
    from pipeline.finetune import train
    return train(domain, examples, base_model, output_dir, goal)


def _default_evaluate(harness, domain, test_examples):
    return _pipeline_evaluate(harness, domain, test_examples)



def _feedback_to_training_example(entry: FeedbackEntry) -> TrainingExample:
    """
    Convert a FeedbackEntry (from the live feedback store) into a
    TrainingExample so it can be passed to cluster_failures() and
    generate_patch().
    """
    return TrainingExample(
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": entry.sub_query},
        ],
        expected_tool_call=entry.model_output,
        source="feedback",
        cluster_id=None,
    )



def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run one retrain cycle for a specialist")
    parser.add_argument("--specialist", required=True, help="specialist_id to retrain")
    parser.add_argument("--registry", default="registry.db")
    parser.add_argument("--feedback", default="feedback.db")
    parser.add_argument("--state", default="scheduler_state.db")
    parser.add_argument("--output", default="adapters")
    parser.add_argument("--min-failures", type=int, default=20)
    args = parser.parse_args()

    from pipeline.providers import provider_from_env

    config = RetrainConfig(
        min_failures_to_trigger=args.min_failures,
        output_dir_base=args.output,
    )

    with (
        SpecialistRegistry(args.registry) as registry,
        FeedbackStore(args.feedback) as feedback_store,
        SchedulerStateStore(args.state) as state_store,
    ):
        scheduler = RetrainScheduler(
            registry=registry,
            feedback_store=feedback_store,
            provider=provider_from_env(),
            config=config,
            state_store=state_store,
        )
        result = scheduler.run_cycle(args.specialist)
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
