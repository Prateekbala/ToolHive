"""
Phase 4 unit tests — all run without GPU, real model, or LLM provider.

Covers:
  - SchedulerStateStore CRUD
  - RetrainScheduler trigger conditions
  - Promotion when new_score >= previous_score
  - No-promotion when new_score < previous_score
  - CPU-only guard (train_fn raises RuntimeError)
  - No eval-examples path (candidate registered but not promoted)
  - run_all across multiple specialists
  - Live monitor failure-rate calculation
  - maybe_rollback integration
  - FeedbackStore.count_since
  - Exit criteria: injected failures trigger a cycle (end-to-end mocked)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from feedback.store import (
    CriticVerdict, FeedbackEntry, FeedbackStore, SignalType,
)
from scheduler.state import SchedulerState, SchedulerStateStore
from scheduler.retrain import (
    CycleResult, RetrainConfig, RetrainScheduler, _feedback_to_training_example,
)
from scheduler.live_monitor import MonitorResult, check_live_accuracy, maybe_rollback
from specialists.registry import SpecialistEntry, SpecialistRegistry
from specialists.runtime.schema import DomainSpec, ParameterSpec, ToolSpec



def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry(
    specialist_id="inventory-v1",
    domain="inventory",
    score=0.90,
    status="active",
) -> SpecialistEntry:
    return SpecialistEntry(
        specialist_id=specialist_id,
        domain=domain,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path=f"adapters/{domain}/v1",
        tools_yaml_path="specialists/domains/inventory/tools.yaml",
        eval_score=score,
        trained_at="2026-06-21T00:00:00Z",
        status=status,
    )


def _make_domain() -> DomainSpec:
    return DomainSpec(
        domain="inventory",
        tools=[
            ToolSpec(
                name="get_inventory",
                description="Check stock",
                parameters={
                    "product_id": ParameterSpec(type="string", required=True),
                },
            ),
            ToolSpec(name="no_tool", description="", parameters={}),
        ],
    )


def _feedback_entry(
    specialist_id="inventory-v1",
    verdict=CriticVerdict.BLOCK,
) -> FeedbackEntry:
    return FeedbackEntry(
        specialist_id=specialist_id,
        sub_query="check inventory for SKU-1",
        model_output={"name": "get_inventory", "parameters": {"product_id": "SKU-1"}},
        critic_verdict=verdict,
        signal_type=SignalType.ADOPTION_DECISION,
        critic_reason="test failure",
        timestamp="2026-06-21T10:00:00+00:00",
    )


@pytest.fixture
def registry(tmp_path):
    r = SpecialistRegistry(tmp_path / "reg.db")
    r.connect()
    yield r
    r.close()


@pytest.fixture
def feedback_store(tmp_path):
    fs = FeedbackStore(tmp_path / "fb.db")
    fs.connect()
    yield fs
    fs.close()


@pytest.fixture
def state_store(tmp_path):
    ss = SchedulerStateStore(tmp_path / "state.db")
    ss.connect()
    yield ss
    ss.close()


def _make_scheduler(
    registry, feedback_store, state_store=None,
    train_fn=None, evaluate_fn=None, config=None,
) -> RetrainScheduler:
    provider = MagicMock()
    return RetrainScheduler(
        registry=registry,
        feedback_store=feedback_store,
        provider=provider,
        config=config or RetrainConfig(min_failures_to_trigger=5),
        state_store=state_store,
        train_fn=train_fn,
        evaluate_fn=evaluate_fn,
    )



class TestSchedulerStateStore:
    def test_missing_returns_none(self, state_store):
        assert state_store.get("inventory-v1") is None

    def test_save_and_get(self, state_store):
        state = SchedulerState(
            specialist_id="inventory-v1",
            last_eval_score=0.91,
            retrain_count=2,
        )
        state_store.save(state)
        loaded = state_store.get("inventory-v1")
        assert loaded is not None
        assert loaded.last_eval_score == pytest.approx(0.91)
        assert loaded.retrain_count == 2

    def test_save_overwrites(self, state_store):
        s = SchedulerState(specialist_id="x", last_eval_score=0.80)
        state_store.save(s)
        s.last_eval_score = 0.95
        state_store.save(s)
        assert state_store.get("x").last_eval_score == pytest.approx(0.95)

    def test_list_all(self, state_store):
        state_store.save(SchedulerState(specialist_id="a"))
        state_store.save(SchedulerState(specialist_id="b"))
        all_states = state_store.list_all()
        ids = {s.specialist_id for s in all_states}
        assert {"a", "b"}.issubset(ids)

    def test_default_timestamp(self, state_store):
        s = SchedulerState(specialist_id="z")
        state_store.save(s)
        loaded = state_store.get("z")
        assert loaded.last_run_timestamp == "1970-01-01T00:00:00Z"

    def test_state_to_dict(self):
        s = SchedulerState(specialist_id="x", last_eval_score=0.90, retrain_count=3)
        d = s.to_dict()
        assert d["specialist_id"] == "x"
        assert d["retrain_count"] == 3



class TestCountSince:
    def test_count_all(self, feedback_store):
        since = "2000-01-01T00:00:00"
        feedback_store.append(_feedback_entry(verdict=CriticVerdict.PASS))
        feedback_store.append(_feedback_entry(verdict=CriticVerdict.FLAG))
        feedback_store.append(_feedback_entry(verdict=CriticVerdict.BLOCK))
        assert feedback_store.count_since("inventory-v1", since) == 3

    def test_count_failures_only(self, feedback_store):
        since = "2000-01-01T00:00:00"
        feedback_store.append(_feedback_entry(verdict=CriticVerdict.PASS))
        feedback_store.append(_feedback_entry(verdict=CriticVerdict.FLAG))
        feedback_store.append(_feedback_entry(verdict=CriticVerdict.BLOCK))
        assert feedback_store.count_failures_since("inventory-v1", since) == 2

    def test_count_zero_before_timestamp(self, feedback_store):
        feedback_store.append(_feedback_entry())
        # Far future timestamp → nothing before it
        assert feedback_store.count_since("inventory-v1", "2099-01-01T00:00:00") == 0

    def test_count_zero_different_specialist(self, feedback_store):
        since = "2000-01-01T00:00:00"
        feedback_store.append(_feedback_entry(specialist_id="other-v1"))
        assert feedback_store.count_since("inventory-v1", since) == 0



class TestRetrainSchedulerTrigger:
    def test_not_triggered_when_insufficient_failures(self, registry, feedback_store):
        registry.register(_entry(status="active"))
        scheduler = _make_scheduler(registry, feedback_store,
                                    config=RetrainConfig(min_failures_to_trigger=10))
        # Insert fewer than 10 failures
        since = "2000-01-01T00:00:00"
        for _ in range(3):
            feedback_store.append(_feedback_entry())
        result = scheduler.run_cycle("inventory-v1")
        assert result.triggered is False
        assert "insufficient" in result.reason
        assert result.n_new_failures == 3

    def test_triggered_at_threshold(self, registry, feedback_store, state_store):
        registry.register(_entry(status="active"))

        # Mock train + evaluate + all the pipeline calls
        mock_train = MagicMock(return_value="/tmp/candidate")
        mock_eval = MagicMock()
        mock_eval.return_value.exact_match = 0.92

        with (
            patch("scheduler.retrain.cluster_failures", return_value=[]),
        ):
            scheduler = _make_scheduler(
                registry, feedback_store, state_store,
                train_fn=mock_train, evaluate_fn=mock_eval,
                config=RetrainConfig(min_failures_to_trigger=5),
            )
            for _ in range(5):
                feedback_store.append(_feedback_entry())
            result = scheduler.run_cycle(
                "inventory-v1",
                domain=_make_domain(),
                eval_examples=[],   # no eval → candidate but not promoted
                train_examples=[],  # no train → empty patch set
            )

        assert result.triggered is True

    def test_unknown_specialist_raises(self, registry, feedback_store):
        scheduler = _make_scheduler(registry, feedback_store)
        with pytest.raises(KeyError, match="specialist not found"):
            scheduler.run_cycle("nonexistent-v1")



class TestPromotionLogic:
    def _run_with_score(
        self, registry, feedback_store, state_store, new_score, previous_score=0.90,
    ) -> CycleResult:
        registry.register(_entry(score=previous_score, status="active"))

        # Pre-set state with known previous score
        state_store.save(SchedulerState(
            specialist_id="inventory-v1",
            last_eval_score=previous_score,
        ))

        from pipeline.datagen import TrainingExample
        fake_train_example = TrainingExample(
            messages=[{"role": "user", "content": "q"}],
            expected_tool_call={"name": "get_inventory", "parameters": {}},
            source="test",
        )

        mock_train = MagicMock(return_value="/tmp/candidate_path")
        mock_eval_result = MagicMock()
        mock_eval_result.exact_match = new_score
        mock_eval = MagicMock(return_value=mock_eval_result)

        # Patch harness loading (don't actually load a model)
        with (
            patch("scheduler.retrain.cluster_failures", return_value=[]),
            patch("scheduler.retrain.ToolCallHarness") as mock_harness_cls,
        ):
            mock_harness_cls.return_value.load.return_value = None
            scheduler = _make_scheduler(
                registry, feedback_store, state_store,
                train_fn=mock_train, evaluate_fn=mock_eval,
                config=RetrainConfig(min_failures_to_trigger=5),
            )
            for _ in range(5):
                feedback_store.append(_feedback_entry())

            result = scheduler.run_cycle(
                "inventory-v1",
                domain=_make_domain(),
                eval_examples=[fake_train_example],
                train_examples=[fake_train_example],
            )
        return result

    def test_promoted_when_score_improves(self, registry, feedback_store, state_store):
        result = self._run_with_score(registry, feedback_store, state_store,
                                       new_score=0.95, previous_score=0.90)
        assert result.promoted is True
        assert result.new_specialist_id is not None
        assert result.new_score == pytest.approx(0.95)

    def test_promoted_when_score_ties(self, registry, feedback_store, state_store):
        result = self._run_with_score(registry, feedback_store, state_store,
                                       new_score=0.90, previous_score=0.90)
        assert result.promoted is True

    def test_not_promoted_when_score_drops(self, registry, feedback_store, state_store):
        with pytest.warns(UserWarning, match="not promoted"):
            result = self._run_with_score(
                registry, feedback_store, state_store,
                new_score=0.85, previous_score=0.90,
            )
        assert result.promoted is False
        assert result.new_specialist_id is None

    def test_promotion_registers_new_version(self, registry, feedback_store, state_store):
        result = self._run_with_score(registry, feedback_store, state_store,
                                       new_score=0.95, previous_score=0.90)
        assert result.new_specialist_id is not None
        new_entry = registry.get(result.new_specialist_id)
        assert new_entry is not None
        assert new_entry.status == "active"

    def test_old_version_rolled_back_on_promotion(self, registry, feedback_store, state_store):
        result = self._run_with_score(registry, feedback_store, state_store,
                                       new_score=0.95, previous_score=0.90)
        old_entry = registry.get("inventory-v1")
        assert old_entry.status == "rolled_back"

    def test_state_updated_after_promotion(self, registry, feedback_store, state_store):
        result = self._run_with_score(registry, feedback_store, state_store,
                                       new_score=0.95, previous_score=0.90)
        state = state_store.get("inventory-v1")
        assert state is not None
        assert state.retrain_count == 1
        assert state.last_eval_score == pytest.approx(0.95)

    def test_state_updated_even_without_promotion(self, registry, feedback_store, state_store):
        with pytest.warns(UserWarning):
            result = self._run_with_score(
                registry, feedback_store, state_store,
                new_score=0.80, previous_score=0.90,
            )
        state = state_store.get("inventory-v1")
        assert state is not None
        assert state.retrain_count == 1
        # Score should NOT update when not promoted
        assert state.last_eval_score == pytest.approx(0.90)



class TestCpuOnlyGuard:
    def test_cpu_raises_no_gpu_returns_gracefully(self, registry, feedback_store):
        registry.register(_entry(status="active"))

        def cpu_train(*args, **kwargs):
            raise RuntimeError("training requires GPU — CUDA not available.")

        with patch("scheduler.retrain.cluster_failures", return_value=[]):
            scheduler = _make_scheduler(
                registry, feedback_store,
                train_fn=cpu_train,
                config=RetrainConfig(min_failures_to_trigger=5),
            )
            for _ in range(5):
                feedback_store.append(_feedback_entry())
            result = scheduler.run_cycle(
                "inventory-v1",
                domain=_make_domain(),
                train_examples=[MagicMock()],  # non-empty so train is called
            )

        assert result.triggered is True
        assert "GPU" in result.reason



class TestNoEvalExamples:
    def test_no_eval_warns_and_skips_promotion(self, registry, feedback_store, state_store):
        registry.register(_entry(status="active"))
        mock_train = MagicMock(return_value="/tmp/candidate")

        with (
            patch("scheduler.retrain.cluster_failures", return_value=[]),
            pytest.warns(UserWarning, match="no eval examples"),
        ):
            scheduler = _make_scheduler(
                registry, feedback_store, state_store,
                train_fn=mock_train,
                config=RetrainConfig(min_failures_to_trigger=5),
            )
            for _ in range(5):
                feedback_store.append(_feedback_entry())
            result = scheduler.run_cycle(
                "inventory-v1",
                domain=_make_domain(),
                train_examples=[MagicMock()],
                eval_examples=[],
            )

        assert result.triggered is True
        assert result.promoted is False
        assert result.adapter_path is not None



class TestRunAll:
    def test_run_all_no_specialists_returns_empty(self, registry, feedback_store):
        scheduler = _make_scheduler(registry, feedback_store)
        results = scheduler.run_all()
        assert results == []

    def test_run_all_runs_each_specialist(self, registry, feedback_store):
        registry.register(_entry("inventory-v1", "inventory", status="active"))
        registry.register(_entry("email-v1", "email", status="active"))
        scheduler = _make_scheduler(
            registry, feedback_store,
            config=RetrainConfig(min_failures_to_trigger=100),
        )
        results = scheduler.run_all()
        assert len(results) == 2
        # No failures inserted → both not triggered
        for r in results:
            assert r.triggered is False



class TestLiveMonitor:
    def test_no_data_no_rollback(self, feedback_store):
        since = "2099-01-01T00:00:00"
        result = check_live_accuracy("inventory-v1", feedback_store, since)
        assert result.should_rollback is False
        assert result.n_total == 0

    def test_low_failure_rate_no_rollback(self, feedback_store):
        since = "2000-01-01T00:00:00"
        for _ in range(9):
            feedback_store.append(_feedback_entry(verdict=CriticVerdict.PASS))
        feedback_store.append(_feedback_entry(verdict=CriticVerdict.FLAG))
        result = check_live_accuracy(
            "inventory-v1", feedback_store, since, rollback_threshold=0.20
        )
        # 1/10 = 10% < 20%
        assert result.should_rollback is False
        assert result.failure_rate == pytest.approx(0.10)

    def test_high_failure_rate_triggers_rollback(self, feedback_store):
        since = "2000-01-01T00:00:00"
        for _ in range(5):
            feedback_store.append(_feedback_entry(verdict=CriticVerdict.PASS))
        for _ in range(5):
            feedback_store.append(_feedback_entry(verdict=CriticVerdict.BLOCK))
        result = check_live_accuracy(
            "inventory-v1", feedback_store, since, rollback_threshold=0.20
        )
        # 5/10 = 50% > 20%
        assert result.should_rollback is True
        assert result.failure_rate == pytest.approx(0.50)

    def test_insufficient_data_no_rollback(self, feedback_store):
        since = "2000-01-01T00:00:00"
        # Only 3 entries (below 10 minimum)
        for _ in range(3):
            feedback_store.append(_feedback_entry(verdict=CriticVerdict.BLOCK))
        result = check_live_accuracy("inventory-v1", feedback_store, since)
        assert result.should_rollback is False
        assert "insufficient" in result.reason

    def test_monitor_result_fields(self, feedback_store):
        since = "2099-01-01T00:00:00"
        result = check_live_accuracy("inventory-v1", feedback_store, since)
        assert isinstance(result, MonitorResult)
        assert hasattr(result, "failure_rate")
        assert hasattr(result, "threshold")

    def test_maybe_rollback_calls_registry(self, registry, feedback_store):
        registry.register(_entry(status="active"))
        result = MonitorResult(
            specialist_id="inventory-v1",
            n_total=20, n_failures=10,
            failure_rate=0.50, threshold=0.20,
            should_rollback=True, reason="test",
        )
        did_rollback = maybe_rollback(result, registry)
        assert did_rollback is True
        assert registry.get("inventory-v1").status == "rolled_back"

    def test_maybe_rollback_no_op_when_false(self, registry, feedback_store):
        registry.register(_entry(status="active"))
        result = MonitorResult(
            specialist_id="inventory-v1",
            n_total=10, n_failures=1,
            failure_rate=0.10, threshold=0.20,
            should_rollback=False, reason="ok",
        )
        did_rollback = maybe_rollback(result, registry)
        assert did_rollback is False
        assert registry.get("inventory-v1").status == "active"



class TestFeedbackToTrainingExample:
    def test_conversion_preserves_query(self):
        fe = _feedback_entry()
        te = _feedback_to_training_example(fe)
        assert te.messages[-1]["content"] == fe.sub_query

    def test_conversion_preserves_tool_call(self):
        fe = _feedback_entry()
        te = _feedback_to_training_example(fe)
        assert te.expected_tool_call == fe.model_output

    def test_source_is_feedback(self):
        te = _feedback_to_training_example(_feedback_entry())
        assert te.source == "feedback"



class TestCycleResult:
    def test_to_dict(self):
        r = CycleResult(
            specialist_id="inventory-v1",
            triggered=True,
            n_new_failures=10,
            promoted=True,
            new_score=0.95,
            previous_score=0.90,
        )
        d = r.to_dict()
        assert d["specialist_id"] == "inventory-v1"
        assert d["promoted"] is True
        assert d["new_score"] == pytest.approx(0.95)



class TestExitCriteria:
    """
    Exit criteria from plan.md §Phase 4:
      A synthetic failure injected into production logs triggers an automatic
      retrain within one scheduled cycle, and the resulting adapter is
      auto-promoted only if it improves the eval score.
    """

    def test_injected_failures_trigger_retrain(self, registry, feedback_store, state_store):
        """Injecting min_failures_to_trigger failures → triggered=True."""
        registry.register(_entry(status="active"))
        config = RetrainConfig(min_failures_to_trigger=10)

        for _ in range(10):
            feedback_store.append(_feedback_entry())

        scheduler = _make_scheduler(registry, feedback_store, state_store, config=config)
        with patch("scheduler.retrain.cluster_failures", return_value=[]):
            result = scheduler.run_cycle(
                "inventory-v1",
                domain=_make_domain(),
                train_examples=[],
                eval_examples=[],
            )
        assert result.triggered is True

    def test_auto_promoted_when_score_improves(self, registry, feedback_store, state_store):
        """After retrain, adapter auto-promoted iff new_score >= previous_score."""
        registry.register(_entry(score=0.90, status="active"))
        state_store.save(SchedulerState(
            specialist_id="inventory-v1",
            last_eval_score=0.90,
        ))

        from pipeline.datagen import TrainingExample
        dummy_example = TrainingExample(
            messages=[{"role": "user", "content": "q"}],
            expected_tool_call={"name": "get_inventory", "parameters": {}},
            source="test",
        )

        mock_train = MagicMock(return_value="/tmp/candidate")
        mock_eval_pass = MagicMock()
        mock_eval_pass.exact_match = 0.95

        for _ in range(10):
            feedback_store.append(_feedback_entry())

        with (
            patch("scheduler.retrain.cluster_failures", return_value=[]),
            patch("scheduler.retrain.ToolCallHarness") as mock_h,
        ):
            mock_h.return_value.load.return_value = None
            scheduler = _make_scheduler(
                registry, feedback_store, state_store,
                train_fn=mock_train,
                evaluate_fn=MagicMock(return_value=mock_eval_pass),
                config=RetrainConfig(min_failures_to_trigger=10),
            )
            result = scheduler.run_cycle(
                "inventory-v1",
                domain=_make_domain(),
                train_examples=[dummy_example],
                eval_examples=[dummy_example],
            )

        assert result.triggered is True
        assert result.promoted is True
        assert result.new_specialist_id is not None

    def test_not_auto_promoted_when_score_drops(self, registry, feedback_store, state_store):
        """Adapter NOT promoted if eval_score < previous_score."""
        registry.register(_entry(score=0.95, status="active"))
        state_store.save(SchedulerState(
            specialist_id="inventory-v1",
            last_eval_score=0.95,
        ))

        from pipeline.datagen import TrainingExample
        dummy_example = TrainingExample(
            messages=[{"role": "user", "content": "q"}],
            expected_tool_call={"name": "get_inventory", "parameters": {}},
            source="test",
        )

        mock_train = MagicMock(return_value="/tmp/candidate")
        mock_eval_fail = MagicMock()
        mock_eval_fail.exact_match = 0.80  # regression!

        for _ in range(10):
            feedback_store.append(_feedback_entry())

        with (
            patch("scheduler.retrain.cluster_failures", return_value=[]),
            patch("scheduler.retrain.ToolCallHarness") as mock_h,
            pytest.warns(UserWarning, match="not promoted"),
        ):
            mock_h.return_value.load.return_value = None
            scheduler = _make_scheduler(
                registry, feedback_store, state_store,
                train_fn=mock_train,
                evaluate_fn=MagicMock(return_value=mock_eval_fail),
                config=RetrainConfig(min_failures_to_trigger=10),
            )
            result = scheduler.run_cycle(
                "inventory-v1",
                domain=_make_domain(),
                train_examples=[dummy_example],
                eval_examples=[dummy_example],
            )

        assert result.triggered is True
        assert result.promoted is False
        # Original adapter still active
        assert registry.get("inventory-v1").status == "active"
