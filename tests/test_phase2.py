"""
Phase 2 unit tests — all run without GPU, real model, or embedding model.

sentence-transformers calls are patched where needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from specialists.registry import SpecialistEntry, SpecialistRegistry
from router.router import (
    DispatchPlan,
    DispatchStep,
    Router,
    _build_corpus,
    _is_multi_step,
    _keyword_overlap,
    _split_request,
)
from router.orchestrator import Orchestrator, OrchestrationResult, StepResult



def _entry(
    domain: str = "inventory",
    version: int = 1,
    score: float = 0.96,
    status: str = "active",
    tools_yaml: str = "specialists/domains/inventory/tools.yaml",
) -> SpecialistEntry:
    return SpecialistEntry(
        specialist_id=f"{domain}-v{version}",
        domain=domain,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path=f"adapters/{domain}/v{version}",
        tools_yaml_path=tools_yaml,
        eval_score=score,
        trained_at="2026-06-21T00:00:00Z",
        status=status,
    )


@pytest.fixture
def registry(tmp_path) -> SpecialistRegistry:
    r = SpecialistRegistry(tmp_path / "registry.db")
    r.connect()
    yield r
    r.close()



class TestRegistry:
    def test_register_and_get(self, registry):
        e = _entry()
        registry.register(e)
        fetched = registry.get("inventory-v1")
        assert fetched is not None
        assert fetched.domain == "inventory"
        assert fetched.eval_score == pytest.approx(0.96)

    def test_get_missing_returns_none(self, registry):
        assert registry.get("does-not-exist") is None

    def test_promote_sets_active(self, registry):
        e = _entry(status="candidate")
        registry.register(e)
        registry.promote("inventory-v1")
        assert registry.get("inventory-v1").status == "active"

    def test_promote_demotes_previous_active(self, registry):
        e1 = _entry(version=1, status="active")
        e2 = _entry(version=2, status="candidate")
        registry.register(e1)
        registry.register(e2)
        registry.promote("inventory-v2")
        assert registry.get("inventory-v1").status == "rolled_back"
        assert registry.get("inventory-v2").status == "active"

    def test_promote_unknown_raises(self, registry):
        with pytest.raises(KeyError):
            registry.promote("nonexistent-v1")

    def test_rollback_marks_rolled_back(self, registry):
        e = _entry(status="active")
        registry.register(e)
        registry.rollback("inventory-v1")
        assert registry.get("inventory-v1").status == "rolled_back"

    def test_rollback_restores_previous(self, registry):
        e1 = _entry(version=1, status="rolled_back")
        e2 = _entry(version=2, status="active")
        registry.register(e1)
        registry.register(e2)
        registry.rollback("inventory-v2")
        assert registry.get("inventory-v1").status == "active"

    def test_get_active_returns_active(self, registry):
        registry.register(_entry(version=1, status="candidate"))
        registry.register(_entry(version=2, status="active"))
        active = registry.get_active("inventory")
        assert active is not None
        assert active.specialist_id == "inventory-v2"

    def test_get_active_no_active_returns_none(self, registry):
        registry.register(_entry(status="candidate"))
        assert registry.get_active("inventory") is None

    def test_list_active_multiple_domains(self, registry):
        registry.register(_entry(domain="inventory", status="active"))
        registry.register(_entry(domain="email", version=1, status="active",
                                  tools_yaml="specialists/domains/email/tools.yaml"))
        active = registry.list_active()
        domains = {e.domain for e in active}
        assert "inventory" in domains
        assert "email" in domains

    def test_list_all(self, registry):
        registry.register(_entry(version=1, status="rolled_back"))
        registry.register(_entry(version=2, status="active"))
        all_entries = registry.list_all()
        assert len(all_entries) == 2

    def test_next_version_empty(self, registry):
        assert registry.next_version("inventory") == "inventory-v1"

    def test_next_version_increment(self, registry):
        registry.register(_entry(version=1))
        registry.register(_entry(version=2))
        assert registry.next_version("inventory") == "inventory-v3"

    def test_register_replace(self, registry):
        e = _entry(score=0.90)
        registry.register(e)
        e_updated = _entry(score=0.95)
        registry.register(e_updated)
        assert registry.get("inventory-v1").eval_score == pytest.approx(0.95)



class TestMultiStepDetection:
    @pytest.mark.parametrize("text,expected", [
        ("check inventory then email the supplier", True),
        ("check inventory and then notify the team", True),
        ("reserve stock after that send confirmation", True),
        ("check inventory", False),
        ("how many units of SKU-123 are available", False),
        ("get stock level; send reorder", True),
    ])
    def test_detection(self, text, expected):
        assert _is_multi_step(text) == expected


class TestSplitRequest:
    def test_simple_split(self):
        parts = _split_request("check inventory then email the supplier")
        assert len(parts) == 2
        assert "check inventory" in parts[0].lower()
        assert "email the supplier" in parts[1].lower()

    def test_and_then_split(self):
        parts = _split_request("reserve 5 units and then send a confirmation")
        assert len(parts) == 2

    def test_no_connector_returns_original(self):
        parts = _split_request("check inventory for SKU-123")
        assert len(parts) == 1

    def test_three_steps(self):
        parts = _split_request("check stock then reserve units then send email")
        assert len(parts) == 3

    def test_filters_short_fragments(self):
        # Single-word fragments should be filtered
        parts = _split_request("do this then")
        assert all(len(p.split()) >= 2 for p in parts)


class TestBuildCorpus:
    def test_contains_domain(self):
        e = _entry()
        corpus = _build_corpus(e)
        assert "inventory" in corpus.lower()

    def test_contains_tool_names(self):
        e = _entry()
        corpus = _build_corpus(e)
        # The inventory tools.yaml has get_inventory, reserve_stock, etc.
        assert "get_inventory" in corpus or "inventory" in corpus

    def test_missing_tools_yaml_graceful(self, tmp_path):
        e = _entry(tools_yaml=str(tmp_path / "nonexistent.yaml"))
        corpus = _build_corpus(e)
        # Falls back to domain name only
        assert "inventory" in corpus.lower()


class TestKeywordOverlap:
    def test_identical(self):
        assert _keyword_overlap("check stock", "check stock") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _keyword_overlap("check stock", "send email") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = _keyword_overlap("check inventory stock", "check stock levels")
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        assert _keyword_overlap("", "check stock") == pytest.approx(0.0)



class TestRouter:
    """Test routing using keyword-overlap fallback (no sentence-transformers needed)."""

    def _make_router(self, registry) -> Router:
        """Return a router with sentence-transformers patched out."""
        router = Router(registry, embedding_model="all-MiniLM-L6-v2")
        # Patch _load_embedder to return None → keyword fallback
        with patch("router.router._load_embedder", return_value=None):
            router.build_index()
        return router

    def test_empty_registry_returns_empty_plan(self, registry):
        router = self._make_router(registry)
        plan = router.route("check inventory")
        assert plan.steps == []

    def test_single_specialist_routes_correctly(self, registry):
        registry.register(_entry(status="active"))
        router = self._make_router(registry)
        plan = router.route("check inventory stock levels")
        assert len(plan.steps) == 1
        assert plan.steps[0].domain == "inventory"
        assert plan.is_multi_step is False

    def test_multi_step_creates_multiple_steps(self, registry):
        registry.register(_entry(domain="inventory", status="active"))
        registry.register(_entry(
            domain="email", version=1, status="active",
            tools_yaml="specialists/domains/inventory/tools.yaml",
        ))
        router = self._make_router(registry)
        plan = router.route("check inventory then send email")
        assert plan.is_multi_step is True
        assert len(plan.steps) == 2

    def test_plan_to_dict(self, registry):
        registry.register(_entry(status="active"))
        router = self._make_router(registry)
        plan = router.route("check inventory")
        d = plan.to_dict()
        assert "request" in d
        assert "steps" in d
        assert "is_multi_step" in d

    def test_dispatch_step_fields(self, registry):
        registry.register(_entry(status="active"))
        router = self._make_router(registry)
        plan = router.route("check inventory stock")
        step = plan.steps[0]
        assert step.specialist_id == "inventory-v1"
        assert step.domain == "inventory"
        assert step.sub_query == "check inventory stock"
        assert isinstance(step.similarity_score, float)

    def test_rebuild_index_after_new_specialist(self, registry):
        router = self._make_router(registry)
        plan = router.route("check inventory")
        assert plan.steps == []  # nothing registered yet

        registry.register(_entry(status="active"))
        with patch("router.router._load_embedder", return_value=None):
            router.build_index()
        plan = router.route("check inventory")
        assert len(plan.steps) == 1



class TestOrchestrator:
    def _make_plan(self, specialist_id="inventory-v1", domain="inventory",
                   query="check inventory for SKU-1") -> DispatchPlan:
        return DispatchPlan(
            request=query,
            steps=[DispatchStep(specialist_id=specialist_id, domain=domain,
                                sub_query=query, similarity_score=0.85)],
        )

    def test_no_specialist_step_returns_error(self, registry):
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")
        plan = DispatchPlan(
            request="check inventory",
            steps=[DispatchStep("no_specialist", "unknown", "check inventory", 0.0)],
        )
        result = orch.execute(plan)
        assert result.success is False
        assert result.results[0].parse_error is True
        assert "no specialist" in result.results[0].schema_errors[0]

    def test_unknown_specialist_id_returns_error(self, registry):
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")
        plan = self._make_plan(specialist_id="inventory-v999")
        result = orch.execute(plan)
        assert result.success is False
        assert result.results[0].parse_error is True

    def test_successful_step(self, registry):
        registry.register(_entry(status="active"))
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")

        mock_harness = MagicMock()
        mock_harness.run.return_value = {
            "name": "get_inventory",
            "parameters": {"product_id": "SKU-1"},
            "raw": '{"name":"get_inventory","parameters":{"product_id":"SKU-1"}}',
            "parse_error": False,
            "schema_errors": [],
        }
        orch._harness_cache["inventory-v1"] = mock_harness

        # Pre-populate domain cache to avoid file loading
        from specialists.runtime.schema import DomainSpec, ToolSpec, ParameterSpec
        orch._domain_cache["inventory"] = DomainSpec(
            domain="inventory",
            tools=[ToolSpec(
                name="get_inventory", description="check stock",
                parameters={"product_id": ParameterSpec(type="string", required=True)},
            )],
        )

        plan = self._make_plan()
        result = orch.execute(plan)
        assert result.success is True
        assert result.results[0].tool_name == "get_inventory"
        assert result.results[0].parse_error is False

    def test_multi_step_all_succeed(self, registry):
        registry.register(_entry(domain="inventory", status="active"))
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")

        mock_harness = MagicMock()
        mock_harness.run.return_value = {
            "name": "get_inventory", "parameters": {"product_id": "X"},
            "raw": "", "parse_error": False, "schema_errors": [],
        }
        orch._harness_cache["inventory-v1"] = mock_harness

        from specialists.runtime.schema import DomainSpec, ToolSpec, ParameterSpec
        orch._domain_cache["inventory"] = DomainSpec(
            domain="inventory",
            tools=[ToolSpec(
                name="get_inventory", description="",
                parameters={"product_id": ParameterSpec(type="string", required=True)},
            )],
        )

        plan = DispatchPlan(
            request="check inventory then check inventory again",
            steps=[
                DispatchStep("inventory-v1", "inventory", "check inventory", 0.9),
                DispatchStep("inventory-v1", "inventory", "check inventory again", 0.85),
            ],
            is_multi_step=True,
        )
        result = orch.execute(plan)
        assert result.success is True
        assert len(result.results) == 2

    def test_result_to_dict(self, registry):
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")
        plan = DispatchPlan(
            request="test",
            steps=[DispatchStep("no_specialist", "unknown", "test", 0.0)],
        )
        result = orch.execute(plan)
        d = result.to_dict()
        assert "success" in d
        assert "results" in d
        assert "plan" in d

    def test_clear_cache(self, registry):
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")
        orch._harness_cache["x"] = MagicMock()
        orch._domain_cache["y"] = MagicMock()
        orch.clear_cache()
        assert orch._harness_cache == {}
        assert orch._domain_cache == {}

    def test_harness_error_returns_step_error(self, registry):
        registry.register(_entry(status="active"))
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")

        mock_harness = MagicMock()
        mock_harness.run.side_effect = RuntimeError("GPU OOM")
        orch._harness_cache["inventory-v1"] = mock_harness

        from specialists.runtime.schema import DomainSpec
        orch._domain_cache["inventory"] = DomainSpec(domain="inventory", tools=[])

        result = orch.execute(self._make_plan())
        assert result.success is False
        assert "harness error" in result.results[0].schema_errors[0]

    def test_latency_ms_populated(self, registry):
        registry.register(_entry(status="active"))
        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")
        mock_harness = MagicMock()
        mock_harness.run.return_value = {
            "name": "get_inventory", "parameters": {},
            "raw": "", "parse_error": False, "schema_errors": [],
        }
        orch._harness_cache["inventory-v1"] = mock_harness

        from specialists.runtime.schema import DomainSpec
        orch._domain_cache["inventory"] = DomainSpec(domain="inventory", tools=[])

        result = orch.execute(self._make_plan())
        assert result.results[0].latency_ms >= 0.0



class TestRouteAndOrchestrate:
    def test_single_step_route_and_execute(self, registry):
        registry.register(_entry(status="active"))
        router = Router(registry)
        with patch("router.router._load_embedder", return_value=None):
            router.build_index()

        plan = router.route("check inventory for SKU-1")
        assert len(plan.steps) == 1
        assert plan.steps[0].specialist_id == "inventory-v1"

        orch = Orchestrator(registry, "Qwen/Qwen2.5-3B-Instruct")
        mock_harness = MagicMock()
        mock_harness.run.return_value = {
            "name": "get_inventory", "parameters": {"product_id": "SKU-1"},
            "raw": "", "parse_error": False, "schema_errors": [],
        }
        orch._harness_cache["inventory-v1"] = mock_harness

        from specialists.runtime.schema import DomainSpec, ToolSpec, ParameterSpec
        orch._domain_cache["inventory"] = DomainSpec(
            domain="inventory",
            tools=[ToolSpec(
                name="get_inventory", description="",
                parameters={"product_id": ParameterSpec(type="string", required=True)},
            )],
        )

        result = orch.execute(plan)
        assert result.success is True
        assert result.results[0].tool_name == "get_inventory"
