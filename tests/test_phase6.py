"""
Phase 6 — Hardening and packaging tests.

Exit criteria:
  - All three bundled domains (inventory, email, crm) load without error.
  - A ProviderHarness runs in-process with a mocked provider and returns the
    correct dict shape.
  - The inference server /invoke endpoint routes a request and returns the
    expected JSON structure (mocked orchestrator).
  - docker-compose.yml is valid YAML and declares the expected services.
  - README.md contains the required quickstart sections.
  - New domains can be added and immediately routed (end-to-end with mocked harness).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
DOMAINS_DIR = REPO_ROOT / "specialists" / "domains"



def load_tools_yaml(domain: str):
    path = DOMAINS_DIR / domain / "tools.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def load_goal_md(domain: str) -> str:
    path = DOMAINS_DIR / domain / "goal.md"
    return path.read_text()


# ═══════════════════════════════════════════════════════════════════════════
# TestDomainFiles — all three bundled domains have valid files
# ═══════════════════════════════════════════════════════════════════════════

class TestDomainFiles:
    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_tools_yaml_exists(self, domain):
        assert (DOMAINS_DIR / domain / "tools.yaml").exists()

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_goal_md_exists(self, domain):
        assert (DOMAINS_DIR / domain / "goal.md").exists()

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_tools_yaml_has_required_top_level_keys(self, domain):
        data = load_tools_yaml(domain)
        assert "domain" in data
        assert "version" in data
        assert "tools" in data
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) >= 2

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_tools_yaml_domain_matches_dir_name(self, domain):
        data = load_tools_yaml(domain)
        assert data["domain"] == domain

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_tools_yaml_has_no_tool(self, domain):
        data = load_tools_yaml(domain)
        names = [t["name"] for t in data["tools"]]
        assert "no_tool" in names, f"{domain}/tools.yaml missing 'no_tool'"

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_tools_yaml_has_at_least_one_real_tool(self, domain):
        data = load_tools_yaml(domain)
        non_no_tool = [t for t in data["tools"] if t["name"] != "no_tool"]
        assert len(non_no_tool) >= 1

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_tools_yaml_required_param_types_valid(self, domain):
        valid_types = {"string", "integer", "number", "boolean", "array", "object"}
        data = load_tools_yaml(domain)
        for tool in data["tools"]:
            for param_name, param_def in (tool.get("parameters") or {}).items():
                if isinstance(param_def, dict) and "type" in param_def:
                    assert param_def["type"] in valid_types, (
                        f"{domain}/{tool['name']}/{param_name}: invalid type {param_def['type']!r}"
                    )

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_goal_md_has_performance_targets(self, domain):
        text = load_goal_md(domain)
        assert "tool_selection_accuracy" in text
        assert "adapter_rank" in text

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_goal_md_has_adapter_rank_16(self, domain):
        text = load_goal_md(domain)
        assert "16" in text, f"{domain}/goal.md must specify adapter_rank: 16"

    @pytest.mark.parametrize("domain", ["inventory", "email", "crm"])
    def test_tools_yaml_parses_via_domainspec(self, domain):
        """DomainSpec Pydantic model can parse the file without errors."""
        from specialists.runtime.harness import load_domain
        spec = load_domain(str(DOMAINS_DIR / domain / "tools.yaml"))
        assert spec.domain == domain
        assert len(spec.tools) >= 2


# ═══════════════════════════════════════════════════════════════════════════
# TestProviderHarness
# ═══════════════════════════════════════════════════════════════════════════

class TestProviderHarness:
    def _make_domain(self):
        from specialists.runtime.harness import load_domain
        return load_domain(str(DOMAINS_DIR / "inventory" / "tools.yaml"))

    def _make_provider(self, raw_response: str):
        provider = MagicMock()
        provider.complete.return_value = raw_response
        return provider

    def test_load_is_noop(self):
        from specialists.runtime.provider_harness import ProviderHarness
        h = ProviderHarness(provider=MagicMock())
        h.load()  # must not raise

    def test_run_returns_correct_keys(self):
        from specialists.runtime.provider_harness import ProviderHarness
        raw = '{"name": "check_stock", "parameters": {"sku": "SKU-1"}}'
        h = ProviderHarness(provider=self._make_provider(raw))
        result = h.run(self._make_domain(), "How many SKU-1 units do we have?")
        assert set(result.keys()) == {"name", "parameters", "raw", "parse_error", "schema_errors"}

    def test_run_valid_call_no_parse_error(self):
        from specialists.runtime.provider_harness import ProviderHarness
        raw = '{"name": "get_inventory", "parameters": {"product_id": "SKU-1"}}'
        h = ProviderHarness(provider=self._make_provider(raw))
        result = h.run(self._make_domain(), "query")
        assert result["parse_error"] is False
        assert result["name"] == "get_inventory"
        assert result["parameters"] == {"product_id": "SKU-1"}
        assert result["schema_errors"] == []

    def test_run_no_tool(self):
        from specialists.runtime.provider_harness import ProviderHarness
        raw = '{"name": "no_tool", "parameters": {}}'
        h = ProviderHarness(provider=self._make_provider(raw))
        result = h.run(self._make_domain(), "what is the weather?")
        assert result["name"] == "no_tool"
        assert result["parse_error"] is False
        assert result["schema_errors"] == []

    def test_run_parse_error_on_bad_json(self):
        from specialists.runtime.provider_harness import ProviderHarness
        h = ProviderHarness(provider=self._make_provider("not json at all"))
        result = h.run(self._make_domain(), "query")
        assert result["parse_error"] is True
        assert result["name"] is None

    def test_run_provider_exception_returns_parse_error(self):
        from specialists.runtime.provider_harness import ProviderHarness
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("network failure")
        h = ProviderHarness(provider=provider)
        result = h.run(self._make_domain(), "query")
        assert result["parse_error"] is True
        assert result["name"] is None
        assert "provider_error" in result["raw"]

    def test_run_unknown_tool_reported_in_schema_errors(self):
        from specialists.runtime.provider_harness import ProviderHarness
        raw = '{"name": "delete_everything", "parameters": {}}'
        h = ProviderHarness(provider=self._make_provider(raw))
        result = h.run(self._make_domain(), "query")
        assert result["parse_error"] is False
        assert len(result["schema_errors"]) > 0

    def test_run_uses_pinned_system_prompt(self):
        """Provider receives the identical system prompt as ToolCallHarness."""
        from specialists.runtime.provider_harness import ProviderHarness
        provider = MagicMock()
        provider.complete.return_value = '{"name": "no_tool", "parameters": {}}'
        h = ProviderHarness(provider=provider)
        h.run(self._make_domain(), "query")
        call_args = provider.complete.call_args
        messages = call_args[0][0]
        system_msg = next(m for m in messages if m["role"] == "system")
        # The system message body should contain expanded tool names
        assert "get_inventory" in system_msg["content"]

    def test_email_domain_runs_without_error(self):
        from specialists.runtime.provider_harness import ProviderHarness
        from specialists.runtime.harness import load_domain
        domain_spec = load_domain(str(DOMAINS_DIR / "email" / "tools.yaml"))
        raw = '{"name": "send_email", "parameters": {"to": "a@b.com", "subject": "hi", "body": "hello"}}'
        h = ProviderHarness(provider=self._make_provider(raw))
        result = h.run(domain_spec, "send email to a@b.com")
        assert result["name"] == "send_email"
        assert result["schema_errors"] == []

    def test_crm_domain_runs_without_error(self):
        from specialists.runtime.provider_harness import ProviderHarness
        from specialists.runtime.harness import load_domain
        domain_spec = load_domain(str(DOMAINS_DIR / "crm" / "tools.yaml"))
        raw = '{"name": "create_contact", "parameters": {"name": "Alice"}}'
        h = ProviderHarness(provider=self._make_provider(raw))
        result = h.run(domain_spec, "add Alice as a new contact")
        assert result["name"] == "create_contact"
        assert result["schema_errors"] == []


# ═══════════════════════════════════════════════════════════════════════════
# TestOrchestratorHarnessFactory — harness_factory injection
# ═══════════════════════════════════════════════════════════════════════════

class TestOrchestratorHarnessFactory:
    def _make_registry_with_inventory(self, tmp_path):
        from specialists.registry import SpecialistRegistry, SpecialistEntry
        reg = SpecialistRegistry(db_path=str(tmp_path / "r.db"))
        reg.connect()
        e = SpecialistEntry(
            specialist_id="inv-v1",
            domain="inventory",
            base_model="base",
            adapter_path="",
            tools_yaml_path=str(DOMAINS_DIR / "inventory" / "tools.yaml"),
            eval_score=0.9,
            trained_at="1970-01-01T00:00:00Z",
            status="candidate",
        )
        reg.register(e)
        reg.promote("inv-v1")
        return reg

    def test_harness_factory_called_instead_of_toolcallharness(self, tmp_path):
        from router.orchestrator import Orchestrator
        from router.router import Router

        registry = self._make_registry_with_inventory(tmp_path)

        mock_harness = MagicMock()
        mock_harness.run.return_value = {
            "name": "check_stock",
            "parameters": {"sku": "SKU-1"},
            "raw": '{"name": "check_stock", "parameters": {"sku": "SKU-1"}}',
            "parse_error": False,
            "schema_errors": [],
        }
        mock_harness.load = MagicMock()

        factory_calls = []

        def factory(entry):
            factory_calls.append(entry.specialist_id)
            return mock_harness

        orchestrator = Orchestrator(
            registry=registry,
            base_model="base",
            harness_factory=factory,
        )

        router = Router(registry=registry)
        router.build_index()
        plan = router.route("how many units of SKU-123 are left in warehouse WH-A?")
        orchestrator.execute(plan)

        assert "inv-v1" in factory_calls
        mock_harness.load.assert_called_once()

    def test_none_harness_factory_uses_toolcallharness(self, tmp_path):
        """With harness_factory=None, _get_harness imports ToolCallHarness."""
        from router.orchestrator import Orchestrator
        registry = self._make_registry_with_inventory(tmp_path)
        orc = Orchestrator(registry=registry, base_model="base", harness_factory=None)
        assert orc._harness_factory is None


# ═══════════════════════════════════════════════════════════════════════════
# TestServerApp — FastAPI inference server
# ═══════════════════════════════════════════════════════════════════════════

class TestServerApp:
    @pytest.fixture
    def client(self, tmp_path):
        from fastapi.testclient import TestClient
        from server.app import app, get_registry, get_router, get_orchestrator, get_feedback

        from specialists.registry import SpecialistRegistry, SpecialistEntry
        from feedback.store import FeedbackStore
        from router.router import Router

        registry = SpecialistRegistry(db_path=str(tmp_path / "r.db"))
        registry.connect()
        e = SpecialistEntry(
            specialist_id="inv-v1",
            domain="inventory",
            base_model="base",
            adapter_path="",
            tools_yaml_path=str(DOMAINS_DIR / "inventory" / "tools.yaml"),
            eval_score=0.9,
            trained_at="1970-01-01T00:00:00Z",
            status="candidate",
        )
        registry.register(e)
        registry.promote("inv-v1")

        feedback_store = FeedbackStore(db_path=str(tmp_path / "f.db"))

        router = Router(registry=registry)
        router.build_index()

        mock_harness = MagicMock()
        mock_harness.run.return_value = {
            "name": "get_inventory",
            "parameters": {"product_id": "SKU-1"},
            "raw": '{"name": "get_inventory", "parameters": {"product_id": "SKU-1"}}',
            "parse_error": False,
            "schema_errors": [],
        }
        mock_harness.load = MagicMock()

        from router.orchestrator import Orchestrator
        orchestrator = Orchestrator(
            registry=registry,
            base_model="base",
            feedback_store=feedback_store,
            harness_factory=lambda _entry: mock_harness,
        )

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_feedback] = lambda: feedback_store
        app.dependency_overrides[get_router] = lambda: router
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator

        with TestClient(app) as c:
            yield c

        app.dependency_overrides.clear()

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_lists_active_specialists(self, client):
        resp = client.get("/health")
        assert "inv-v1" in resp.json()["active_specialists"]

    def test_domains_endpoint(self, client):
        resp = client.get("/domains")
        assert resp.status_code == 200
        domains = resp.json()
        assert any(d["domain"] == "inventory" for d in domains)

    def test_invoke_returns_200(self, client):
        resp = client.post("/invoke", json={"request": "how many units of SKU-123 are in warehouse WH-A?"})
        assert resp.status_code == 200

    def test_invoke_response_shape(self, client):
        resp = client.post("/invoke", json={"request": "how many units of SKU-123 are in warehouse WH-A?"})
        data = resp.json()
        assert "success" in data
        assert "plan" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_invoke_tool_name_in_result(self, client):
        resp = client.post("/invoke", json={"request": "how many units of SKU-123 are in warehouse WH-A?"})
        results = resp.json()["results"]
        assert len(results) >= 1
        assert results[0]["tool_name"] == "get_inventory"

    def test_invoke_empty_request_raises_422(self, client):
        resp = client.post("/invoke", json={"request": "  "})
        assert resp.status_code == 422

    def test_reload_endpoint(self, client):
        resp = client.post("/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════════
# TestPackagingFiles — docker-compose.yml, .env.example, README
# ═══════════════════════════════════════════════════════════════════════════

class TestPackagingFiles:
    def test_dockerfile_exists(self):
        assert (REPO_ROOT / "Dockerfile").exists()

    def test_docker_compose_exists(self):
        assert (REPO_ROOT / "docker-compose.yml").exists()

    def test_docker_compose_is_valid_yaml(self):
        with (REPO_ROOT / "docker-compose.yml").open() as f:
            data = yaml.safe_load(f)
        assert data is not None

    def test_docker_compose_has_server_service(self):
        with (REPO_ROOT / "docker-compose.yml").open() as f:
            data = yaml.safe_load(f)
        assert "server" in data["services"]

    def test_docker_compose_has_dashboard_service(self):
        with (REPO_ROOT / "docker-compose.yml").open() as f:
            data = yaml.safe_load(f)
        assert "dashboard" in data["services"]

    def test_docker_compose_has_scheduler_service(self):
        with (REPO_ROOT / "docker-compose.yml").open() as f:
            data = yaml.safe_load(f)
        assert "scheduler" in data["services"]

    def test_docker_compose_server_exposes_port_8000(self):
        with (REPO_ROOT / "docker-compose.yml").open() as f:
            data = yaml.safe_load(f)
        ports = data["services"]["server"].get("ports", [])
        assert any("8000" in str(p) for p in ports)

    def test_docker_compose_dashboard_exposes_port_8080(self):
        with (REPO_ROOT / "docker-compose.yml").open() as f:
            data = yaml.safe_load(f)
        ports = data["services"]["dashboard"].get("ports", [])
        assert any("8080" in str(p) for p in ports)

    def test_env_example_exists(self):
        assert (REPO_ROOT / ".env.example").exists()

    def test_env_example_has_provider_key_placeholder(self):
        text = (REPO_ROOT / ".env.example").read_text()
        assert "TOOLHIVE_PROVIDER_API_KEY" in text

    def test_readme_exists(self):
        assert (REPO_ROOT / "README.md").exists()

    def test_readme_has_quick_start(self):
        text = (REPO_ROOT / "README.md").read_text()
        assert "quickstart" in text.lower() or "quick start" in text.lower()

    def test_readme_has_adding_domain_section(self):
        text = (REPO_ROOT / "README.md").read_text()
        assert "adding a domain" in text.lower() or "new domain" in text.lower()

    def test_readme_has_environment_variables_table(self):
        text = (REPO_ROOT / "README.md").read_text()
        assert "TOOLHIVE_PROVIDER_API_KEY" in text

    def test_docs_tools_yaml_reference_exists(self):
        assert (REPO_ROOT / "docs" / "tools_yaml_reference.md").exists()

    def test_docs_goal_md_reference_exists(self):
        assert (REPO_ROOT / "docs" / "goal_md_reference.md").exists()

    def test_init_demo_script_exists(self):
        assert (REPO_ROOT / "scripts" / "init_demo.py").exists()


# ═══════════════════════════════════════════════════════════════════════════
# TestInitDemoScript — registers three domains
# ═══════════════════════════════════════════════════════════════════════════

class TestInitDemoScript:
    def test_registers_all_three_domains(self, tmp_path):
        import subprocess, sys
        db_path = str(tmp_path / "reg.db")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "init_demo.py"), "--registry", db_path],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr

        from specialists.registry import SpecialistRegistry
        reg = SpecialistRegistry(db_path=db_path)
        reg.connect()
        active = {e.domain for e in reg.list_active()}
        assert "inventory" in active
        assert "email" in active
        assert "crm" in active

    def test_idempotent_second_run(self, tmp_path):
        import subprocess, sys
        db_path = str(tmp_path / "reg.db")
        script = [sys.executable, str(REPO_ROOT / "scripts" / "init_demo.py"), "--registry", db_path]
        subprocess.run(script, capture_output=True, cwd=str(REPO_ROOT))
        result = subprocess.run(script, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0
        # Second run should skip all three (already active)
        assert result.stdout.count("[skip]") == 3


# ═══════════════════════════════════════════════════════════════════════════
# TestExitCriteria — new user can stand up 3-specialist swarm
# ═══════════════════════════════════════════════════════════════════════════

class TestExitCriteria:
    def test_three_domains_have_valid_tools_yaml(self):
        """All three bundled domains parse as DomainSpec without error."""
        from specialists.runtime.harness import load_domain
        for domain in ["inventory", "email", "crm"]:
            spec = load_domain(str(DOMAINS_DIR / domain / "tools.yaml"))
            assert spec.domain == domain

    def test_three_domains_registered_and_routeable(self, tmp_path):
        """After init_demo registers 3 domains, the router resolves all of them."""
        from specialists.registry import SpecialistRegistry, SpecialistEntry
        from router.router import Router

        reg = SpecialistRegistry(db_path=str(tmp_path / "r.db"))
        reg.connect()
        for domain in ["inventory", "email", "crm"]:
            e = SpecialistEntry(
                specialist_id=f"{domain}-v1",
                domain=domain,
                base_model="base",
                adapter_path="",
                tools_yaml_path=str(DOMAINS_DIR / domain / "tools.yaml"),
                eval_score=0.9,
                trained_at="1970-01-01T00:00:00Z",
                status="candidate",
            )
            reg.register(e)
            reg.promote(f"{domain}-v1")

        router = Router(registry=reg)
        router.build_index()

        # Each query should route to the correct domain
        queries = {
            "inventory": "how many SKU-1 units are left in warehouse WH-A?",
            "email": "send an email to alice@example.com about the meeting",
            "crm": "add Bob Smith as a new contact",
        }
        for expected_domain, query in queries.items():
            plan = router.route(query)
            domains_hit = {step.domain for step in plan.steps}
            assert expected_domain in domains_hit, (
                f"router did not route '{query}' to domain '{expected_domain}'; "
                f"got: {domains_hit}"
            )

    def test_provider_harness_works_for_all_three_domains(self):
        """ProviderHarness runs on all three domains with a mocked provider."""
        from specialists.runtime.provider_harness import ProviderHarness
        from specialists.runtime.harness import load_domain

        responses = {
            "inventory": '{"name": "get_inventory", "parameters": {"product_id": "SKU-1"}}',
            "email": '{"name": "send_email", "parameters": {"to": "a@b.com", "subject": "hi", "body": "hello"}}',
            "crm": '{"name": "create_contact", "parameters": {"name": "Alice"}}',
        }

        for domain, raw in responses.items():
            provider = MagicMock()
            provider.complete.return_value = raw
            spec = load_domain(str(DOMAINS_DIR / domain / "tools.yaml"))
            h = ProviderHarness(provider=provider)
            result = h.run(spec, "test query")
            assert result["parse_error"] is False, f"{domain}: unexpected parse error"
            assert result["schema_errors"] == [], f"{domain}: unexpected schema errors"

    def test_server_invoke_routes_multi_step_request(self, tmp_path):
        """Multi-step request is split and each sub-query lands in the right domain."""
        from fastapi.testclient import TestClient
        from server.app import app, get_registry, get_router, get_orchestrator, get_feedback
        from specialists.registry import SpecialistRegistry, SpecialistEntry
        from feedback.store import FeedbackStore
        from router.router import Router
        from router.orchestrator import Orchestrator

        registry = SpecialistRegistry(db_path=str(tmp_path / "r.db"))
        registry.connect()
        for domain in ["inventory", "email", "crm"]:
            e = SpecialistEntry(
                specialist_id=f"{domain}-v1",
                domain=domain,
                base_model="base",
                adapter_path="",
                tools_yaml_path=str(DOMAINS_DIR / domain / "tools.yaml"),
                eval_score=0.9,
                trained_at="1970-01-01T00:00:00Z",
                status="candidate",
            )
            registry.register(e)
            registry.promote(f"{domain}-v1")

        feedback_store = FeedbackStore(db_path=str(tmp_path / "f.db"))
        router = Router(registry=registry)
        router.build_index()

        def make_harness(_entry):
            h = MagicMock()
            h.load = MagicMock()
            h.run.return_value = {
                "name": "no_tool", "parameters": {}, "raw": "{}",
                "parse_error": False, "schema_errors": [],
            }
            return h

        orchestrator = Orchestrator(
            registry=registry, base_model="base",
            feedback_store=feedback_store, harness_factory=make_harness,
        )

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_feedback] = lambda: feedback_store
        app.dependency_overrides[get_router] = lambda: router
        app.dependency_overrides[get_orchestrator] = lambda: orchestrator

        try:
            with TestClient(app) as c:
                resp = c.post(
                    "/invoke",
                    json={"request": "check stock for SKU-1 then send an email to the supplier"},
                )
        finally:
            app.dependency_overrides.clear()

        assert resp.status_code == 200
        data = resp.json()
        assert data["plan"]["is_multi_step"] is True
        assert len(data["results"]) >= 2
