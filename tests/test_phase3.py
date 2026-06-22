"""
Phase 3 unit tests — all run without GPU, real model, or LLM provider.

Covers:
  - Schema validation (Layer 1, pure Python)
  - Semantic critic (Layer 2, mocked LLM provider)
  - Escalation (Layer 3, mocked second provider)
  - PII scrubbing
  - Calibration precision/recall
  - build_injection_gold_set exit-criteria (≥90% recall on schema layer alone)
  - Orchestrator critic integration
  - Feedback store logging via orchestrator
"""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from feedback.store import CriticVerdict, FeedbackStore, SignalType
from critic.verifier import (
    CriticResult,
    CriticVerifier,
    _check_schema,
    _scrub_pii,
)
from critic.calibration import (
    GoldExample,
    CalibrationReport,
    calibrate,
    build_injection_gold_set,
    KNOWN_BIAS_CATEGORIES,
)
from specialists.runtime.schema import DomainSpec, ParameterSpec, ToolSpec



def _make_domain() -> DomainSpec:
    return DomainSpec(
        domain="inventory",
        tools=[
            ToolSpec(
                name="get_inventory",
                description="Check current stock levels",
                parameters={
                    "product_id": ParameterSpec(type="string", required=True,
                                                description="Product SKU"),
                    "warehouse": ParameterSpec(type="string", required=False,
                                               description="Warehouse code",
                                               enum=["WH-A", "WH-B"]),
                },
            ),
            ToolSpec(
                name="reserve_stock",
                description="Reserve a quantity of stock",
                parameters={
                    "product_id": ParameterSpec(type="string", required=True),
                    "quantity": ParameterSpec(type="integer", required=True),
                },
            ),
            ToolSpec(name="no_tool", description="No tool needed", parameters={}),
        ],
    )


def _make_provider_returning(response_dict: dict) -> MagicMock:
    provider = MagicMock()
    provider.complete.return_value = json.dumps(response_dict)
    return provider



class TestSchemaValidation:
    def test_valid_call_passes(self):
        domain = _make_domain()
        result = _check_schema(
            {"name": "get_inventory", "parameters": {"product_id": "SKU-1"}},
            domain, parse_error=False,
        )
        assert result.verdict == CriticVerdict.PASS
        assert result.layer == "schema"

    def test_parse_error_blocks(self):
        result = _check_schema({}, _make_domain(), parse_error=True)
        assert result.verdict == CriticVerdict.BLOCK
        assert "parsed" in result.reason

    def test_missing_name_blocks(self):
        result = _check_schema({"parameters": {}}, _make_domain(), parse_error=False)
        assert result.verdict == CriticVerdict.BLOCK

    def test_unknown_tool_blocks(self):
        result = _check_schema(
            {"name": "nonexistent_tool", "parameters": {}}, _make_domain(), parse_error=False
        )
        assert result.verdict == CriticVerdict.BLOCK
        assert "unknown tool" in result.reason

    def test_missing_required_param_blocks(self):
        result = _check_schema(
            {"name": "get_inventory", "parameters": {}}, _make_domain(), parse_error=False
        )
        assert result.verdict == CriticVerdict.BLOCK
        assert "product_id" in result.reason

    def test_type_mismatch_flags(self):
        # quantity must be integer
        result = _check_schema(
            {"name": "reserve_stock", "parameters": {"product_id": "SKU-1", "quantity": "five"}},
            _make_domain(), parse_error=False,
        )
        assert result.verdict == CriticVerdict.FLAG
        assert "quantity" in result.reason

    def test_enum_violation_flags(self):
        result = _check_schema(
            {"name": "get_inventory", "parameters": {"product_id": "X", "warehouse": "WH-Z"}},
            _make_domain(), parse_error=False,
        )
        assert result.verdict == CriticVerdict.FLAG
        assert "warehouse" in result.reason

    def test_unknown_param_flags_hallucination(self):
        result = _check_schema(
            {"name": "get_inventory", "parameters": {"product_id": "X", "ghost_param": "y"}},
            _make_domain(), parse_error=False,
        )
        assert result.verdict == CriticVerdict.FLAG
        assert "ghost_param" in result.reason

    def test_no_tool_always_passes(self):
        result = _check_schema(
            {"name": "no_tool", "parameters": {}}, _make_domain(), parse_error=False
        )
        assert result.verdict == CriticVerdict.PASS

    def test_boolean_param_type_checked(self):
        domain = DomainSpec(domain="test", tools=[
            ToolSpec(name="act", description="", parameters={
                "flag": ParameterSpec(type="boolean", required=True),
            })
        ])
        result = _check_schema(
            {"name": "act", "parameters": {"flag": "true"}},
            domain, parse_error=False,
        )
        assert result.verdict == CriticVerdict.FLAG

    def test_valid_optional_enum_passes(self):
        result = _check_schema(
            {"name": "get_inventory", "parameters": {"product_id": "X", "warehouse": "WH-A"}},
            _make_domain(), parse_error=False,
        )
        assert result.verdict == CriticVerdict.PASS



class TestCriticVerifierSchemaLayer:
    """Verifier without provider — only schema layer fires."""

    def test_block_propagates_without_provider(self):
        critic = CriticVerifier()
        result = critic.verify("q", {}, _make_domain(), parse_error=True)
        assert result.verdict == CriticVerdict.BLOCK

    def test_pass_without_provider(self):
        critic = CriticVerifier()
        result = critic.verify(
            "check inventory",
            {"name": "get_inventory", "parameters": {"product_id": "X"}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.PASS

    def test_no_provider_skips_semantic(self):
        critic = CriticVerifier(provider=None)
        result = critic.verify(
            "reserve units",
            {"name": "reserve_stock", "parameters": {"product_id": "X", "quantity": 3}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.PASS
        assert result.layer == "schema"


class TestCriticVerifierSemanticLayer:
    """Verifier with mocked provider."""

    def test_semantic_pass(self):
        provider = _make_provider_returning({"verdict": "pass", "reason": "all good"})
        critic = CriticVerifier(provider=provider)
        result = critic.verify(
            "check inventory",
            {"name": "get_inventory", "parameters": {"product_id": "SKU-1"}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.PASS
        assert result.layer == "semantic"

    def test_semantic_flag(self):
        provider = _make_provider_returning({"verdict": "flag", "reason": "wrong param value"})
        critic = CriticVerifier(provider=provider)
        result = critic.verify(
            "check inventory",
            {"name": "get_inventory", "parameters": {"product_id": "SKU-1"}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.FLAG

    def test_semantic_block(self):
        provider = _make_provider_returning({"verdict": "block", "reason": "wrong tool"})
        critic = CriticVerifier(provider=provider)
        result = critic.verify(
            "reserve units",
            {"name": "get_inventory", "parameters": {"product_id": "SKU-1"}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.BLOCK

    def test_semantic_corrected_output(self):
        provider = _make_provider_returning({
            "verdict": "flag",
            "reason": "wrong value",
            "corrected_parameters": {"product_id": "CORRECT-SKU"},
        })
        critic = CriticVerifier(provider=provider)
        result = critic.verify(
            "check inventory",
            {"name": "get_inventory", "parameters": {"product_id": "bad"}},
            _make_domain(),
        )
        assert result.corrected_output is not None
        assert result.corrected_output["parameters"]["product_id"] == "CORRECT-SKU"

    def test_schema_block_does_not_call_provider(self):
        provider = MagicMock()
        critic = CriticVerifier(provider=provider)
        critic.verify("q", {}, _make_domain(), parse_error=True)
        provider.complete.assert_not_called()

    def test_provider_exception_is_flag(self):
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("timeout")
        critic = CriticVerifier(provider=provider)
        result = critic.verify(
            "check inventory",
            {"name": "get_inventory", "parameters": {"product_id": "X"}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.FLAG
        assert result.confidence == 0.0

    def test_malformed_json_from_provider_is_flag(self):
        provider = MagicMock()
        provider.complete.return_value = "not json"
        critic = CriticVerifier(provider=provider)
        result = critic.verify(
            "check inventory",
            {"name": "get_inventory", "parameters": {"product_id": "X"}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.FLAG


class TestCriticVerifierEscalation:
    """Escalation fires only on FLAG from semantic layer."""

    def test_escalation_resolves_flag_to_block(self):
        provider = _make_provider_returning({"verdict": "flag", "reason": "unsure"})
        escalation = _make_provider_returning({"verdict": "block", "reason": "confirmed bad"})
        critic = CriticVerifier(provider=provider, escalation_provider=escalation)
        result = critic.verify(
            "reserve units",
            {"name": "reserve_stock", "parameters": {"product_id": "X", "quantity": 5}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.BLOCK
        assert result.layer == "escalation"

    def test_escalation_resolves_flag_to_pass(self):
        provider = _make_provider_returning({"verdict": "flag", "reason": "unsure"})
        escalation = _make_provider_returning({"verdict": "pass", "reason": "actually fine"})
        critic = CriticVerifier(provider=provider, escalation_provider=escalation)
        result = critic.verify(
            "reserve units",
            {"name": "reserve_stock", "parameters": {"product_id": "X", "quantity": 5}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.PASS
        assert result.layer == "escalation"

    def test_escalation_not_called_on_pass(self):
        provider = _make_provider_returning({"verdict": "pass", "reason": "fine"})
        escalation = MagicMock()
        critic = CriticVerifier(provider=provider, escalation_provider=escalation)
        critic.verify(
            "check inventory",
            {"name": "get_inventory", "parameters": {"product_id": "X"}},
            _make_domain(),
        )
        escalation.complete.assert_not_called()

    def test_escalation_error_keeps_flag(self):
        provider = _make_provider_returning({"verdict": "flag", "reason": "unsure"})
        escalation = MagicMock()
        escalation.complete.side_effect = RuntimeError("escalation timeout")
        critic = CriticVerifier(provider=provider, escalation_provider=escalation)
        result = critic.verify(
            "reserve units",
            {"name": "reserve_stock", "parameters": {"product_id": "X", "quantity": 5}},
            _make_domain(),
        )
        assert result.verdict == CriticVerdict.FLAG
        assert result.layer == "escalation"

    def test_critic_result_to_dict(self):
        r = CriticResult(verdict=CriticVerdict.PASS, reason="ok", layer="schema")
        d = r.to_dict()
        assert d["verdict"] == "pass"
        assert d["layer"] == "schema"
        assert "confidence" in d



class TestPiiScrubbing:
    def test_email_scrubbed(self):
        assert "[EMAIL]" in _scrub_pii("send to user@example.com please")

    def test_phone_scrubbed(self):
        assert "[PHONE]" in _scrub_pii("call 555-123-4567")

    def test_ssn_scrubbed(self):
        assert "[SSN]" in _scrub_pii("SSN: 123-45-6789")

    def test_ip_scrubbed(self):
        assert "[IP]" in _scrub_pii("server at 192.168.1.100")

    def test_no_pii_unchanged(self):
        text = "check inventory for SKU-123"
        assert _scrub_pii(text) == text

    def test_multiple_pii_types(self):
        text = "user user@test.com at 10.0.0.1"
        scrubbed = _scrub_pii(text)
        assert "[EMAIL]" in scrubbed
        assert "[IP]" in scrubbed



class TestCalibration:
    def _make_gold(self, verdicts: list[CriticVerdict]) -> list[GoldExample]:
        domain = _make_domain()
        gold = []
        for i, v in enumerate(verdicts):
            if v == CriticVerdict.PASS:
                gold.append(GoldExample(
                    sub_query=f"check {i}",
                    tool_call={"name": "get_inventory", "parameters": {"product_id": f"S{i}"}},
                    expected_verdict=CriticVerdict.PASS,
                ))
            else:
                gold.append(GoldExample(
                    sub_query=f"bad {i}",
                    tool_call={"name": "get_inventory", "parameters": {}},
                    expected_verdict=CriticVerdict.BLOCK,
                ))
        return gold

    def test_perfect_precision_recall(self):
        domain = _make_domain()
        critic = CriticVerifier()  # schema-only: deterministic
        gold = self._make_gold(
            [CriticVerdict.PASS] * 5 + [CriticVerdict.BLOCK] * 5
        )
        report = calibrate(critic, domain, gold)
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(1.0)
        assert report.passes_threshold is True
        assert report.n_examples == 10

    def test_all_false_negatives_zero_recall(self):
        domain = DomainSpec(domain="x", tools=[
            ToolSpec(name="act", description="", parameters={
                "p": ParameterSpec(type="string", required=False)
            })
        ])
        critic = CriticVerifier()
        # All examples are BLOCK but critic will PASS them (optional param, no parse error)
        gold = [
            GoldExample(
                sub_query="q",
                tool_call={"name": "act", "parameters": {}},
                expected_verdict=CriticVerdict.BLOCK,
                description="critic won't catch this (no required params)",
            )
            for _ in range(5)
        ]
        report = calibrate(critic, domain, gold)
        assert report.recall == pytest.approx(0.0)
        assert report.passes_threshold is False

    def test_passes_threshold_flag(self):
        domain = _make_domain()
        critic = CriticVerifier()
        gold = self._make_gold([CriticVerdict.PASS] * 10 + [CriticVerdict.BLOCK] * 10)
        report = calibrate(critic, domain, gold)
        assert report.passes_threshold == (report.precision >= 0.85)

    def test_bias_breakdown_populated(self):
        domain = _make_domain()
        critic = CriticVerifier()
        gold = [
            GoldExample(
                sub_query="q",
                tool_call={"name": "get_inventory", "parameters": {}},
                expected_verdict=CriticVerdict.BLOCK,
                bias_categories=["required_param_miss"],
            )
        ]
        with pytest.warns(UserWarning, match="gold set has only"):
            report = calibrate(critic, domain, gold)
        assert "required_param_miss" in report.bias_breakdown

    def test_small_gold_set_warns(self):
        domain = _make_domain()
        critic = CriticVerifier()
        gold = [
            GoldExample(
                sub_query="q",
                tool_call={"name": "get_inventory", "parameters": {"product_id": "X"}},
                expected_verdict=CriticVerdict.PASS,
            )
        ]
        with pytest.warns(UserWarning, match="gold set has only"):
            calibrate(critic, domain, gold)

    def test_report_summary(self):
        domain = _make_domain()
        critic = CriticVerifier()
        gold = self._make_gold([CriticVerdict.PASS] * 5 + [CriticVerdict.BLOCK] * 5)
        report = calibrate(critic, domain, gold)
        summary = report.summary()
        assert "Precision" in summary
        assert "Recall" in summary

    def test_known_bias_categories_count(self):
        assert len(KNOWN_BIAS_CATEGORIES) >= 14



class TestInjectionGoldSet:
    def test_generates_examples(self):
        domain = _make_domain()
        gold = build_injection_gold_set(domain)
        assert len(gold) > 0

    def test_contains_good_and_bad(self):
        domain = _make_domain()
        gold = build_injection_gold_set(domain)
        verdicts = {ex.expected_verdict for ex in gold}
        assert CriticVerdict.PASS in verdicts
        assert CriticVerdict.BLOCK in verdicts

    def test_schema_critic_recall_meets_exit_criterion(self):
        """
        EXIT CRITERIA: critic (schema layer alone, no LLM) must achieve
        ≥90% recall on the injected gold set for the inventory domain.
        """
        domain = _make_domain()
        critic = CriticVerifier()  # schema layer only
        gold = build_injection_gold_set(domain)

        with pytest.warns(UserWarning) if len(gold) < 10 else contextlib_nullcontext():
            report = calibrate(critic, domain, gold)

        assert report.recall >= 0.90, (
            f"Schema-layer critic recall {report.recall:.1%} is below the 90% exit criterion.\n"
            f"{report.summary()}"
        )



class TestOrchestratorCriticIntegration:
    def _make_registry_and_entry(self, tmp_path):
        from specialists.registry import SpecialistRegistry, SpecialistEntry
        reg = SpecialistRegistry(tmp_path / "reg.db")
        reg.connect()
        entry = SpecialistEntry(
            specialist_id="inventory-v1",
            domain="inventory",
            base_model="Qwen/Qwen2.5-3B-Instruct",
            adapter_path="adapters/inventory/v1",
            tools_yaml_path="specialists/domains/inventory/tools.yaml",
            eval_score=0.96,
            trained_at="2026-06-21T00:00:00Z",
            status="active",
        )
        reg.register(entry)
        return reg, entry

    def _make_plan(self):
        from router.router import DispatchPlan, DispatchStep
        return DispatchPlan(
            request="check inventory",
            steps=[DispatchStep(
                specialist_id="inventory-v1",
                domain="inventory",
                sub_query="check inventory for SKU-1",
                similarity_score=0.9,
            )],
        )

    def _mock_harness(self, tool_name="get_inventory", parse_error=False):
        mock = MagicMock()
        mock.run.return_value = {
            "name": tool_name,
            "parameters": {"product_id": "SKU-1"},
            "raw": "",
            "parse_error": parse_error,
            "schema_errors": [],
        }
        return mock

    def test_critic_verdict_in_step_result(self, tmp_path):
        reg, _ = self._make_registry_and_entry(tmp_path)
        domain = _make_domain()
        critic = CriticVerifier()
        from router.orchestrator import Orchestrator
        orch = Orchestrator(reg, "model", critic=critic)
        orch._harness_cache["inventory-v1"] = self._mock_harness()
        orch._domain_cache["inventory"] = domain

        result = orch.execute(self._make_plan())
        assert result.results[0].critic_verdict == "pass"
        assert result.results[0].critic_layer == "schema"

    def test_critic_block_makes_success_false(self, tmp_path):
        reg, _ = self._make_registry_and_entry(tmp_path)
        domain = _make_domain()
        critic = CriticVerifier()
        from router.orchestrator import Orchestrator
        orch = Orchestrator(reg, "model", critic=critic)
        orch._harness_cache["inventory-v1"] = self._mock_harness(parse_error=True)
        orch._domain_cache["inventory"] = domain

        result = orch.execute(self._make_plan())
        assert result.results[0].critic_verdict == "block"
        assert result.success is False

    def test_feedback_logged_on_critic_pass(self, tmp_path):
        reg, _ = self._make_registry_and_entry(tmp_path)
        domain = _make_domain()
        critic = CriticVerifier()
        store = FeedbackStore(tmp_path / "fb.db")
        store.connect()
        from router.orchestrator import Orchestrator
        orch = Orchestrator(reg, "model", critic=critic, feedback_store=store)
        orch._harness_cache["inventory-v1"] = self._mock_harness()
        orch._domain_cache["inventory"] = domain

        orch.execute(self._make_plan())

        since = "2000-01-01T00:00:00"
        # PASS verdict → adoption_decision signal, not a failure
        failures = store.failures_since("inventory-v1", since)
        # A passing step is not a failure, so failures list should be empty
        assert len(failures) == 0
        store.close()

    def test_feedback_logged_on_critic_block(self, tmp_path):
        reg, _ = self._make_registry_and_entry(tmp_path)
        domain = _make_domain()
        critic = CriticVerifier()
        store = FeedbackStore(tmp_path / "fb.db")
        store.connect()
        from router.orchestrator import Orchestrator
        orch = Orchestrator(reg, "model", critic=critic, feedback_store=store)
        orch._harness_cache["inventory-v1"] = self._mock_harness(parse_error=True)
        orch._domain_cache["inventory"] = domain

        orch.execute(self._make_plan())

        since = "2000-01-01T00:00:00"
        failures = store.failures_since("inventory-v1", since)
        assert len(failures) == 1
        assert failures[0].critic_verdict == CriticVerdict.BLOCK
        store.close()

    def test_no_critic_means_no_verdict(self, tmp_path):
        reg, _ = self._make_registry_and_entry(tmp_path)
        domain = _make_domain()
        from router.orchestrator import Orchestrator
        orch = Orchestrator(reg, "model")  # no critic
        orch._harness_cache["inventory-v1"] = self._mock_harness()
        orch._domain_cache["inventory"] = domain

        result = orch.execute(self._make_plan())
        assert result.results[0].critic_verdict is None

    def test_orchestrator_to_dict_includes_critic(self, tmp_path):
        reg, _ = self._make_registry_and_entry(tmp_path)
        domain = _make_domain()
        critic = CriticVerifier()
        from router.orchestrator import Orchestrator
        orch = Orchestrator(reg, "model", critic=critic)
        orch._harness_cache["inventory-v1"] = self._mock_harness()
        orch._domain_cache["inventory"] = domain

        result = orch.execute(self._make_plan())
        d = result.to_dict()
        assert "critic_verdict" in d["results"][0]
        assert "critic_reason" in d["results"][0]



from contextlib import contextmanager

@contextmanager
def contextlib_nullcontext():
    yield
