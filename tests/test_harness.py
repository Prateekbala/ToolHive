"""
Unit tests for Phase 0 components.

Tests here must run without a GPU or downloaded model (CI-safe).
Model-dependent integration tests are marked with @pytest.mark.requires_model
and skipped in CI.
"""

import pytest

from specialists.runtime.parser import parse_tool_call, validate_against_spec
from specialists.runtime.schema import DomainSpec, ParameterSpec, ToolSpec



def _make_domain() -> DomainSpec:
    return DomainSpec(
        domain="inventory",
        tools=[
            ToolSpec(
                name="get_inventory",
                description="Check stock levels",
                parameters={
                    "product_id": ParameterSpec(type="string", required=True),
                    "warehouse": ParameterSpec(type="string", required=False),
                },
            ),
            ToolSpec(
                name="reserve_stock",
                description="Reserve units",
                parameters={
                    "product_id": ParameterSpec(type="string", required=True),
                    "quantity": ParameterSpec(type="integer", required=True),
                    "order_id": ParameterSpec(type="string", required=True),
                },
            ),
            ToolSpec(name="no_tool", description="No match", parameters={}),
        ],
    )



class TestParseToolCall:
    def test_raw_json(self):
        text = '{"name": "get_inventory", "parameters": {"product_id": "SKU-123"}}'
        result = parse_tool_call(text)
        assert result is not None
        assert result["name"] == "get_inventory"
        assert result["parameters"]["product_id"] == "SKU-123"

    def test_fenced_code_block_with_lang(self):
        text = '```json\n{"name": "reserve_stock", "parameters": {"product_id": "X", "quantity": 5, "order_id": "ORD-1"}}\n```'
        result = parse_tool_call(text)
        assert result is not None
        assert result["name"] == "reserve_stock"

    def test_fenced_code_block_no_lang(self):
        text = '```\n{"name": "no_tool", "parameters": {}}\n```'
        result = parse_tool_call(text)
        assert result is not None
        assert result["name"] == "no_tool"

    def test_json_embedded_in_text(self):
        text = 'Sure! {"name": "get_inventory", "parameters": {"product_id": "SKU-99"}}'
        result = parse_tool_call(text)
        assert result is not None
        assert result["name"] == "get_inventory"

    def test_no_tool_empty_params(self):
        text = '{"name": "no_tool", "parameters": {}}'
        result = parse_tool_call(text)
        assert result is not None
        assert result["name"] == "no_tool"
        assert result["parameters"] == {}

    def test_parameters_defaults_to_empty_dict(self):
        text = '{"name": "get_inventory"}'
        result = parse_tool_call(text)
        assert result is not None
        assert result["parameters"] == {}

    def test_malformed_json_returns_none(self):
        assert parse_tool_call("I don't know how to answer that.") is None

    def test_partial_json_returns_none(self):
        assert parse_tool_call('{"name": "get_inventory"') is None

    def test_empty_string_returns_none(self):
        assert parse_tool_call("") is None

    def test_whitespace_only_returns_none(self):
        assert parse_tool_call("   \n  ") is None

    def test_object_inside_array_is_extracted(self):
        # Parser is lenient: extracts the inner object even if the model
        # wrapped the call in an array — better than silently failing.
        result = parse_tool_call('[{"name": "get_inventory"}]')
        assert result is not None
        assert result["name"] == "get_inventory"

    def test_nested_json_in_parameters(self):
        text = '{"name": "get_inventory", "parameters": {"product_id": "SKU-1", "warehouse": "WH-A"}}'
        result = parse_tool_call(text)
        assert result is not None
        assert result["parameters"]["warehouse"] == "WH-A"



class TestDomainSpec:
    def test_valid_domain(self):
        domain = _make_domain()
        assert domain.domain == "inventory"
        assert len(domain.tools) == 3

    def test_tool_map(self):
        domain = _make_domain()
        tm = domain.tool_map()
        assert "get_inventory" in tm
        assert "no_tool" in tm

    def test_required_params(self):
        domain = _make_domain()
        tool = domain.tool_map()["get_inventory"]
        assert tool.required_params() == ["product_id"]

    def test_to_prompt_list(self):
        domain = _make_domain()
        prompt_list = domain.to_prompt_list()
        assert len(prompt_list) == 3
        inv = next(t for t in prompt_list if t["name"] == "get_inventory")
        assert inv["parameters"]["product_id"]["required"] is True
        assert inv["parameters"]["warehouse"]["required"] is False

    def test_from_dict_valid(self):
        data = {
            "domain": "test",
            "tools": [
                {"name": "foo", "description": "bar", "parameters": {}}
            ],
        }
        spec = DomainSpec.model_validate(data)
        assert spec.domain == "test"

    def test_from_dict_missing_tools_raises(self):
        with pytest.raises(Exception):
            DomainSpec.model_validate({"domain": "test"})

    def test_version_defaults(self):
        data = {"domain": "test", "tools": []}
        spec = DomainSpec.model_validate(data)
        assert spec.version == "1.0"



class TestValidateAgainstSpec:
    def setup_method(self):
        self.tool_map = _make_domain().tool_map()

    def test_valid_call(self):
        call = {"name": "get_inventory", "parameters": {"product_id": "SKU-1"}}
        assert validate_against_spec(call, self.tool_map) == []

    def test_valid_with_optional(self):
        call = {"name": "get_inventory", "parameters": {"product_id": "SKU-1", "warehouse": "WH-A"}}
        assert validate_against_spec(call, self.tool_map) == []

    def test_missing_required_param(self):
        call = {"name": "get_inventory", "parameters": {}}
        errors = validate_against_spec(call, self.tool_map)
        assert any("product_id" in e for e in errors)

    def test_hallucinated_param(self):
        call = {"name": "get_inventory", "parameters": {"product_id": "SKU-1", "color": "red"}}
        errors = validate_against_spec(call, self.tool_map)
        assert any("color" in e for e in errors)

    def test_unknown_tool(self):
        call = {"name": "explode_warehouse", "parameters": {}}
        errors = validate_against_spec(call, self.tool_map)
        assert any("unknown tool" in e for e in errors)

    def test_no_tool_is_valid(self):
        call = {"name": "no_tool", "parameters": {}}
        assert validate_against_spec(call, self.tool_map) == []



class TestFeedbackStore:
    def test_append_and_fetch(self, tmp_path):
        from feedback.store import (
            CriticVerdict,
            FeedbackEntry,
            FeedbackStore,
            SignalType,
        )

        db = tmp_path / "test.db"
        entry = FeedbackEntry(
            specialist_id="inventory-v1",
            sub_query="how many SKU-1 in WH-A?",
            model_output={"name": "get_inventory", "parameters": {"product_id": "SKU-1"}},
            critic_verdict=CriticVerdict.FLAG,
            critic_reason="hallucinated warehouse param",
            signal_type=SignalType.ADOPTION_DECISION,
        )

        with FeedbackStore(db) as store:
            store.append(entry)
            failures = store.failures_since("inventory-v1", "2000-01-01T00:00:00+00:00")

        assert len(failures) == 1
        assert failures[0].specialist_id == "inventory-v1"
        assert failures[0].critic_verdict == CriticVerdict.FLAG

    def test_count_failures(self, tmp_path):
        from feedback.store import (
            CriticVerdict,
            FeedbackEntry,
            FeedbackStore,
            SignalType,
        )

        db = tmp_path / "test.db"
        with FeedbackStore(db) as store:
            for i in range(5):
                store.append(
                    FeedbackEntry(
                        specialist_id="inventory-v1",
                        sub_query=f"query {i}",
                        model_output={"name": "no_tool", "parameters": {}},
                        critic_verdict=CriticVerdict.FLAG,
                        signal_type=SignalType.MISSING_KNOWLEDGE,
                    )
                )
            count = store.count_failures_since("inventory-v1", "2000-01-01T00:00:00+00:00")

        assert count == 5

    def test_pass_verdicts_excluded_from_failures(self, tmp_path):
        from feedback.store import (
            CriticVerdict,
            FeedbackEntry,
            FeedbackStore,
            SignalType,
        )

        db = tmp_path / "test.db"
        with FeedbackStore(db) as store:
            store.append(
                FeedbackEntry(
                    specialist_id="inventory-v1",
                    sub_query="fine query",
                    model_output={"name": "get_inventory", "parameters": {"product_id": "X"}},
                    critic_verdict=CriticVerdict.PASS,
                    signal_type=SignalType.ADOPTION_DECISION,
                )
            )
            failures = store.failures_since("inventory-v1", "2000-01-01T00:00:00+00:00")

        assert len(failures) == 0
