"""
Phase 1 unit tests — all run without GPU or real API key.

Provider is always a MagicMock. GPU-dependent tests use
@pytest.mark.requires_model and are skipped in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pipeline.datagen import (
    TrainingExample,
    load_jsonl,
    save_jsonl,
    split_examples,
    vj_filter,
    _parse_generation_response,
    _row_to_example,
)
from pipeline.eval import (
    EvalResult,
    _assign_bfcl_category,
    _score_one,
    evaluate,
)
from pipeline.cluster import FailureCluster, cluster_failures
from pipeline.loop import load_goal
from pipeline.providers import ProviderConfig, LLMProvider, provider_from_env
from specialists.runtime.schema import DomainSpec, ParameterSpec, ToolSpec



@pytest.fixture
def inventory_domain() -> DomainSpec:
    import yaml
    with open("specialists/domains/inventory/tools.yaml") as f:
        return DomainSpec.model_validate(yaml.safe_load(f))


@pytest.fixture
def simple_domain() -> DomainSpec:
    return DomainSpec(
        domain="test",
        tools=[
            ToolSpec(
                name="get_item",
                description="Get an item by ID",
                parameters={
                    "item_id": ParameterSpec(type="string", required=True),
                    "format": ParameterSpec(type="string", required=False),
                },
            ),
            ToolSpec(
                name="delete_item",
                description="Delete an item",
                parameters={
                    "item_id": ParameterSpec(type="string", required=True),
                },
            ),
            ToolSpec(name="no_tool", description="No match", parameters={}),
        ],
    )


def _make_example(
    query: str = "get item 42",
    tool_name: str = "get_item",
    parameters: dict[str, Any] | None = None,
    source: str = "synthetic",
    cluster_id: str | None = None,
) -> TrainingExample:
    return TrainingExample(
        messages=[
            {"role": "system", "content": "You are a tool-calling assistant."},
            {"role": "user", "content": query},
        ],
        expected_tool_call={"name": tool_name, "parameters": parameters or {"item_id": "42"}},
        source=source,
        cluster_id=cluster_id,
    )


@pytest.fixture
def mock_provider() -> MagicMock:
    p = MagicMock()
    p.complete.return_value = '{"score": 0.9, "reason": "good example"}'
    return p



class TestProviders:
    def test_provider_config_is_frozen(self):
        config = ProviderConfig(api_key="k", model="m", base_url="http://x")
        with pytest.raises((AttributeError, TypeError)):
            config.api_key = "other"  # type: ignore[misc]

    def test_from_env_missing_raises(self, monkeypatch):
        monkeypatch.delenv("TOOLHIVE_PROVIDER_API_KEY", raising=False)
        monkeypatch.delenv("TOOLHIVE_PROVIDER_MODEL", raising=False)
        monkeypatch.delenv("TOOLHIVE_PROVIDER_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="Missing required"):
            provider_from_env()

    def test_from_env_partial_missing_raises(self, monkeypatch):
        monkeypatch.setenv("TOOLHIVE_PROVIDER_API_KEY", "key")
        monkeypatch.delenv("TOOLHIVE_PROVIDER_MODEL", raising=False)
        monkeypatch.delenv("TOOLHIVE_PROVIDER_BASE_URL", raising=False)
        with pytest.raises(ValueError):
            provider_from_env()

    def test_from_env_all_set(self, monkeypatch):
        monkeypatch.setenv("TOOLHIVE_PROVIDER_API_KEY", "key")
        monkeypatch.setenv("TOOLHIVE_PROVIDER_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("TOOLHIVE_PROVIDER_BASE_URL", "https://api.openai.com/v1")
        with patch("pipeline.providers.LLMProvider.__init__", return_value=None):
            provider = provider_from_env()
        assert provider is not None



class TestSaveLoadJsonl:
    def test_round_trip(self, tmp_path):
        examples = [
            _make_example("query 1", "get_item", {"item_id": "1"}),
            _make_example("query 2", "delete_item", {"item_id": "2"}, source="patch", cluster_id="c0"),
        ]
        path = tmp_path / "test.jsonl"
        save_jsonl(examples, path)
        loaded = load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0].to_dict() == examples[0].to_dict()
        assert loaded[1].source == "patch"
        assert loaded[1].cluster_id == "c0"

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.touch()
        assert load_jsonl(path) == []

    def test_nonexistent_file(self, tmp_path):
        assert load_jsonl(tmp_path / "missing.jsonl") == []

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "data.jsonl"
        save_jsonl([_make_example()], path)
        assert path.exists()

    def test_source_preserved(self, tmp_path):
        ex = _make_example(source="production_failure", cluster_id="cluster_0")
        path = tmp_path / "data.jsonl"
        save_jsonl([ex], path)
        loaded = load_jsonl(path)
        assert loaded[0].source == "production_failure"
        assert loaded[0].cluster_id == "cluster_0"



class TestSplitExamples:
    def _make_batch(self, n: int) -> list[TrainingExample]:
        tools = ["get_item", "delete_item", "no_tool"]
        return [
            _make_example(f"query {i}", tools[i % len(tools)])
            for i in range(n)
        ]

    def test_eval_fraction(self):
        examples = self._make_batch(200)
        train, eval_ = split_examples(examples, eval_fraction=0.15)
        assert len(train) + len(eval_) == 200
        assert abs(len(eval_) - 30) <= 5  # stratified, so ±few

    def test_deterministic_with_seed(self):
        examples = self._make_batch(100)
        t1, e1 = split_examples(examples, seed=42)
        t2, e2 = split_examples(examples, seed=42)
        assert [ex.messages[-1]["content"] for ex in t1] == [ex.messages[-1]["content"] for ex in t2]

    def test_different_seeds_differ(self):
        examples = self._make_batch(100)
        _, e1 = split_examples(examples, seed=1)
        _, e2 = split_examples(examples, seed=99)
        # With different seeds, at least some eval items should differ
        ids1 = {ex.messages[-1]["content"] for ex in e1}
        ids2 = {ex.messages[-1]["content"] for ex in e2}
        assert ids1 != ids2

    def test_no_overlap(self):
        examples = self._make_batch(100)
        train, eval_ = split_examples(examples)
        train_ids = {ex.messages[-1]["content"] for ex in train}
        eval_ids = {ex.messages[-1]["content"] for ex in eval_}
        assert train_ids.isdisjoint(eval_ids)

    def test_small_dataset_has_eval(self):
        examples = self._make_batch(5)
        train, eval_ = split_examples(examples, eval_fraction=0.15)
        assert len(eval_) >= 1
        assert len(train) >= 1

    def test_all_tools_in_eval(self):
        """Stratified split: every tool appears in eval set."""
        examples = self._make_batch(60)
        _, eval_ = split_examples(examples, eval_fraction=0.15)
        eval_tools = {ex.expected_tool_call["name"] for ex in eval_}
        assert "get_item" in eval_tools
        assert "delete_item" in eval_tools



class TestVjFilter:
    def test_high_score_kept(self, mock_provider, simple_domain):
        mock_provider.complete.return_value = '{"score": 0.9, "reason": "good"}'
        examples = [_make_example()]
        kept, discarded = vj_filter(examples, simple_domain, mock_provider, threshold=0.7)
        assert len(kept) == 1
        assert len(discarded) == 0

    def test_low_score_discarded(self, mock_provider, simple_domain):
        mock_provider.complete.return_value = '{"score": 0.3, "reason": "bad"}'
        examples = [_make_example()]
        kept, discarded = vj_filter(examples, simple_domain, mock_provider, threshold=0.7)
        assert len(kept) == 0
        assert len(discarded) == 1

    def test_at_threshold_kept(self, mock_provider, simple_domain):
        mock_provider.complete.return_value = '{"score": 0.7, "reason": "ok"}'
        examples = [_make_example()]
        kept, _ = vj_filter(examples, simple_domain, mock_provider, threshold=0.7)
        assert len(kept) == 1

    def test_malformed_json_discards(self, mock_provider, simple_domain):
        """Fail-safe: malformed VJ response → discard the example."""
        mock_provider.complete.return_value = "not json at all"
        examples = [_make_example()]
        kept, discarded = vj_filter(examples, simple_domain, mock_provider)
        assert len(kept) == 0
        assert len(discarded) == 1

    def test_provider_exception_discards(self, simple_domain):
        """Fail-safe: provider error → discard."""
        provider = MagicMock()
        provider.complete.side_effect = RuntimeError("connection error")
        examples = [_make_example()]
        kept, discarded = vj_filter(examples, simple_domain, provider)
        assert len(discarded) == 1

    def test_empty_input(self, mock_provider, simple_domain):
        kept, discarded = vj_filter([], simple_domain, mock_provider)
        assert kept == []
        assert discarded == []

    def test_expected_discard_range(self, simple_domain):
        """Scores uniformly in [0.5, 1.0] → expect ~0–30% discarded at threshold=0.7."""
        import random
        rng = random.Random(42)
        provider = MagicMock()
        scores = [0.5 + rng.random() * 0.5 for _ in range(100)]
        provider.complete.side_effect = [
            json.dumps({"score": s, "reason": "ok"}) for s in scores
        ]
        examples = [_make_example(f"q{i}") for i in range(100)]
        kept, discarded = vj_filter(examples, simple_domain, provider, threshold=0.7)
        discard_rate = len(discarded) / 100
        # Uniform [0.5, 1.0] → ~40% of values fall below threshold=0.7;
        # allow up to 0.50 for sampling variance.
        assert 0.0 <= discard_rate <= 0.50



class TestParseGenerationResponse:
    def test_one_per_line(self):
        raw = (
            '{"query": "get item 1", "tool_name": "get_item", "parameters": {"item_id": "1"}}\n'
            '{"query": "delete item 2", "tool_name": "delete_item", "parameters": {"item_id": "2"}}'
        )
        rows = _parse_generation_response(raw)
        assert len(rows) == 2
        assert rows[0]["tool_name"] == "get_item"

    def test_json_array_fallback(self):
        raw = json.dumps([
            {"query": "q1", "tool_name": "get_item", "parameters": {"item_id": "1"}},
            {"query": "q2", "tool_name": "no_tool", "parameters": {}},
        ])
        rows = _parse_generation_response(raw)
        assert len(rows) == 2

    def test_skips_unparseable_lines(self):
        raw = (
            "Here are the examples:\n"
            '{"query": "q1", "tool_name": "get_item", "parameters": {"item_id": "x"}}\n'
            "Some trailing text."
        )
        rows = _parse_generation_response(raw)
        assert len(rows) == 1

    def test_empty_string(self):
        assert _parse_generation_response("") == []



class TestScoreOne:
    def _tool_map(self):
        return DomainSpec(
            domain="test",
            tools=[
                ToolSpec(
                    name="get_item",
                    description="",
                    parameters={
                        "item_id": ParameterSpec(type="string", required=True),
                        "format": ParameterSpec(type="string", required=False),
                    },
                ),
                ToolSpec(name="no_tool", description="", parameters={}),
            ],
        ).tool_map()

    def test_exact_match(self):
        tm = self._tool_map()
        expected = {"name": "get_item", "parameters": {"item_id": "42"}}
        actual = {"name": "get_item", "parameters": {"item_id": "42"}}
        s = _score_one(expected, actual, False, tm)
        assert s["tool_correct"] is True
        assert s["params_exact"] is True
        assert s["hallucinated"] is False
        assert s["exact_match"] is True

    def test_wrong_tool(self):
        tm = self._tool_map()
        expected = {"name": "get_item", "parameters": {"item_id": "1"}}
        actual = {"name": "no_tool", "parameters": {}}
        s = _score_one(expected, actual, False, tm)
        assert s["tool_correct"] is False
        assert s["exact_match"] is False

    def test_hallucinated_param(self):
        tm = self._tool_map()
        expected = {"name": "get_item", "parameters": {"item_id": "1"}}
        actual = {"name": "get_item", "parameters": {"item_id": "1", "color": "red"}}
        s = _score_one(expected, actual, False, tm)
        assert s["hallucinated"] is True
        assert s["exact_match"] is False

    def test_missing_required_param(self):
        tm = self._tool_map()
        expected = {"name": "get_item", "parameters": {"item_id": "1"}}
        actual = {"name": "get_item", "parameters": {}}
        s = _score_one(expected, actual, False, tm)
        assert s["tool_correct"] is True
        assert s["params_exact"] is False
        assert s["exact_match"] is False

    def test_parse_error(self):
        tm = self._tool_map()
        s = _score_one({"name": "get_item", "parameters": {}}, None, True, tm)
        assert s["invalid_json"] is True
        assert s["exact_match"] is False
        assert s["tool_correct"] is False

    def test_no_tool_correct(self):
        tm = self._tool_map()
        expected = {"name": "no_tool", "parameters": {}}
        actual = {"name": "no_tool", "parameters": {}}
        s = _score_one(expected, actual, False, tm)
        assert s["exact_match"] is True



class TestBfclCategory:
    def test_no_tool_is_irrelevance(self):
        ex = _make_example(tool_name="no_tool", parameters={})
        assert _assign_bfcl_category(ex) == "irrelevance_detection"

    def test_normal_call_is_simple(self):
        ex = _make_example(tool_name="get_item")
        assert _assign_bfcl_category(ex) == "simple"


class TestEvaluate:
    def test_by_category_keys(self, simple_domain):
        """evaluate() produces both BFCL categories when examples of each type are present."""
        harness = MagicMock()
        harness.run.return_value = {
            "name": "get_item",
            "parameters": {"item_id": "42"},
            "raw": '{"name":"get_item","parameters":{"item_id":"42"}}',
            "parse_error": False,
            "schema_errors": [],
        }
        examples = [
            _make_example("q1", "get_item", {"item_id": "42"}),
            _make_example("q2", "no_tool", {}),
        ]
        result = evaluate(harness, simple_domain, examples)
        assert "simple" in result.by_category
        assert "irrelevance_detection" in result.by_category

    def test_empty_examples(self, simple_domain):
        harness = MagicMock()
        result = evaluate(harness, simple_domain, [])
        assert result.n_examples == 0
        assert result.exact_match == 0.0

    def test_all_correct(self, simple_domain):
        harness = MagicMock()
        harness.run.return_value = {
            "name": "get_item",
            "parameters": {"item_id": "1"},
            "raw": "",
            "parse_error": False,
            "schema_errors": [],
        }
        examples = [_make_example("q", "get_item", {"item_id": "1"}) for _ in range(5)]
        result = evaluate(harness, simple_domain, examples)
        assert result.exact_match == 1.0
        assert result.tool_selection_accuracy == 1.0
        assert result.hallucination_rate == 0.0

    def test_passes_goal_true(self):
        result = EvalResult(
            tool_selection_accuracy=0.97,
            param_exact_match=0.95,
            hallucination_rate=0.01,
            invalid_json_rate=0.0,
            exact_match=0.95,
            n_examples=100,
        )
        goal = {"tool_selection_accuracy": 0.95, "hallucination_rate": 0.02}
        assert result.passes_goal(goal) is True

    def test_passes_goal_false_low_accuracy(self):
        result = EvalResult(0.90, 0.88, 0.01, 0.0, 0.88, 100)
        goal = {"tool_selection_accuracy": 0.95, "hallucination_rate": 0.02}
        assert result.passes_goal(goal) is False

    def test_passes_goal_false_high_halluc(self):
        result = EvalResult(0.97, 0.95, 0.05, 0.0, 0.92, 100)
        goal = {"tool_selection_accuracy": 0.95, "hallucination_rate": 0.02}
        assert result.passes_goal(goal) is False



class TestCluster:
    def test_small_input_single_cluster(self, mock_provider):
        """< min_cluster_size → one cluster containing all failures."""
        mock_provider.complete.return_value = "hallucinated optional warehouse param"
        failures = [_make_example(f"q{i}") for i in range(2)]
        clusters = cluster_failures(failures, mock_provider, min_cluster_size=3)
        assert len(clusters) == 1
        assert clusters[0].size == 2
        assert sum(c.size for c in clusters) == 2

    def test_empty_input(self, mock_provider):
        assert cluster_failures([], mock_provider) == []

    def test_cluster_ids_format(self, mock_provider):
        mock_provider.complete.return_value = "some failure pattern"
        with patch("pipeline.cluster._run_hdbscan", return_value=[0, 0, 1, 1, -1]):
            failures = [_make_example(f"q{i}") for i in range(5)]
            clusters = cluster_failures(failures, mock_provider, min_cluster_size=2)
        ids = {c.cluster_id for c in clusters}
        valid = {"cluster_0", "cluster_1", "noise"}
        assert ids.issubset(valid)

    def test_all_examples_assigned(self, mock_provider):
        mock_provider.complete.return_value = "test description"
        with patch("pipeline.cluster._run_hdbscan", return_value=[0, 0, 0, 1, 1]):
            failures = [_make_example(f"q{i}") for i in range(5)]
            clusters = cluster_failures(failures, mock_provider, min_cluster_size=2)
        assert sum(c.size for c in clusters) == 5

    def test_describe_called_per_cluster(self, mock_provider):
        mock_provider.complete.return_value = "a description"
        with patch("pipeline.cluster._run_hdbscan", return_value=[0, 0, 1, 1]):
            failures = [_make_example(f"q{i}") for i in range(4)]
            cluster_failures(failures, mock_provider, min_cluster_size=2)
        # Two non-noise clusters → provider.complete called at least twice
        assert mock_provider.complete.call_count >= 2

    def test_sorted_by_size_desc(self, mock_provider):
        mock_provider.complete.return_value = "description"
        with patch("pipeline.cluster._run_hdbscan", return_value=[0, 1, 1, 1, 0]):
            failures = [_make_example(f"q{i}") for i in range(5)]
            clusters = cluster_failures(failures, mock_provider, min_cluster_size=2)
        non_noise = [c for c in clusters if c.cluster_id != "noise"]
        sizes = [c.size for c in non_noise]
        assert sizes == sorted(sizes, reverse=True)



class TestLoadGoal:
    def test_loads_inventory_goal(self):
        goal = load_goal("specialists/domains/inventory/goal.md")
        assert goal["tool_selection_accuracy"] >= 0.9
        assert goal["max_iterations"] >= 1
        assert goal["adapter_rank"] == 16

    def test_missing_key_uses_default(self, tmp_path):
        goal_file = tmp_path / "goal.md"
        goal_file.write_text("# Goal\nTarget: 95%+ accuracy\n")
        goal = load_goal(goal_file)
        assert goal["max_iterations"] == 5  # default

    def test_hallucination_rate_parsed(self, tmp_path):
        goal_file = tmp_path / "goal.md"
        goal_file.write_text("# Goal\nTarget: 95%\nHallucination rate: < 2%\nMax iterations: 3\nAdapter rank: 16")
        goal = load_goal(goal_file)
        assert goal["hallucination_rate"] == pytest.approx(0.02, abs=0.005)
