"""
Phase 5 unit tests — all run without GPU, real LLM, or network.

Covers:
  - FeedbackStore.list_since / list_all_since
  - Metrics: active_specialists, accuracy_over_time, recent_failure_groups,
             retrain_history, router_accuracy_summary
  - Alerts: send_alert (mocked HTTP), check_and_alert threshold gate
  - FastAPI endpoints via TestClient (with dependency overrides)
  - Exit criteria: dashboard returns data for 3 concurrent specialist domains
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from feedback.store import CriticVerdict, FeedbackEntry, FeedbackStore, SignalType
from specialists.registry import SpecialistEntry, SpecialistRegistry
from scheduler.state import SchedulerState, SchedulerStateStore
from dashboard.metrics import (
    active_specialists,
    accuracy_over_time,
    recent_failure_groups,
    retrain_history,
    router_accuracy_summary,
)
from dashboard.alerts import AlertConfig, AlertPayload, check_and_alert, send_alert



def _ts(offset_hours: float = 0.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=offset_hours)).isoformat()


def _entry(
    specialist_id="inventory-v1",
    domain="inventory",
    score=0.92,
    status="active",
    version=1,
) -> SpecialistEntry:
    return SpecialistEntry(
        specialist_id=specialist_id,
        domain=domain,
        base_model="Qwen/Qwen2.5-3B-Instruct",
        adapter_path=f"adapters/{domain}/v{version}",
        tools_yaml_path=f"specialists/domains/{domain}/tools.yaml",
        eval_score=score,
        trained_at=_ts(24),
        status=status,
    )


def _fb(
    specialist_id="inventory-v1",
    verdict=CriticVerdict.PASS,
    reason="",
    query="check inventory",
    offset_hours=0.5,
) -> FeedbackEntry:
    return FeedbackEntry(
        specialist_id=specialist_id,
        sub_query=query,
        model_output={"name": "get_inventory", "parameters": {"product_id": "X"}},
        critic_verdict=verdict,
        critic_reason=reason,
        signal_type=SignalType.ADOPTION_DECISION,
        timestamp=_ts(offset_hours),
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



class TestListSince:
    def test_list_since_returns_all_verdicts(self, feedback_store):
        since = _ts(48)
        feedback_store.append(_fb(verdict=CriticVerdict.PASS))
        feedback_store.append(_fb(verdict=CriticVerdict.FLAG))
        feedback_store.append(_fb(verdict=CriticVerdict.BLOCK))
        result = feedback_store.list_since("inventory-v1", since)
        assert len(result) == 3

    def test_list_since_respects_timestamp(self, feedback_store):
        feedback_store.append(_fb())
        future_since = _ts(-1)  # 1 hour in the future
        result = feedback_store.list_since("inventory-v1", future_since)
        assert len(result) == 0

    def test_list_since_filters_by_specialist(self, feedback_store):
        since = _ts(48)
        feedback_store.append(_fb(specialist_id="inventory-v1"))
        feedback_store.append(_fb(specialist_id="email-v1"))
        result = feedback_store.list_since("inventory-v1", since)
        assert all(e.specialist_id == "inventory-v1" for e in result)
        assert len(result) == 1

    def test_list_all_since_returns_all_specialists(self, feedback_store):
        since = _ts(48)
        feedback_store.append(_fb(specialist_id="inventory-v1"))
        feedback_store.append(_fb(specialist_id="email-v1"))
        feedback_store.append(_fb(specialist_id="crm-v1"))
        result = feedback_store.list_all_since(since)
        assert len(result) == 3
        domains = {e.specialist_id for e in result}
        assert domains == {"inventory-v1", "email-v1", "crm-v1"}

    def test_list_all_since_empty_db(self, feedback_store):
        assert feedback_store.list_all_since(_ts(48)) == []



class TestActiveSpecialists:
    def test_returns_active_only(self, registry):
        registry.register(_entry(status="active"))
        registry.register(_entry(specialist_id="inventory-v2", version=2, status="candidate"))
        result = active_specialists(registry)
        assert len(result) == 1
        assert result[0]["specialist_id"] == "inventory-v1"

    def test_empty_registry(self, registry):
        assert active_specialists(registry) == []

    def test_fields_present(self, registry):
        registry.register(_entry(status="active"))
        result = active_specialists(registry)
        r = result[0]
        assert "specialist_id" in r
        assert "domain" in r
        assert "eval_score" in r
        assert "status" in r
        assert "trained_at" in r



class TestAccuracyOverTime:
    def test_empty_returns_none_accuracy(self, feedback_store):
        result = accuracy_over_time("inventory-v1", feedback_store, n_buckets=4)
        assert result["overall_accuracy"] is None
        assert result["n_total"] == 0
        assert len(result["buckets"]) == 4

    def test_all_pass(self, feedback_store):
        for _ in range(5):
            feedback_store.append(_fb(verdict=CriticVerdict.PASS, offset_hours=0.1))
        result = accuracy_over_time("inventory-v1", feedback_store, n_buckets=24)
        assert result["overall_accuracy"] == pytest.approx(1.0)
        assert result["n_total"] == 5

    def test_mixed_verdicts(self, feedback_store):
        for _ in range(8):
            feedback_store.append(_fb(verdict=CriticVerdict.PASS, offset_hours=0.1))
        for _ in range(2):
            feedback_store.append(_fb(verdict=CriticVerdict.BLOCK, offset_hours=0.1))
        result = accuracy_over_time("inventory-v1", feedback_store, n_buckets=24)
        assert result["overall_accuracy"] == pytest.approx(0.8)
        assert result["n_total"] == 10

    def test_bucket_count_matches(self, feedback_store):
        result = accuracy_over_time("inventory-v1", feedback_store, n_buckets=12)
        assert len(result["buckets"]) == 12

    def test_bucket_fields(self, feedback_store):
        feedback_store.append(_fb(offset_hours=0.1))
        result = accuracy_over_time("inventory-v1", feedback_store, n_buckets=2)
        bucket = next((b for b in result["buckets"] if b["n_total"] > 0), None)
        assert bucket is not None
        assert "bucket_start" in bucket
        assert "n_total" in bucket
        assert "n_pass" in bucket
        assert "accuracy" in bucket



class TestRecentFailureGroups:
    def test_empty_failures(self, feedback_store):
        since = _ts(48)
        result = recent_failure_groups("inventory-v1", feedback_store, since)
        assert result["n_failures"] == 0
        assert result["groups"] == []

    def test_groups_by_reason(self, feedback_store):
        since = _ts(48)
        for _ in range(3):
            feedback_store.append(_fb(verdict=CriticVerdict.BLOCK, reason="missing product_id"))
        for _ in range(2):
            feedback_store.append(_fb(verdict=CriticVerdict.FLAG, reason="wrong type"))
        result = recent_failure_groups("inventory-v1", feedback_store, since)
        assert result["n_failures"] == 5
        groups = result["groups"]
        assert groups[0]["reason"] == "missing product_id"
        assert groups[0]["count"] == 3
        assert groups[1]["count"] == 2

    def test_limit_respected(self, feedback_store):
        since = _ts(48)
        for i in range(10):
            feedback_store.append(_fb(verdict=CriticVerdict.BLOCK, reason=f"reason_{i}"))
        result = recent_failure_groups("inventory-v1", feedback_store, since, limit=5)
        assert len(result["groups"]) <= 5

    def test_example_query_present(self, feedback_store):
        since = _ts(48)
        feedback_store.append(_fb(verdict=CriticVerdict.BLOCK, reason="oops", query="what is SKU-1"))
        result = recent_failure_groups("inventory-v1", feedback_store, since)
        assert result["groups"][0]["example_query"] == "what is SKU-1"



class TestRetrainHistory:
    def test_empty_domain(self, registry, state_store):
        result = retrain_history("inventory", registry, state_store)
        assert result["domain"] == "inventory"
        assert result["n_versions"] == 0
        assert result["versions"] == []

    def test_multiple_versions(self, registry):
        registry.register(_entry(specialist_id="inventory-v1", version=1, status="rolled_back", score=0.88))
        registry.register(_entry(specialist_id="inventory-v2", version=2, status="active", score=0.95))
        result = retrain_history("inventory", registry)
        assert result["n_versions"] == 2
        # Sorted by trained_at
        assert len(result["versions"]) == 2

    def test_version_fields(self, registry):
        registry.register(_entry(status="active"))
        result = retrain_history("inventory", registry)
        v = result["versions"][0]
        assert "specialist_id" in v
        assert "eval_score" in v
        assert "status" in v
        assert "trained_at" in v

    def test_scheduler_info_included(self, registry, state_store):
        registry.register(_entry(status="active"))
        state_store.save(SchedulerState(
            specialist_id="inventory-v1",
            last_eval_score=0.92,
            retrain_count=3,
        ))
        result = retrain_history("inventory", registry, state_store)
        assert result["scheduler"]["retrain_count"] == 3
        assert result["scheduler"]["last_eval_score"] == pytest.approx(0.92)

    def test_no_state_store_returns_empty_scheduler(self, registry):
        registry.register(_entry(status="active"))
        result = retrain_history("inventory", registry, state_store=None)
        assert result["scheduler"] == {}



class TestRouterAccuracySummary:
    def test_empty_feedback(self, feedback_store):
        since = _ts(48)
        result = router_accuracy_summary(feedback_store, since)
        assert result["n_total"] == 0
        assert result["accuracy"] is None
        assert result["by_specialist"] == {}

    def test_single_specialist(self, feedback_store):
        since = _ts(48)
        for _ in range(8):
            feedback_store.append(_fb(verdict=CriticVerdict.PASS, offset_hours=0.1))
        for _ in range(2):
            feedback_store.append(_fb(verdict=CriticVerdict.BLOCK, offset_hours=0.1))
        result = router_accuracy_summary(feedback_store, since)
        assert result["n_total"] == 10
        assert result["accuracy"] == pytest.approx(0.8)

    def test_by_specialist_breakdown(self, feedback_store):
        since = _ts(48)
        for _ in range(3):
            feedback_store.append(_fb(specialist_id="inventory-v1", verdict=CriticVerdict.PASS, offset_hours=0.1))
        for _ in range(2):
            feedback_store.append(_fb(specialist_id="email-v1", verdict=CriticVerdict.FLAG, offset_hours=0.1))
        result = router_accuracy_summary(feedback_store, since)
        assert "inventory-v1" in result["by_specialist"]
        assert "email-v1" in result["by_specialist"]
        assert result["by_specialist"]["inventory-v1"]["accuracy"] == pytest.approx(1.0)
        assert result["by_specialist"]["email-v1"]["accuracy"] == pytest.approx(0.0)



class TestAlerts:
    def _config(self) -> AlertConfig:
        return AlertConfig(webhook_url="https://hooks.example.com/test", threshold=0.20)

    def test_send_alert_returns_true_on_2xx(self):
        mock_http = MagicMock(return_value=200)
        result = send_alert("inventory-v1", 0.35, self._config(), http_post_fn=mock_http)
        assert result is True
        mock_http.assert_called_once()

    def test_send_alert_returns_false_on_5xx(self):
        mock_http = MagicMock(return_value=503)
        result = send_alert("inventory-v1", 0.35, self._config(), http_post_fn=mock_http)
        assert result is False

    def test_alert_payload_fields(self):
        captured = {}

        def capture_http(url, body):
            captured["url"] = url
            captured["body"] = json.loads(body)
            return 200

        send_alert("inventory-v1", 0.35, self._config(), http_post_fn=capture_http)
        assert captured["body"]["alert_type"] == "accuracy_degradation"
        assert captured["body"]["specialist_id"] == "inventory-v1"
        assert captured["body"]["failure_rate"] == pytest.approx(0.35, abs=0.001)
        assert "timestamp" in captured["body"]

    def test_check_and_alert_fires_above_threshold(self):
        mock_http = MagicMock(return_value=200)
        fired = check_and_alert("inv-v1", 0.35, self._config(), http_post_fn=mock_http)
        assert fired is True
        mock_http.assert_called_once()

    def test_check_and_alert_no_fire_below_threshold(self):
        mock_http = MagicMock(return_value=200)
        fired = check_and_alert("inv-v1", 0.10, self._config(), http_post_fn=mock_http)
        assert fired is False
        mock_http.assert_not_called()

    def test_check_and_alert_exactly_at_threshold_no_fire(self):
        mock_http = MagicMock(return_value=200)
        fired = check_and_alert("inv-v1", 0.20, self._config(), http_post_fn=mock_http)
        assert fired is False

    def test_alert_payload_to_dict(self):
        p = AlertPayload(
            alert_type="accuracy_degradation",
            specialist_id="x-v1",
            failure_rate=0.30,
            threshold=0.20,
            timestamp="2026-06-21T00:00:00Z",
        )
        d = p.to_dict()
        assert d["specialist_id"] == "x-v1"
        assert d["failure_rate"] == pytest.approx(0.30, abs=0.001)



class TestAPI:
    """Test FastAPI endpoints using TestClient with injected store dependencies."""

    @pytest.fixture
    def client(self, registry, feedback_store, state_store):
        from fastapi.testclient import TestClient
        from dashboard.api import app, get_registry, get_feedback, get_state_store, get_alert_config

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_feedback] = lambda: feedback_store
        app.dependency_overrides[get_state_store] = lambda: state_store
        app.dependency_overrides[get_alert_config] = lambda: None

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c

        app.dependency_overrides.clear()

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_specialists_empty(self, client):
        resp = client.get("/specialists")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_specialists_returns_active(self, client, registry):
        registry.register(_entry(status="active"))
        resp = client.get("/specialists")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["specialist_id"] == "inventory-v1"

    def test_accuracy_endpoint(self, client, feedback_store):
        for _ in range(5):
            feedback_store.append(_fb(verdict=CriticVerdict.PASS, offset_hours=0.1))
        resp = client.get("/specialists/inventory-v1/accuracy?n_buckets=4&bucket_minutes=360")
        assert resp.status_code == 200
        data = resp.json()
        assert "buckets" in data
        assert len(data["buckets"]) == 4
        assert data["n_total"] == 5

    def test_failures_endpoint(self, client, feedback_store):
        feedback_store.append(_fb(verdict=CriticVerdict.BLOCK, reason="bad param"))
        resp = client.get("/specialists/inventory-v1/failures")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_failures"] == 1
        assert data["groups"][0]["reason"] == "bad param"

    def test_history_endpoint_not_found(self, client):
        resp = client.get("/specialists/nonexistent-v1/history")
        assert resp.status_code == 404

    def test_history_endpoint_found(self, client, registry):
        registry.register(_entry(status="active"))
        resp = client.get("/specialists/inventory-v1/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["domain"] == "inventory"

    def test_domain_history_endpoint(self, client, registry):
        registry.register(_entry(status="active"))
        resp = client.get("/domains/inventory/history")
        assert resp.status_code == 200
        assert resp.json()["domain"] == "inventory"

    def test_router_accuracy_endpoint(self, client, feedback_store):
        for _ in range(3):
            feedback_store.append(_fb(verdict=CriticVerdict.PASS, offset_hours=0.1))
        feedback_store.append(_fb(verdict=CriticVerdict.FLAG, offset_hours=0.1))
        resp = client.get("/router/accuracy")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_total"] == 4
        assert data["accuracy"] == pytest.approx(0.75)

    def test_alert_test_no_config(self, client):
        resp = client.post("/alerts/test", json={"specialist_id": "x", "failure_rate": 0.5})
        assert resp.status_code == 422

    def test_alert_test_with_config(self, client, registry, feedback_store, state_store):
        from fastapi.testclient import TestClient as TC
        from dashboard.api import app, get_alert_config
        import dashboard.alerts as alerts_mod

        captured = {}

        def mock_http(url, body):
            captured["called"] = True
            return True  # _urllib_post returns bool, not status code

        config = AlertConfig(webhook_url="https://example.com/hook", threshold=0.20)
        original = alerts_mod._urllib_post
        alerts_mod._urllib_post = mock_http
        app.dependency_overrides[get_alert_config] = lambda: config

        try:
            with TC(app) as c:
                resp = c.post("/alerts/test", json={"specialist_id": "inv-v1", "failure_rate": 0.50})
        finally:
            alerts_mod._urllib_post = original
            app.dependency_overrides[get_alert_config] = lambda: None

        assert resp.status_code == 200
        assert resp.json()["fired"] is True

    def test_html_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "ToolHive Dashboard" in resp.text
        assert "text/html" in resp.headers["content-type"]



class TestExitCriteria:
    """
    Phase 5 exit criteria:
      Dashboard reflects real data from Phases 1–4 end to end on at least
      3 specialist domains running concurrently.
    """

    DOMAINS = ["inventory", "email", "crm"]

    def _populate(self, registry, feedback_store, state_store):
        for i, domain in enumerate(self.DOMAINS):
            sid = f"{domain}-v1"
            e = SpecialistEntry(
                specialist_id=sid,
                domain=domain,
                base_model="Qwen/Qwen2.5-3B-Instruct",
                adapter_path=f"adapters/{domain}/v1",
                tools_yaml_path=f"specialists/domains/{domain}/tools.yaml",
                eval_score=0.90 + i * 0.02,
                trained_at=_ts(24),
                status="active",
            )
            registry.register(e)
            state_store.save(SchedulerState(
                specialist_id=sid,
                last_eval_score=0.90 + i * 0.02,
                retrain_count=i + 1,
            ))
            # Add a mix of pass and fail feedback
            for _ in range(8):
                feedback_store.append(FeedbackEntry(
                    specialist_id=sid,
                    sub_query=f"query for {domain}",
                    model_output={"name": "tool", "parameters": {}},
                    critic_verdict=CriticVerdict.PASS,
                    signal_type=SignalType.ADOPTION_DECISION,
                    timestamp=_ts(0.5),
                ))
            for _ in range(2):
                feedback_store.append(FeedbackEntry(
                    specialist_id=sid,
                    sub_query=f"bad query for {domain}",
                    model_output={"name": "wrong", "parameters": {}},
                    critic_verdict=CriticVerdict.BLOCK,
                    critic_reason="wrong tool selected",
                    signal_type=SignalType.MISSING_KNOWLEDGE,
                    timestamp=_ts(0.5),
                ))

    def test_three_domains_visible_in_specialists_list(self, registry, feedback_store, state_store):
        self._populate(registry, feedback_store, state_store)
        result = active_specialists(registry)
        assert len(result) == 3
        domains = {r["domain"] for r in result}
        assert domains == set(self.DOMAINS)

    def test_accuracy_available_per_domain(self, registry, feedback_store, state_store):
        self._populate(registry, feedback_store, state_store)
        for domain in self.DOMAINS:
            sid = f"{domain}-v1"
            result = accuracy_over_time(sid, feedback_store)
            assert result["n_total"] == 10
            assert result["overall_accuracy"] == pytest.approx(0.80)

    def test_router_accuracy_covers_all_domains(self, registry, feedback_store, state_store):
        self._populate(registry, feedback_store, state_store)
        since = _ts(48)
        result = router_accuracy_summary(feedback_store, since)
        assert result["n_total"] == 30   # 10 per domain * 3 domains
        assert result["accuracy"] == pytest.approx(0.80)
        for domain in self.DOMAINS:
            assert f"{domain}-v1" in result["by_specialist"]

    def test_retrain_history_per_domain(self, registry, feedback_store, state_store):
        self._populate(registry, feedback_store, state_store)
        for i, domain in enumerate(self.DOMAINS):
            result = retrain_history(domain, registry, state_store)
            assert result["n_versions"] == 1
            assert result["scheduler"]["retrain_count"] == i + 1

    def test_failure_groups_per_domain(self, registry, feedback_store, state_store):
        self._populate(registry, feedback_store, state_store)
        since = _ts(48)
        for domain in self.DOMAINS:
            sid = f"{domain}-v1"
            result = recent_failure_groups(sid, feedback_store, since)
            assert result["n_failures"] == 2
            assert result["groups"][0]["reason"] == "wrong tool selected"

    def test_api_three_domains(self, registry, feedback_store, state_store):
        from fastapi.testclient import TestClient
        from dashboard.api import app, get_registry, get_feedback, get_state_store, get_alert_config

        self._populate(registry, feedback_store, state_store)

        app.dependency_overrides[get_registry] = lambda: registry
        app.dependency_overrides[get_feedback] = lambda: feedback_store
        app.dependency_overrides[get_state_store] = lambda: state_store
        app.dependency_overrides[get_alert_config] = lambda: None

        try:
            with TestClient(app) as client:
                # /specialists lists all 3
                specs = client.get("/specialists").json()
                assert len(specs) == 3

                # /router/accuracy covers all 3
                router = client.get("/router/accuracy").json()
                assert router["n_total"] == 30
                assert len(router["by_specialist"]) == 3

                # per-domain accuracy and history work for all 3
                for domain in self.DOMAINS:
                    sid = f"{domain}-v1"
                    acc = client.get(f"/specialists/{sid}/accuracy?n_buckets=4").json()
                    assert acc["n_total"] == 10

                    hist = client.get(f"/domains/{domain}/history").json()
                    assert hist["n_versions"] == 1
        finally:
            app.dependency_overrides.clear()
