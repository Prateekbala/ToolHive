"""
Phase 5 — ToolHive dashboard FastAPI app.

Endpoints:
  GET  /health                              — liveness probe
  GET  /specialists                         — list all active specialists
  GET  /specialists/{id}/accuracy           — accuracy over time (24h by default)
  GET  /specialists/{id}/failures           — recent failure groups
  GET  /specialists/{id}/history            — retrain version history
  GET  /router/accuracy                     — overall + per-specialist pass rate
  POST /alerts/test                         — fire a test alert to the webhook
  GET  /                                    — minimal HTML dashboard

Configuration (env vars):
  TOOLHIVE_REGISTRY_DB   path to registry.db   (default: registry.db)
  TOOLHIVE_FEEDBACK_DB   path to feedback.db   (default: feedback.db)
  TOOLHIVE_STATE_DB      path to scheduler_state.db (default: scheduler_state.db)
  TOOLHIVE_ALERT_WEBHOOK webhook URL for alerts (optional)
  TOOLHIVE_ALERT_THRESHOLD failure-rate threshold (default: 0.20)

Run:
  uvicorn dashboard.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from specialists.registry import SpecialistRegistry
from feedback.store import FeedbackStore
from scheduler.state import SchedulerStateStore
from dashboard.metrics import (
    active_specialists,
    accuracy_over_time,
    recent_failure_groups,
    retrain_history,
    router_accuracy_summary,
)
from dashboard.alerts import AlertConfig, check_and_alert



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = SpecialistRegistry(
        os.environ.get("TOOLHIVE_REGISTRY_DB", "registry.db")
    )
    app.state.registry.connect()

    app.state.feedback = FeedbackStore(
        os.environ.get("TOOLHIVE_FEEDBACK_DB", "feedback.db")
    )
    app.state.feedback.connect()

    app.state.state_store = SchedulerStateStore(
        os.environ.get("TOOLHIVE_STATE_DB", "scheduler_state.db")
    )
    app.state.state_store.connect()

    app.state.alert_config = _build_alert_config()
    yield

    app.state.registry.close()
    app.state.feedback.close()
    app.state.state_store.close()


def _build_alert_config() -> AlertConfig | None:
    webhook = os.environ.get("TOOLHIVE_ALERT_WEBHOOK")
    if not webhook:
        return None
    threshold = float(os.environ.get("TOOLHIVE_ALERT_THRESHOLD", "0.20"))
    return AlertConfig(webhook_url=webhook, threshold=threshold)



app = FastAPI(
    title="ToolHive Dashboard",
    description="Observability dashboard for the ToolHive specialist swarm",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



def get_registry(request: Request) -> SpecialistRegistry:
    return request.app.state.registry


def get_feedback(request: Request) -> FeedbackStore:
    return request.app.state.feedback


def get_state_store(request: Request) -> SchedulerStateStore:
    return request.app.state.state_store


def get_alert_config(request: Request) -> AlertConfig | None:
    return request.app.state.alert_config



class AlertTestRequest(BaseModel):
    specialist_id: str
    failure_rate: float



@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/specialists", response_model=list[dict])
def list_specialists(
    registry: SpecialistRegistry = Depends(get_registry),
) -> list[dict[str, Any]]:
    return active_specialists(registry)


@app.get("/specialists/{specialist_id}/accuracy")
def specialist_accuracy(
    specialist_id: str,
    n_buckets: int = 24,
    bucket_minutes: int = 60,
    feedback: FeedbackStore = Depends(get_feedback),
) -> dict[str, Any]:
    return accuracy_over_time(specialist_id, feedback, n_buckets, bucket_minutes)


@app.get("/specialists/{specialist_id}/failures")
def specialist_failures(
    specialist_id: str,
    hours: int = 24,
    limit: int = 20,
    feedback: FeedbackStore = Depends(get_feedback),
) -> dict[str, Any]:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return recent_failure_groups(specialist_id, feedback, since, limit)


@app.get("/specialists/{specialist_id}/history")
def specialist_history(
    specialist_id: str,
    registry: SpecialistRegistry = Depends(get_registry),
    state_store: SchedulerStateStore = Depends(get_state_store),
) -> dict[str, Any]:
    entry = registry.get(specialist_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"specialist {specialist_id!r} not found")
    return retrain_history(entry.domain, registry, state_store)


@app.get("/domains/{domain}/history")
def domain_history(
    domain: str,
    registry: SpecialistRegistry = Depends(get_registry),
    state_store: SchedulerStateStore = Depends(get_state_store),
) -> dict[str, Any]:
    return retrain_history(domain, registry, state_store)


@app.get("/router/accuracy")
def router_accuracy(
    hours: int = 24,
    feedback: FeedbackStore = Depends(get_feedback),
) -> dict[str, Any]:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return router_accuracy_summary(feedback, since)


@app.post("/alerts/test")
def test_alert(
    body: AlertTestRequest,
    alert_config: AlertConfig | None = Depends(get_alert_config),
) -> dict[str, Any]:
    if alert_config is None:
        raise HTTPException(
            status_code=422,
            detail="No alert webhook configured. Set TOOLHIVE_ALERT_WEBHOOK env var.",
        )
    fired = check_and_alert(
        specialist_id=body.specialist_id,
        failure_rate=body.failure_rate,
        config=alert_config,
    )
    return {"fired": fired, "specialist_id": body.specialist_id, "failure_rate": body.failure_rate}


@app.get("/", response_class=HTMLResponse)
def dashboard_ui() -> str:
    return _DASHBOARD_HTML



_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ToolHive Dashboard</title>
  <style>
    body { font-family: monospace; background: #0d1117; color: #c9d1d9; margin: 2rem; }
    h1 { color: #58a6ff; }
    h2 { color: #79c0ff; border-bottom: 1px solid #21262d; padding-bottom: 0.3rem; }
    .card { background: #161b22; border: 1px solid #21262d; border-radius: 6px;
            padding: 1rem; margin: 1rem 0; }
    .pass  { color: #3fb950; }
    .flag  { color: #d29922; }
    .block { color: #f85149; }
    .active { color: #3fb950; }
    .rolled_back { color: #8b949e; }
    table { border-collapse: collapse; width: 100%; }
    th { text-align: left; color: #8b949e; font-size: 0.85rem; padding: 0.3rem 0.5rem; }
    td { padding: 0.3rem 0.5rem; border-top: 1px solid #21262d; }
    #loading { color: #8b949e; }
  </style>
</head>
<body>
  <h1>ToolHive Dashboard</h1>
  <div id="loading">Loading...</div>
  <div id="content" style="display:none">
    <h2>Active Specialists</h2>
    <div class="card">
      <table>
        <thead><tr><th>ID</th><th>Domain</th><th>Eval Score</th><th>Status</th><th>Trained</th></tr></thead>
        <tbody id="specialists-table"></tbody>
      </table>
    </div>
    <h2>Router Accuracy (24h)</h2>
    <div class="card" id="router-accuracy"></div>
  </div>
  <script>
    async function load() {
      const [specs, router] = await Promise.all([
        fetch('/specialists').then(r => r.json()),
        fetch('/router/accuracy').then(r => r.json()),
      ]);

      const tbody = document.getElementById('specialists-table');
      for (const s of specs) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${s.specialist_id}</td>
          <td>${s.domain}</td>
          <td>${(s.eval_score * 100).toFixed(1)}%</td>
          <td class="${s.status}">${s.status}</td>
          <td>${s.trained_at.substring(0, 10)}</td>
        `;
        tbody.appendChild(tr);
      }

      const acc = router.accuracy !== null ? (router.accuracy * 100).toFixed(1) + '%' : 'no data';
      document.getElementById('router-accuracy').innerHTML =
        `<b>Overall:</b> ${acc} &nbsp; <b>Total requests:</b> ${router.n_total} &nbsp; ` +
        `<b>Pass:</b> ${router.n_pass} &nbsp; <b>Fail:</b> ${router.n_total - router.n_pass}`;

      document.getElementById('loading').style.display = 'none';
      document.getElementById('content').style.display = 'block';
    }
    load().catch(e => {
      document.getElementById('loading').textContent = 'Error loading data: ' + e;
    });
  </script>
</body>
</html>"""
