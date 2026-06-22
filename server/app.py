"""
ToolHive inference server — exposes the Router + Orchestrator + Critic as HTTP.

Environment variables
---------------------
TOOLHIVE_REGISTRY_DB       path to registry.db  (default: registry.db)
TOOLHIVE_FEEDBACK_DB       path to feedback.db  (default: feedback.db)
TOOLHIVE_BASE_MODEL        HuggingFace model id used when GPU adapters are loaded
                           (default: Qwen/Qwen2.5-3B-Instruct)
TOOLHIVE_PROVIDER_BASE_URL OpenAI-compatible base URL for the provider API
TOOLHIVE_PROVIDER_MODEL    Model name sent to the provider API
TOOLHIVE_PROVIDER_API_KEY  API key; if set, ProviderHarness is used for inference
                           instead of loading local GPU adapters
TOOLHIVE_CRITIC_ENABLED    1 / true / yes to enable critic (default: 1)

Usage (dev)
-----------
  uvicorn server.app:app --reload --port 8000

Usage (docker-compose)
----------------------
  See docker-compose.yml — the "server" service builds and runs this.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel


_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry_path = os.environ.get("TOOLHIVE_REGISTRY_DB", "registry.db")
    feedback_path = os.environ.get("TOOLHIVE_FEEDBACK_DB", "feedback.db")
    base_model = os.environ.get("TOOLHIVE_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    critic_enabled = os.environ.get("TOOLHIVE_CRITIC_ENABLED", "1").lower() in {"1", "true", "yes"}

    from specialists.registry import SpecialistRegistry
    from feedback.store import FeedbackStore
    from router.router import Router
    from router.orchestrator import Orchestrator
    from critic.verifier import CriticVerifier

    registry = SpecialistRegistry(db_path=registry_path)
    registry.connect()
    feedback_store = FeedbackStore(db_path=feedback_path)
    feedback_store.connect()

    harness_factory = _build_harness_factory()

    router = Router(registry=registry)
    router.build_index()

    critic = CriticVerifier(provider=_build_provider()) if critic_enabled else None

    orchestrator = Orchestrator(
        registry=registry,
        base_model=base_model,
        critic=critic,
        feedback_store=feedback_store,
        harness_factory=harness_factory,
    )

    _state["registry"] = registry
    _state["feedback_store"] = feedback_store
    _state["router"] = router
    _state["orchestrator"] = orchestrator

    yield

    _state.clear()
    registry.close()
    feedback_store.close()


def _build_provider():
    """Return an LLMProvider if env vars are configured, else None."""
    api_key = os.environ.get("TOOLHIVE_PROVIDER_API_KEY", "")
    if not api_key:
        return None
    from pipeline.providers import LLMProvider, ProviderConfig
    config = ProviderConfig(
        base_url=os.environ.get("TOOLHIVE_PROVIDER_BASE_URL", "https://api.openai.com/v1"),
        model=os.environ.get("TOOLHIVE_PROVIDER_MODEL", "gpt-4o-mini"),
        api_key=api_key,
    )
    return LLMProvider(config)


def _build_harness_factory():
    """
    If TOOLHIVE_PROVIDER_API_KEY is set, return a factory that creates
    ProviderHarness instances (no GPU needed).  Otherwise return None so the
    orchestrator falls back to ToolCallHarness.
    """
    provider = _build_provider()
    if provider is None:
        return None
    from specialists.runtime.provider_harness import ProviderHarness
    def factory(_entry):
        return ProviderHarness(provider=provider)
    return factory



app = FastAPI(
    title="ToolHive",
    description="Self-healing swarm of specialist tool-calling agents",
    version="0.1.0",
    lifespan=lifespan,
)



def get_registry():
    return _state["registry"]


def get_feedback():
    return _state["feedback_store"]


def get_router():
    return _state["router"]


def get_orchestrator():
    return _state["orchestrator"]



class InvokeRequest(BaseModel):
    request: str
    session_id: str | None = None   # reserved for future multi-turn support


class InvokeResponse(BaseModel):
    success: bool
    plan: dict[str, Any]
    results: list[dict[str, Any]]



@app.get("/health")
def health(registry=Depends(get_registry)):
    active = []
    try:
        active = [e.specialist_id for e in registry.list_active()]
    except Exception:
        pass
    return {"status": "ok", "active_specialists": active}


@app.get("/domains")
def list_domains(registry=Depends(get_registry)):
    entries = registry.list_active()
    return [
        {
            "specialist_id": e.specialist_id,
            "domain": e.domain,
            "eval_score": e.eval_score,
            "trained_at": e.trained_at,
        }
        for e in entries
    ]


@app.post("/invoke", response_model=InvokeResponse)
def invoke(
    body: InvokeRequest,
    router=Depends(get_router),
    orchestrator=Depends(get_orchestrator),
):
    if not body.request.strip():
        raise HTTPException(status_code=422, detail="request must not be empty")

    plan = router.route(body.request)
    result = orchestrator.execute(plan)
    data = result.to_dict()
    return InvokeResponse(
        success=data["success"],
        plan=data["plan"],
        results=data["results"],
    )


@app.post("/reload")
def reload_index(router=Depends(get_router)):
    """Rebuild the router embedding index after new specialists are registered."""
    router.build_index()
    return {"status": "ok"}
