"""
Phase 2 — Router / planner agent.

Routes an incoming request to the correct specialist(s) using embedding
similarity between the request and each specialist's tool-description corpus.

Architecture (from architecture.md §2.1):
  v1: embedding similarity + simple multi-step heuristics
  v2 candidate: small trained classifier once enough routing data exists

Multi-step heuristic: if the request contains step-connector phrases
("then", "and then", "after that", etc.), the request is split into
sub-queries and each is routed independently.

Requires sentence-transformers: pip install 'toolhive[train]'
Falls back to keyword-overlap scoring if sentence-transformers is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from specialists.registry import SpecialistRegistry, SpecialistEntry

# Step-connector patterns — longer alternatives before shorter ones to avoid
# partial matches (e.g., "and then" before "then").
_STEP_CONNECTOR = re.compile(
    r"\s*(?:,\s*)?(?:and then|after that|after which|followed by|subsequently"
    r"|afterwards|,?\s*then|;\s*)\s*",
    re.IGNORECASE,
)

# Minimum cosine similarity to accept a routing match.
# Below this the router emits a "no_specialist" step instead of a wrong one.
_MIN_SIMILARITY = 0.15


@dataclass
class DispatchStep:
    specialist_id: str
    domain: str
    sub_query: str
    similarity_score: float = 0.0


@dataclass
class DispatchPlan:
    request: str
    steps: list[DispatchStep]
    is_multi_step: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "is_multi_step": self.is_multi_step,
            "steps": [
                {
                    "specialist_id": s.specialist_id,
                    "domain": s.domain,
                    "sub_query": s.sub_query,
                    "similarity_score": round(s.similarity_score, 4),
                }
                for s in self.steps
            ],
        }


class Router:
    """
    Embedding-similarity router.

    Call build_index() once after constructing (or whenever active
    specialists change) before routing any requests.
    """

    def __init__(
        self,
        registry: "SpecialistRegistry",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._registry = registry
        self._embedding_model = embedding_model
        self._embedder: Any = None
        self._entries: list["SpecialistEntry"] = []
        self._corpus_embeddings: Any = None  # np.ndarray shape (n_specialists, dim)


    def build_index(self) -> None:
        """
        Load all active specialists and compute their corpus embeddings.
        Must be called before route().
        """
        self._entries = self._registry.list_active()
        if not self._entries:
            self._corpus_embeddings = None
            return

        corpora = [_build_corpus(e) for e in self._entries]
        self._embedder = _load_embedder(self._embedding_model)
        if self._embedder is not None:
            self._corpus_embeddings = self._embedder.encode(
                corpora, show_progress_bar=False, normalize_embeddings=True
            )
        else:
            # Fallback: store raw corpora for keyword scoring
            self._corpus_embeddings = corpora


    def route(self, request: str) -> DispatchPlan:
        """
        Route a request to one or more specialists.
        Returns a DispatchPlan with one step per sub-query.
        """
        if not self._entries:
            return DispatchPlan(request=request, steps=[], is_multi_step=False)

        multi = _is_multi_step(request)
        sub_queries = _split_request(request) if multi else [request]

        steps = [self._route_single(q) for q in sub_queries]
        return DispatchPlan(request=request, steps=steps, is_multi_step=multi)

    def _route_single(self, query: str) -> DispatchStep:
        """Find the best specialist for a single query."""
        if self._embedder is not None:
            import numpy as np
            q_emb = self._embedder.encode(
                [query], show_progress_bar=False, normalize_embeddings=True
            )[0]
            scores = (self._corpus_embeddings @ q_emb).tolist()
        else:
            # Keyword-overlap fallback
            scores = [
                _keyword_overlap(query, corpus)
                for corpus in self._corpus_embeddings
            ]

        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_score = scores[best_idx]
        best_entry = self._entries[best_idx]

        if best_score < _MIN_SIMILARITY:
            return DispatchStep(
                specialist_id="no_specialist",
                domain="unknown",
                sub_query=query,
                similarity_score=best_score,
            )

        return DispatchStep(
            specialist_id=best_entry.specialist_id,
            domain=best_entry.domain,
            sub_query=query,
            similarity_score=best_score,
        )



def _build_corpus(entry: "SpecialistEntry") -> str:
    """
    Build the text corpus that represents a specialist for embedding.
    Concatenates domain name + all tool names + descriptions + param names.
    Reads tools.yaml if the file exists; gracefully skips if not.
    """
    parts = [f"Domain: {entry.domain}"]
    tools_path = Path(entry.tools_yaml_path)
    if tools_path.exists():
        try:
            import yaml
            with tools_path.open() as f:
                data = yaml.safe_load(f)
            for tool in data.get("tools", []):
                if tool.get("name") == "no_tool":
                    continue
                parts.append(f"Tool: {tool['name']}")
                if tool.get("description"):
                    parts.append(tool["description"])
                for param_name, param in tool.get("parameters", {}).items():
                    parts.append(param_name)
                    if isinstance(param, dict) and param.get("description"):
                        parts.append(param["description"])
        except Exception:
            pass  # tools.yaml unreadable — fall back to domain name only
    return " ".join(parts)


def _is_multi_step(request: str) -> bool:
    """Return True if the request contains step-connector phrases."""
    return bool(_STEP_CONNECTOR.search(request))


def _split_request(request: str) -> list[str]:
    """
    Split a multi-step request into individual sub-queries.
    Each sub-query is stripped and filtered for minimum length.
    """
    parts = _STEP_CONNECTOR.split(request)
    sub_queries = [p.strip() for p in parts if p and p.strip()]
    # Filter out very short fragments (likely split artefacts)
    return [q for q in sub_queries if len(q.split()) >= 2] or [request]


def _load_embedder(model_name: str) -> Any:
    """Load sentence-transformers embedder; returns None if not installed."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        return None


def _keyword_overlap(query: str, corpus: str) -> float:
    """
    Fallback scoring when sentence-transformers is unavailable.
    Query-coverage metric: fraction of query words that appear in the corpus.
    Recall-oriented so that long corpora (many tool descriptions) are not
    penalised the way Jaccard similarity would penalise them.
    """
    q_words = set(query.lower().split())
    c_words = set(corpus.lower().split())
    if not q_words or not c_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)
