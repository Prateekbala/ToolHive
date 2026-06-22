"""
Phase 1 — Failure clustering.

Embeds failed examples with sentence-transformers, clusters with HDBSCAN,
and generates a short description per cluster via the provider.

This converts a flat list of failures into a structured taxonomy of *why*
the specialist is failing, enabling targeted patch-example generation
instead of generic data augmentation.

Reused for both:
  - Initial training (cluster synthetic failures to patch)
  - Ongoing retraining (cluster production failures from feedback store)

NOTE: PII scrubbing before _describe_cluster() is required for production
deployments that send failure content to an external provider. Marked as
TODO inline — implement in Phase 3 alongside the critic agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.datagen import TrainingExample
    from pipeline.providers import LLMProvider

# Pinned cluster description prompt — never dynamically assembled.
_CLUSTER_DESCRIPTION_SYSTEM = """\
You are analyzing failures from a tool-calling AI assistant.

Below are examples where the assistant gave the wrong answer.
Each shows the user query and what the correct answer should have been.

Failures:
{failure_examples}

In one sentence (max 15 words), describe the common failure pattern.
Focus on what specifically went wrong:
  - wrong tool chosen
  - hallucinated parameter value
  - missing required parameter
  - failed to use no_tool for out-of-domain request
  - etc.

Respond with ONLY the one-sentence description, no preamble or JSON wrapper.
"""

_NOISE_DESCRIPTION = "mixed failures with no clear common pattern"


@dataclass
class FailureCluster:
    cluster_id: str       # "cluster_0", "cluster_1", ..., "noise"
    description: str      # LLM-generated one-liner
    examples: list["TrainingExample"]
    size: int


def cluster_failures(
    failures: list["TrainingExample"],
    provider: "LLMProvider",
    min_cluster_size: int = 3,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> list[FailureCluster]:
    """
    1. Embed user query from each failure using sentence-transformers.
    2. Cluster with HDBSCAN(min_cluster_size=min_cluster_size).
    3. Generate a description per cluster via provider.
    4. Return clusters sorted by size descending.

    If len(failures) < min_cluster_size, returns one cluster with all failures.
    HDBSCAN label -1 (noise) becomes cluster_id="noise".
    """
    if not failures:
        return []

    if len(failures) < min_cluster_size:
        desc = _describe_cluster(failures, provider)
        return [FailureCluster("cluster_0", desc, failures, len(failures))]

    labels = _run_hdbscan(failures, min_cluster_size, embedding_model)
    grouped: dict[int, list["TrainingExample"]] = {}
    for ex, label in zip(failures, labels):
        grouped.setdefault(label, []).append(ex)

    clusters: list[FailureCluster] = []
    for label, members in grouped.items():
        if label == -1:
            cluster_id = "noise"
            # TODO (Phase 3): scrub PII from failure content before sending
            # to external provider — replace with local model for sensitive deployments.
            desc = _describe_cluster(members, provider) if len(members) >= 2 else _NOISE_DESCRIPTION
        else:
            cluster_id = f"cluster_{label}"
            desc = _describe_cluster(members, provider)
        clusters.append(FailureCluster(cluster_id, desc, members, len(members)))

    # Sort by size descending; put noise last
    clusters.sort(key=lambda c: (c.cluster_id == "noise", -c.size))
    return clusters


def _run_hdbscan(
    failures: list["TrainingExample"],
    min_cluster_size: int,
    embedding_model: str,
) -> list[int]:
    """Embed queries and return HDBSCAN cluster labels."""
    try:
        from sentence_transformers import SentenceTransformer
        import hdbscan as hdbscan_lib
    except ImportError as e:
        raise ImportError(
            "sentence-transformers and hdbscan required: "
            "pip install 'toolhive[train]'"
        ) from e

    texts = [ex.messages[-1]["content"] for ex in failures]
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(texts, show_progress_bar=False)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        prediction_data=False,
    )
    return clusterer.fit_predict(embeddings).tolist()


def _describe_cluster(
    cluster_examples: list["TrainingExample"],
    provider: "LLMProvider",
    max_examples_in_prompt: int = 5,
) -> str:
    """Call provider to generate a one-line failure pattern description."""
    sample = cluster_examples[:max_examples_in_prompt]
    lines = []
    for ex in sample:
        query = ex.messages[-1]["content"]
        expected = json.dumps(ex.expected_tool_call, separators=(",", ":"))
        lines.append(f"Query: {query}\nExpected: {expected}")

    failure_text = "\n---\n".join(lines)
    prompt = _CLUSTER_DESCRIPTION_SYSTEM.format(failure_examples=failure_text)

    try:
        return provider.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=64,
        ).strip()
    except Exception:
        return _NOISE_DESCRIPTION
