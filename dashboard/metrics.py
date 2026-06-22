"""
Phase 5 — Dashboard metrics aggregation.

Pure functions over the existing stores — no side effects, no HTTP.

All functions return plain dicts so the FastAPI layer can JSON-serialize them
without any special encoder.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from feedback.store import FeedbackStore, FeedbackEntry
    from specialists.registry import SpecialistRegistry
    from scheduler.state import SchedulerStateStore

_PASS_VERDICTS = {"pass"}
_FAIL_VERDICTS = {"flag", "block"}



def active_specialists(registry: "SpecialistRegistry") -> list[dict[str, Any]]:
    """Return all active specialists with their current metadata."""
    return [
        {
            "specialist_id": e.specialist_id,
            "domain": e.domain,
            "base_model": e.base_model,
            "eval_score": round(e.eval_score, 4),
            "status": e.status,
            "trained_at": e.trained_at,
            "adapter_path": e.adapter_path,
        }
        for e in registry.list_active()
    ]



def accuracy_over_time(
    specialist_id: str,
    feedback_store: "FeedbackStore",
    n_buckets: int = 24,
    bucket_minutes: int = 60,
) -> dict[str, Any]:
    """
    Compute per-bucket accuracy (pass rate) for a specialist over the last
    n_buckets * bucket_minutes of feedback entries.

    Returns:
        {
          "specialist_id": ...,
          "bucket_minutes": ...,
          "buckets": [{"bucket_start": ISO, "n_total": int, "n_pass": int, "accuracy": float}],
          "overall_accuracy": float,
          "n_total": int,
        }
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=n_buckets * bucket_minutes)
    since = window_start.isoformat()

    entries = feedback_store.list_since(specialist_id, since)

    # Build bucket boundaries
    bucket_starts = [
        window_start + timedelta(minutes=i * bucket_minutes)
        for i in range(n_buckets)
    ]
    bucket_ends = bucket_starts[1:] + [now]

    buckets = []
    for b_start, b_end in zip(bucket_starts, bucket_ends):
        b_entries = [
            e for e in entries
            if b_start <= _parse_ts(e.timestamp) < b_end
        ]
        n_total = len(b_entries)
        n_pass = sum(1 for e in b_entries if e.critic_verdict.value in _PASS_VERDICTS)
        buckets.append({
            "bucket_start": b_start.isoformat(),
            "n_total": n_total,
            "n_pass": n_pass,
            "accuracy": round(n_pass / n_total, 4) if n_total else None,
        })

    n_total = len(entries)
    n_pass = sum(1 for e in entries if e.critic_verdict.value in _PASS_VERDICTS)

    return {
        "specialist_id": specialist_id,
        "bucket_minutes": bucket_minutes,
        "buckets": buckets,
        "overall_accuracy": round(n_pass / n_total, 4) if n_total else None,
        "n_total": n_total,
    }



def recent_failure_groups(
    specialist_id: str,
    feedback_store: "FeedbackStore",
    since: str,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Group recent critic failures by critic_reason.  Returns the top `limit`
    groups sorted by count descending, with the most-recent timestamp and an
    example query from each group.
    """
    failures = feedback_store.failures_since(specialist_id, since)

    groups: dict[str, dict[str, Any]] = {}
    for entry in failures:
        reason = entry.critic_reason or "(no reason)"
        if reason not in groups:
            groups[reason] = {
                "reason": reason,
                "verdict": entry.critic_verdict.value,
                "count": 0,
                "last_seen": entry.timestamp,
                "example_query": entry.sub_query,
            }
        groups[reason]["count"] += 1
        if entry.timestamp > groups[reason]["last_seen"]:
            groups[reason]["last_seen"] = entry.timestamp

    sorted_groups = sorted(groups.values(), key=lambda g: -g["count"])

    return {
        "specialist_id": specialist_id,
        "since": since,
        "n_failures": len(failures),
        "groups": sorted_groups[:limit],
    }



def retrain_history(
    domain: str,
    registry: "SpecialistRegistry",
    state_store: "SchedulerStateStore | None" = None,
) -> dict[str, Any]:
    """
    Return the full adapter version history for a domain, plus scheduler state.
    """
    all_entries = registry.list_all()
    domain_entries = [e for e in all_entries if e.domain == domain]
    domain_entries.sort(key=lambda e: e.trained_at)

    versions = [
        {
            "specialist_id": e.specialist_id,
            "eval_score": round(e.eval_score, 4),
            "status": e.status,
            "trained_at": e.trained_at,
            "adapter_path": e.adapter_path,
        }
        for e in domain_entries
    ]

    scheduler_info: dict[str, Any] = {}
    if state_store:
        active = registry.get_active(domain)
        if active:
            state = state_store.get(active.specialist_id)
            if state:
                scheduler_info = {
                    "last_run": state.last_run_timestamp,
                    "retrain_count": state.retrain_count,
                    "last_eval_score": round(state.last_eval_score, 4),
                }

    return {
        "domain": domain,
        "n_versions": len(versions),
        "versions": versions,
        "scheduler": scheduler_info,
    }



def router_accuracy_summary(
    feedback_store: "FeedbackStore",
    since: str,
) -> dict[str, Any]:
    """
    Overall and per-specialist pass rate across all feedback entries.
    """
    all_entries = feedback_store.list_all_since(since)

    by_specialist: dict[str, dict[str, int]] = defaultdict(lambda: {"n_total": 0, "n_pass": 0})
    n_total = 0
    n_pass = 0

    for entry in all_entries:
        sid = entry.specialist_id
        verdict = entry.critic_verdict.value
        by_specialist[sid]["n_total"] += 1
        n_total += 1
        if verdict in _PASS_VERDICTS:
            by_specialist[sid]["n_pass"] += 1
            n_pass += 1

    per_spec = {
        sid: {
            "n_total": counts["n_total"],
            "n_pass": counts["n_pass"],
            "accuracy": round(counts["n_pass"] / counts["n_total"], 4)
            if counts["n_total"] else None,
        }
        for sid, counts in by_specialist.items()
    }

    return {
        "since": since,
        "n_total": n_total,
        "n_pass": n_pass,
        "accuracy": round(n_pass / n_total, 4) if n_total else None,
        "by_specialist": per_spec,
    }



def _parse_ts(ts: str) -> datetime:
    """Parse an ISO timestamp, handling both Z suffix and +00:00."""
    ts = ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
