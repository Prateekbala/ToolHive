"""
Phase 4 — Live accuracy monitor.

After a new adapter is promoted, the scheduler periodically calls
check_live_accuracy() to detect post-promotion accuracy degradation.
If the live failure rate (critic FLAG + BLOCK verdicts / total requests)
exceeds rollback_threshold, the adapter is automatically rolled back.

Architecture.md §2.5:
  "Demotes/rolls back automatically if a promoted adapter's live accuracy
  (via critic verdicts) drops post-deployment."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from feedback.store import FeedbackStore
    from specialists.registry import SpecialistRegistry

# Default: roll back if more than 20% of recent requests result in FLAG or BLOCK
_DEFAULT_ROLLBACK_THRESHOLD = 0.20
# Minimum requests to have data before considering a rollback
_MIN_REQUESTS_FOR_DECISION = 10


@dataclass
class MonitorResult:
    specialist_id: str
    n_total: int
    n_failures: int
    failure_rate: float
    threshold: float
    should_rollback: bool
    reason: str


def check_live_accuracy(
    specialist_id: str,
    feedback_store: "FeedbackStore",
    since_timestamp: str,
    rollback_threshold: float = _DEFAULT_ROLLBACK_THRESHOLD,
) -> MonitorResult:
    """
    Compute the live failure rate for a specialist from the feedback store.

    A "failure" is any entry with critic_verdict == 'flag' or 'block'.
    If the failure rate exceeds rollback_threshold AND there are at least
    _MIN_REQUESTS_FOR_DECISION entries, should_rollback=True.

    Args:
        specialist_id: The specialist to monitor.
        feedback_store: Connected FeedbackStore instance.
        since_timestamp: Only look at entries after this ISO timestamp
                         (typically the promotion timestamp).
        rollback_threshold: Failure rate above which rollback is triggered.
    """
    n_total = feedback_store.count_since(specialist_id, since_timestamp)
    n_failures = feedback_store.count_failures_since(specialist_id, since_timestamp)

    if n_total == 0:
        return MonitorResult(
            specialist_id=specialist_id,
            n_total=0,
            n_failures=0,
            failure_rate=0.0,
            threshold=rollback_threshold,
            should_rollback=False,
            reason="no data since promotion — skipping rollback check",
        )

    failure_rate = n_failures / n_total

    if n_total < _MIN_REQUESTS_FOR_DECISION:
        return MonitorResult(
            specialist_id=specialist_id,
            n_total=n_total,
            n_failures=n_failures,
            failure_rate=failure_rate,
            threshold=rollback_threshold,
            should_rollback=False,
            reason=f"insufficient data ({n_total} < {_MIN_REQUESTS_FOR_DECISION} min requests)",
        )

    should_rollback = failure_rate > rollback_threshold
    reason = (
        f"failure rate {failure_rate:.1%} exceeds threshold {rollback_threshold:.1%}"
        if should_rollback
        else f"failure rate {failure_rate:.1%} within threshold {rollback_threshold:.1%}"
    )

    return MonitorResult(
        specialist_id=specialist_id,
        n_total=n_total,
        n_failures=n_failures,
        failure_rate=failure_rate,
        threshold=rollback_threshold,
        should_rollback=should_rollback,
        reason=reason,
    )


def maybe_rollback(
    monitor_result: MonitorResult,
    registry: "SpecialistRegistry",
) -> bool:
    """
    If monitor_result.should_rollback is True, roll back the specialist
    in the registry and return True.  Otherwise returns False.
    """
    if not monitor_result.should_rollback:
        return False
    try:
        registry.rollback(monitor_result.specialist_id)
        return True
    except KeyError:
        return False
