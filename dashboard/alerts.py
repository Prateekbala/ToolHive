"""
Phase 5 — Alerting hooks.

Sends a webhook POST when a specialist's live accuracy drops below threshold.
Uses stdlib urllib so no extra runtime dependency is needed.

Inject `http_post_fn` in tests to avoid real HTTP calls.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Any


@dataclass
class AlertConfig:
    webhook_url: str
    threshold: float = 0.20
    cooldown_minutes: int = 60


@dataclass
class AlertPayload:
    alert_type: str
    specialist_id: str
    failure_rate: float
    threshold: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "specialist_id": self.specialist_id,
            "failure_rate": round(self.failure_rate, 4),
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "message": self.message,
        }


def send_alert(
    specialist_id: str,
    failure_rate: float,
    config: AlertConfig,
    http_post_fn: Callable[[str, bytes], int] | None = None,
) -> bool:
    """
    POST an accuracy-degradation alert to config.webhook_url.

    Args:
        specialist_id: The specialist whose accuracy dropped.
        failure_rate: The observed failure rate (0.0–1.0).
        config: AlertConfig with webhook URL and threshold.
        http_post_fn: Optional injectable HTTP POST function for testing.
                      Signature: (url, body_bytes) -> http_status_code.
                      Defaults to stdlib urllib.request.

    Returns:
        True if the webhook returned 2xx, False otherwise.
    """
    payload = AlertPayload(
        alert_type="accuracy_degradation",
        specialist_id=specialist_id,
        failure_rate=failure_rate,
        threshold=config.threshold,
        message=(
            f"Specialist {specialist_id!r} failure rate {failure_rate:.1%} "
            f"exceeds threshold {config.threshold:.1%} — rollback may be triggered."
        ),
    )
    body = json.dumps(payload.to_dict()).encode()

    if http_post_fn is not None:
        status = http_post_fn(config.webhook_url, body)
        return 200 <= status < 300

    return _urllib_post(config.webhook_url, body)


def _urllib_post(url: str, body: bytes) -> bool:
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def check_and_alert(
    specialist_id: str,
    failure_rate: float,
    config: AlertConfig,
    http_post_fn: Callable[[str, bytes], int] | None = None,
) -> bool:
    """
    Send an alert only if failure_rate exceeds config.threshold.
    Returns True if an alert was fired.
    """
    if failure_rate <= config.threshold:
        return False
    return send_alert(specialist_id, failure_rate, config, http_post_fn)
