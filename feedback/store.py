"""
Feedback store — append-only SQLite log of production requests, specialist
outputs, and critic verdicts.

This is the raw material for the retrain flywheel. No manual labeling required;
critic flags + user corrections substitute for labels.

Four signal types are tracked per entry (AITL taxonomy, EMNLP 2025):
  adoption_decision   – tool call was accepted or overridden by the caller
  pairwise_preference – critic A/B comparison with a corrected alternative
  knowledge_relevance – right tool intent, wrong parameters
  missing_knowledge   – no tool matched (triggers tools.yaml review, not retrain)

Schema is append-only. Never UPDATE or DELETE rows — the full history is the
ground truth for the retrain scheduler.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class SignalType(str, Enum):
    ADOPTION_DECISION = "adoption_decision"
    PAIRWISE_PREFERENCE = "pairwise_preference"
    KNOWLEDGE_RELEVANCE = "knowledge_relevance"
    MISSING_KNOWLEDGE = "missing_knowledge"


class CriticVerdict(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    BLOCK = "block"


@dataclass
class FeedbackEntry:
    specialist_id: str
    sub_query: str
    model_output: dict[str, Any]
    critic_verdict: CriticVerdict
    signal_type: SignalType
    critic_reason: str = ""
    vj_quality_score: float | None = None
    user_feedback: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    request_id      TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    specialist_id   TEXT NOT NULL,
    sub_query       TEXT NOT NULL,
    model_output    TEXT NOT NULL,   -- JSON
    critic_verdict  TEXT NOT NULL,
    critic_reason   TEXT NOT NULL DEFAULT '',
    signal_type     TEXT NOT NULL,
    vj_quality_score REAL,
    user_feedback   TEXT
);
CREATE INDEX IF NOT EXISTS idx_specialist ON feedback (specialist_id);
CREATE INDEX IF NOT EXISTS idx_verdict    ON feedback (critic_verdict);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON feedback (timestamp);
"""


class FeedbackStore:
    def __init__(self, db_path: str | Path = "feedback.db"):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FeedbackStore":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def append(self, entry: FeedbackEntry) -> None:
        assert self._conn, "call connect() first"
        self._conn.execute(
            """
            INSERT INTO feedback
              (request_id, timestamp, specialist_id, sub_query, model_output,
               critic_verdict, critic_reason, signal_type, vj_quality_score, user_feedback)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                entry.request_id,
                entry.timestamp,
                entry.specialist_id,
                entry.sub_query,
                json.dumps(entry.model_output),
                entry.critic_verdict.value,
                entry.critic_reason,
                entry.signal_type.value,
                entry.vj_quality_score,
                entry.user_feedback,
            ),
        )
        self._conn.commit()

    def failures_since(
        self,
        specialist_id: str,
        since_timestamp: str,
        verdicts: tuple[str, ...] = ("flag", "block"),
    ) -> list[FeedbackEntry]:
        """
        Fetch all flagged/blocked entries for a specialist since a given timestamp.
        Used by the retrain scheduler to pull new failures for clustering.
        """
        assert self._conn
        placeholders = ",".join("?" * len(verdicts))
        rows = self._conn.execute(
            f"""
            SELECT request_id, timestamp, specialist_id, sub_query, model_output,
                   critic_verdict, critic_reason, signal_type, vj_quality_score, user_feedback
            FROM feedback
            WHERE specialist_id = ?
              AND timestamp > ?
              AND critic_verdict IN ({placeholders})
            ORDER BY timestamp ASC
            """,
            (specialist_id, since_timestamp, *verdicts),
        ).fetchall()

        return [_row_to_entry(r) for r in rows]

    def count_failures_since(self, specialist_id: str, since_timestamp: str) -> int:
        assert self._conn
        row = self._conn.execute(
            """
            SELECT COUNT(*) FROM feedback
            WHERE specialist_id = ? AND timestamp > ?
              AND critic_verdict IN ('flag', 'block')
            """,
            (specialist_id, since_timestamp),
        ).fetchone()
        return row[0] if row else 0

    def count_since(self, specialist_id: str, since_timestamp: str) -> int:
        """Count ALL entries (pass + flag + block) for a specialist since a timestamp."""
        assert self._conn
        row = self._conn.execute(
            "SELECT COUNT(*) FROM feedback WHERE specialist_id = ? AND timestamp > ?",
            (specialist_id, since_timestamp),
        ).fetchone()
        return row[0] if row else 0

    def list_since(
        self,
        specialist_id: str,
        since_timestamp: str,
        limit: int = 5000,
    ) -> list[FeedbackEntry]:
        """Return all entries (any verdict) for a specialist since a timestamp."""
        assert self._conn
        rows = self._conn.execute(
            """
            SELECT request_id, timestamp, specialist_id, sub_query, model_output,
                   critic_verdict, critic_reason, signal_type, vj_quality_score, user_feedback
            FROM feedback
            WHERE specialist_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (specialist_id, since_timestamp, limit),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def list_all_since(
        self,
        since_timestamp: str,
        limit: int = 10000,
    ) -> list[FeedbackEntry]:
        """Return all entries across all specialists since a timestamp."""
        assert self._conn
        rows = self._conn.execute(
            """
            SELECT request_id, timestamp, specialist_id, sub_query, model_output,
                   critic_verdict, critic_reason, signal_type, vj_quality_score, user_feedback
            FROM feedback
            WHERE timestamp > ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (since_timestamp, limit),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]


def _row_to_entry(row: tuple[Any, ...]) -> FeedbackEntry:
    (
        request_id, timestamp, specialist_id, sub_query, model_output_json,
        critic_verdict, critic_reason, signal_type, vj_quality_score, user_feedback,
    ) = row
    return FeedbackEntry(
        request_id=request_id,
        timestamp=timestamp,
        specialist_id=specialist_id,
        sub_query=sub_query,
        model_output=json.loads(model_output_json),
        critic_verdict=CriticVerdict(critic_verdict),
        critic_reason=critic_reason,
        signal_type=SignalType(signal_type),
        vj_quality_score=vj_quality_score,
        user_feedback=user_feedback,
    )
