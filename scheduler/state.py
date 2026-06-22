"""
Phase 4 — Scheduler state store.

Persists per-specialist retrain metadata: when the last cycle ran, what eval
score the active adapter achieved, how many retrains have happened, and where
the stable held-out eval JSONL lives (for regression testing across retrains).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS scheduler_state (
    specialist_id      TEXT PRIMARY KEY,
    last_run_timestamp TEXT NOT NULL DEFAULT '1970-01-01T00:00:00Z',
    last_eval_score    REAL NOT NULL DEFAULT 0.0,
    retrain_count      INTEGER NOT NULL DEFAULT 0,
    train_data_path    TEXT NOT NULL DEFAULT '',
    eval_data_path     TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class SchedulerState:
    specialist_id: str
    last_run_timestamp: str = "1970-01-01T00:00:00Z"
    last_eval_score: float = 0.0
    retrain_count: int = 0
    train_data_path: str = ""   # path to train.jsonl (appended each cycle)
    eval_data_path: str = ""    # path to eval.jsonl  (stable held-out set)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SchedulerStateStore:
    def __init__(self, db_path: str | Path = "scheduler_state.db") -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "SchedulerStateStore":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def get(self, specialist_id: str) -> SchedulerState | None:
        assert self._conn
        row = self._conn.execute(
            "SELECT * FROM scheduler_state WHERE specialist_id=?", (specialist_id,)
        ).fetchone()
        return _row_to_state(row) if row else None

    def save(self, state: SchedulerState) -> None:
        assert self._conn
        self._conn.execute(
            """
            INSERT OR REPLACE INTO scheduler_state
              (specialist_id, last_run_timestamp, last_eval_score,
               retrain_count, train_data_path, eval_data_path)
            VALUES (?,?,?,?,?,?)
            """,
            (
                state.specialist_id, state.last_run_timestamp, state.last_eval_score,
                state.retrain_count, state.train_data_path, state.eval_data_path,
            ),
        )
        self._conn.commit()

    def list_all(self) -> list[SchedulerState]:
        assert self._conn
        rows = self._conn.execute(
            "SELECT * FROM scheduler_state ORDER BY specialist_id"
        ).fetchall()
        return [_row_to_state(r) for r in rows]


def _row_to_state(row: sqlite3.Row) -> SchedulerState:
    return SchedulerState(
        specialist_id=row["specialist_id"],
        last_run_timestamp=row["last_run_timestamp"],
        last_eval_score=row["last_eval_score"],
        retrain_count=row["retrain_count"],
        train_data_path=row["train_data_path"],
        eval_data_path=row["eval_data_path"],
    )
