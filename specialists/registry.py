"""
Specialist registry — SQLite-backed catalogue of trained LoRA adapters.

Every trained adapter is registered here with its domain, eval score, and
lifecycle status. The router reads from this table to know which specialists
are available and what domains they cover.

Status transitions:
  candidate  →  active    (promote — demotes any existing active for the domain)
  active     →  rolled_back (rollback — used by the retrain scheduler in Phase 4)

Only one adapter per domain can be "active" at any time. Multiple "candidate"
and "rolled_back" entries may coexist for history and fast rollback.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS specialists (
    specialist_id   TEXT PRIMARY KEY,
    domain          TEXT NOT NULL,
    base_model      TEXT NOT NULL,
    adapter_path    TEXT NOT NULL,
    tools_yaml_path TEXT NOT NULL,
    eval_score      REAL NOT NULL DEFAULT 0.0,
    trained_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'candidate'
                    CHECK(status IN ('active', 'candidate', 'rolled_back'))
);
CREATE INDEX IF NOT EXISTS idx_domain_status ON specialists (domain, status);
"""

_STATUS_ACTIVE = "active"
_STATUS_CANDIDATE = "candidate"
_STATUS_ROLLED_BACK = "rolled_back"


@dataclass
class SpecialistEntry:
    specialist_id: str       # e.g. "inventory-v3"
    domain: str              # e.g. "inventory"
    base_model: str          # HuggingFace model name
    adapter_path: str        # path to LoRA adapter directory
    tools_yaml_path: str     # path to domain's tools.yaml (for router corpus)
    eval_score: float        # exact_match on held-out eval set
    trained_at: str          # ISO timestamp
    status: str = _STATUS_CANDIDATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpecialistRegistry:
    def __init__(self, db_path: str | Path = "registry.db") -> None:
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

    def __enter__(self) -> "SpecialistRegistry":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


    def register(self, entry: SpecialistEntry) -> None:
        """Insert or replace a specialist entry."""
        assert self._conn
        self._conn.execute(
            """
            INSERT OR REPLACE INTO specialists
              (specialist_id, domain, base_model, adapter_path, tools_yaml_path,
               eval_score, trained_at, status)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                entry.specialist_id, entry.domain, entry.base_model,
                entry.adapter_path, entry.tools_yaml_path, entry.eval_score,
                entry.trained_at, entry.status,
            ),
        )
        self._conn.commit()

    def promote(self, specialist_id: str) -> None:
        """
        Set specialist_id to 'active' and demote any existing active
        adapter for the same domain to 'rolled_back'.
        Raises KeyError if specialist_id is not found.
        """
        assert self._conn
        entry = self.get(specialist_id)
        if entry is None:
            raise KeyError(f"specialist_id not found: {specialist_id!r}")

        # Demote current active for this domain
        self._conn.execute(
            "UPDATE specialists SET status=? WHERE domain=? AND status=?",
            (_STATUS_ROLLED_BACK, entry.domain, _STATUS_ACTIVE),
        )
        # Activate the new one
        self._conn.execute(
            "UPDATE specialists SET status=? WHERE specialist_id=?",
            (_STATUS_ACTIVE, specialist_id),
        )
        self._conn.commit()

    def rollback(self, specialist_id: str) -> None:
        """
        Mark specialist_id as 'rolled_back' and restore the most recent
        previously-active adapter for the same domain.
        """
        assert self._conn
        entry = self.get(specialist_id)
        if entry is None:
            raise KeyError(f"specialist_id not found: {specialist_id!r}")

        self._conn.execute(
            "UPDATE specialists SET status=? WHERE specialist_id=?",
            (_STATUS_ROLLED_BACK, specialist_id),
        )

        # Restore the most recent rolled-back entry that is not this one
        prev = self._conn.execute(
            """
            SELECT specialist_id FROM specialists
            WHERE domain=? AND status=? AND specialist_id!=?
            ORDER BY trained_at DESC LIMIT 1
            """,
            (entry.domain, _STATUS_ROLLED_BACK, specialist_id),
        ).fetchone()
        if prev:
            self._conn.execute(
                "UPDATE specialists SET status=? WHERE specialist_id=?",
                (_STATUS_ACTIVE, prev["specialist_id"]),
            )
        self._conn.commit()


    def get(self, specialist_id: str) -> SpecialistEntry | None:
        assert self._conn
        row = self._conn.execute(
            "SELECT * FROM specialists WHERE specialist_id=?", (specialist_id,)
        ).fetchone()
        return _row_to_entry(row) if row else None

    def get_active(self, domain: str) -> SpecialistEntry | None:
        """Return the currently-active specialist for a domain, or None."""
        assert self._conn
        row = self._conn.execute(
            "SELECT * FROM specialists WHERE domain=? AND status=?",
            (domain, _STATUS_ACTIVE),
        ).fetchone()
        return _row_to_entry(row) if row else None

    def list_active(self) -> list[SpecialistEntry]:
        """Return all active specialists (one per domain at most)."""
        assert self._conn
        rows = self._conn.execute(
            "SELECT * FROM specialists WHERE status=? ORDER BY domain",
            (_STATUS_ACTIVE,),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def list_all(self) -> list[SpecialistEntry]:
        assert self._conn
        rows = self._conn.execute(
            "SELECT * FROM specialists ORDER BY domain, trained_at DESC"
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def next_version(self, domain: str) -> str:
        """
        Return the next specialist_id for a domain.
        Example: if 'inventory-v1' and 'inventory-v2' exist, returns 'inventory-v3'.
        """
        assert self._conn
        rows = self._conn.execute(
            "SELECT specialist_id FROM specialists WHERE domain=?", (domain,)
        ).fetchall()
        if not rows:
            return f"{domain}-v1"
        versions = []
        for r in rows:
            try:
                v = int(r["specialist_id"].rsplit("-v", 1)[-1])
                versions.append(v)
            except (ValueError, IndexError):
                pass
        next_v = max(versions, default=0) + 1
        return f"{domain}-v{next_v}"


def _row_to_entry(row: sqlite3.Row) -> SpecialistEntry:
    return SpecialistEntry(
        specialist_id=row["specialist_id"],
        domain=row["domain"],
        base_model=row["base_model"],
        adapter_path=row["adapter_path"],
        tools_yaml_path=row["tools_yaml_path"],
        eval_score=row["eval_score"],
        trained_at=row["trained_at"],
        status=row["status"],
    )
