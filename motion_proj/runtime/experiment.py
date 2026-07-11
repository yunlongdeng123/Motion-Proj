"""零交互 run manifest、JSONL 指标和 SQLite 实验注册表。"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .atomic import atomic_write_json

RUN_STATES = {"queued", "running", "retrying", "completed", "pruned", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunManifest:
    run_id: str
    command: list[str]
    config_fingerprint: str
    cache_fingerprint: str
    seed: int
    git: dict[str, Any]
    environment: dict[str, Any]
    data_split: str | None = None
    parent_run_id: str | None = None
    started_at: str = field(default_factory=utc_now)
    ended_at: str | None = None
    exit_reason: str | None = None
    status: str = "running"

    def save(self, path: str) -> None:
        atomic_write_json(path, asdict(self))


class JsonlMetrics:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def append(self, step: int, values: dict[str, Any]) -> None:
        row = {"time": utc_now(), "step": int(step), **values}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class ExperimentRegistry:
    """本地 SQLite 注册表；写操作使用事务且无需登录服务。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, status TEXT NOT NULL, parent_run_id TEXT,
                    config_fingerprint TEXT NOT NULL, work_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    exit_reason TEXT, summary_json TEXT
                )"""
            )

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def register(self, run_id: str, status: str, config_fingerprint: str, work_dir: str,
                 parent_run_id: str | None = None) -> None:
        self._check_state(status)
        now = utc_now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (run_id, status, parent_run_id, config_fingerprint, work_dir, now, now),
            )

    def update(self, run_id: str, status: str, *, exit_reason: str | None = None,
               summary: dict | None = None) -> None:
        self._check_state(status)
        with self._connect() as db:
            cur = db.execute(
                "UPDATE runs SET status=?, updated_at=?, exit_reason=?, summary_json=? WHERE run_id=?",
                (status, utc_now(), exit_reason, json.dumps(summary) if summary is not None else None, run_id),
            )
            if cur.rowcount != 1:
                raise KeyError(run_id)

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT run_id,status,parent_run_id,config_fingerprint,work_dir,created_at,updated_at,exit_reason,summary_json FROM runs"
        params: tuple = ()
        if status:
            self._check_state(status)
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at"
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = [dict(row) for row in db.execute(query, params)]
        for row in rows:
            row["summary"] = json.loads(row.pop("summary_json")) if row.get("summary_json") else None
        return rows

    @staticmethod
    def _check_state(status: str) -> None:
        if status not in RUN_STATES:
            raise ValueError(f"未知实验状态: {status}")
