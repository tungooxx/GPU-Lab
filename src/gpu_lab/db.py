import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from .models import Instance, Job


class Repository:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS instances (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS artifacts (id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL, checksum TEXT NOT NULL, mime_type TEXT, created_at TEXT NOT NULL, UNIQUE(job_id,path));
        CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, job_id TEXT, event_type TEXT NOT NULL, message TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY, tool_name TEXT NOT NULL, arguments_json TEXT NOT NULL, outcome TEXT NOT NULL, error_message TEXT, duration_ms INTEGER NOT NULL, created_at TEXT NOT NULL);
        """)
        self.conn.commit()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def save_instance(self, item: Instance) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO instances VALUES (?,?,?)",
                (item.id, item.model_dump_json(), self._now()),
            )
            self.conn.commit()

    def get_instance(self, instance_id: str) -> Instance | None:
        with self._lock:
            row = self.conn.execute("SELECT data FROM instances WHERE id=?", (instance_id,)).fetchone()
        return Instance.model_validate_json(row["data"]) if row else None

    def list_instances(self) -> list[Instance]:
        with self._lock:
            return [
                Instance.model_validate_json(r["data"])
                for r in self.conn.execute("SELECT data FROM instances ORDER BY updated_at DESC")
            ]

    def save_job(self, item: Job) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?,?,?)",
                (item.job_id, item.model_dump_json(), self._now()),
            )
            self.conn.commit()

    def claim_job(self, item: Job) -> tuple[str, Job]:
        """Atomically claim a job ID across gateway processes.

        ``claimed`` means this caller owns submission. ``existing`` means another
        caller already owns or launched the identical request. An ``unknown`` job
        is reclaimable because startup reconciliation has established that its
        prior submitter cannot prove a live process.
        """
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                row = self.conn.execute(
                    "SELECT data FROM jobs WHERE job_id=?", (item.job_id,)
                ).fetchone()
                if row:
                    existing = Job.model_validate_json(row["data"])
                    if existing.metadata.get("request_fingerprint") != item.metadata.get(
                        "request_fingerprint"
                    ):
                        self.conn.rollback()
                        return "conflict", existing
                    if existing.status != "unknown":
                        self.conn.commit()
                        return "existing", existing
                    self.conn.execute(
                        "UPDATE jobs SET data=?,updated_at=? WHERE job_id=?",
                        (item.model_dump_json(), self._now(), item.job_id),
                    )
                else:
                    self.conn.execute(
                        "INSERT INTO jobs VALUES (?,?,?)",
                        (item.job_id, item.model_dump_json(), self._now()),
                    )
                self.conn.commit()
                return "claimed", item
            except Exception:
                self.conn.rollback()
                raise

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self.conn.execute("SELECT data FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return Job.model_validate_json(row["data"]) if row else None

    def list_jobs(
        self, instance_id: str | None = None, status: str | None = None, limit: int = 50
    ) -> list[Job]:
        with self._lock:
            jobs = [
                Job.model_validate_json(r["data"])
                for r in self.conn.execute("SELECT data FROM jobs ORDER BY updated_at DESC")
            ]
        return [
            j
            for j in jobs
            if (not instance_id or j.instance_id == instance_id)
            and (not status or j.status == status)
        ][:limit]

    def event(
        self, job_id: str | None, event_type: str, message: str, metadata: dict | None = None
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(job_id,event_type,message,metadata_json,created_at) VALUES(?,?,?,?,?)",
                (job_id, event_type, message, json.dumps(metadata or {}), self._now()),
            )
            self.conn.commit()

    def audit(
        self,
        tool_name: str,
        arguments: dict,
        outcome: str,
        duration_ms: int,
        error_message: str | None = None,
    ) -> None:
        with self._lock:
            try:
                self.conn.execute(
                    "INSERT INTO audit_log(tool_name,arguments_json,outcome,error_message,duration_ms,created_at) VALUES(?,?,?,?,?,?)",
                    (tool_name, json.dumps(arguments, default=str), outcome, error_message, duration_ms, self._now()),
                )
                self.conn.commit()
            except sqlite3.Error:
                self.conn.rollback()

    def list_audit(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT tool_name,arguments_json,outcome,error_message,duration_ms,created_at FROM audit_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "tool": row["tool_name"],
                    "arguments": json.loads(row["arguments_json"]),
                    "outcome": row["outcome"],
                    "error": row["error_message"],
                    "duration_ms": row["duration_ms"],
                    "at": row["created_at"],
                }
                for row in rows
            ]
