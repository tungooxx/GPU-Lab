"""Durable operational state for the Chucky Lab cockpit and worker runtimes.

This module deliberately stores browser/runtime coordination separately from
Research OS scientific objects. A browser response is never evidence and a
runtime failure never changes a hypothesis or cancels an ExperimentRun.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

from .errors import GPUError
from .lab import LabController
from .research import ResearchStore

WEB_STATUSES = {
    "UNATTACHED", "STARTING", "LOGIN_REQUIRED", "READY", "PROMPT_SUBMITTED",
    "RESPONSE_IN_PROGRESS", "RESPONSE_COMPLETE", "IDLE", "ERROR", "DISCONNECTED",
}
ATTACHED_WEB_STATUSES = {
    "LOGIN_REQUIRED", "READY", "PROMPT_SUBMITTED", "RESPONSE_IN_PROGRESS", "RESPONSE_COMPLETE", "IDLE",
}
TURN_OUTCOMES = {
    "CONTINUE", "WAITING_DEPENDENCY", "IDLE", "BLOCKED", "REPLAN", "HUMAN_REQUIRED", "ERROR",
}
WAKE_STATUSES = {"PENDING", "DISPATCHED", "COMPLETED", "CANCELLED", "FAILED"}


class CockpitController:
    """Persist project controls and browser-worker operations without scientific authority."""

    def __init__(self, store: ResearchStore, lab: LabController | None = None,
                 max_turns_per_work_item: int = 20, max_consecutive_continues: int = 3):
        self.store = store
        self.lab = lab or LabController(store)
        self.max_turns_per_work_item = max_turns_per_work_item
        self.max_consecutive_continues = max_consecutive_continues
        self._migrate()
        self.recover_inflight_wakes()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _migrate(self) -> None:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_cockpit_migration'))")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lab_project_controls (
                    project_id UUID PRIMARY KEY REFERENCES research_projects(id),
                    autopilot_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    auto_continue_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    paused BOOLEAN NOT NULL DEFAULT FALSE, updated_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lab_worker_runtimes (
                    id UUID PRIMARY KEY, worker_id UUID NOT NULL REFERENCES research_workers(id),
                    worker_session_id UUID NOT NULL REFERENCES research_worker_sessions(id),
                    project_id UUID NOT NULL REFERENCES research_projects(id),
                    runtime_type TEXT NOT NULL, status TEXT NOT NULL, browser_profile_key TEXT,
                    logical_page_key TEXT, conversation_url TEXT, attached_at TIMESTAMPTZ,
                    last_seen_at TIMESTAMPTZ, last_error TEXT, created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL, UNIQUE(worker_session_id)
                );
                CREATE INDEX IF NOT EXISTS lab_worker_runtimes_project_idx
                    ON lab_worker_runtimes(project_id,status,updated_at DESC);
                CREATE TABLE IF NOT EXISTS lab_worker_turns (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    worker_id UUID NOT NULL REFERENCES research_workers(id),
                    worker_session_id UUID NOT NULL REFERENCES research_worker_sessions(id),
                    work_item_id UUID REFERENCES lab_work_items(id), outcome TEXT NOT NULL,
                    summary TEXT NOT NULL, status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL
                );
                CREATE INDEX IF NOT EXISTS lab_worker_turns_project_idx
                    ON lab_worker_turns(project_id,created_at DESC);
                CREATE TABLE IF NOT EXISTS lab_worker_wake_requests (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    worker_id UUID NOT NULL REFERENCES research_workers(id),
                    worker_session_id UUID NOT NULL REFERENCES research_worker_sessions(id),
                    work_item_id UUID REFERENCES lab_work_items(id), turn_id UUID REFERENCES lab_worker_turns(id),
                    reason TEXT NOT NULL, status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
                    dispatched_at TIMESTAMPTZ, completed_at TIMESTAMPTZ, failure_reason TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS lab_worker_wakes_one_pending_per_session
                    ON lab_worker_wake_requests(worker_session_id)
                    WHERE status IN ('PENDING','DISPATCHED');
            """)

    def _session(self, cur, project_id: str, worker_id: str, session_id: str) -> dict:
        self.lab._worker(cur, worker_id)
        return self.lab._session(cur, session_id, worker_id, project_id)

    def controls_get(self, project_id: str) -> dict[str, Any]:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM research_projects WHERE id=%s", (project_id,))
            if not cur.fetchone():
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            cur.execute("SELECT autopilot_enabled,auto_continue_enabled,paused,updated_at FROM lab_project_controls WHERE project_id=%s", (project_id,))
            row = cur.fetchone()
            return {"project_id": project_id, **(row or {"autopilot_enabled": False, "auto_continue_enabled": False, "paused": False, "updated_at": None})}

    def controls_set(self, project_id: str, worker_id: str, session_id: str, *,
                     autopilot_enabled: bool | None = None,
                     auto_continue_enabled: bool | None = None,
                     paused: bool | None = None) -> dict[str, Any]:
        if all(value is None for value in (autopilot_enabled, auto_continue_enabled, paused)):
            raise GPUError("LAB_CONTROL_UPDATE_EMPTY", "Specify at least one project control")
        if any(value is not None and not isinstance(value, bool) for value in (autopilot_enabled, auto_continue_enabled, paused)):
            raise GPUError("LAB_CONTROL_INVALID", "Controls must be booleans")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, project_id, worker_id, session_id)
            cur.execute("SELECT autopilot_enabled,auto_continue_enabled,paused FROM lab_project_controls WHERE project_id=%s FOR UPDATE", (project_id,))
            prior = cur.fetchone() or {"autopilot_enabled": False, "auto_continue_enabled": False, "paused": False}
            values = {
                "autopilot_enabled": prior["autopilot_enabled"] if autopilot_enabled is None else autopilot_enabled,
                "auto_continue_enabled": prior["auto_continue_enabled"] if auto_continue_enabled is None else auto_continue_enabled,
                "paused": prior["paused"] if paused is None else paused,
            }
            cur.execute("INSERT INTO lab_project_controls(project_id,autopilot_enabled,auto_continue_enabled,paused,updated_at) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(project_id) DO UPDATE SET autopilot_enabled=EXCLUDED.autopilot_enabled,auto_continue_enabled=EXCLUDED.auto_continue_enabled,paused=EXCLUDED.paused,updated_at=EXCLUDED.updated_at", (project_id, values["autopilot_enabled"], values["auto_continue_enabled"], values["paused"], now))
            self.store._event(cur, project_id, "LAB_PROJECT_CONTROLS_UPDATED", None, values)
        return {"project_id": project_id, **values, "updated_at": now}

    def controls_set_operator(self, project_id: str, *, autopilot_enabled: bool | None = None,
                              auto_continue_enabled: bool | None = None,
                              paused: bool | None = None) -> dict[str, Any]:
        """Apply a human-authenticated cockpit control update without impersonating a worker."""
        if all(value is None for value in (autopilot_enabled, auto_continue_enabled, paused)):
            raise GPUError("LAB_CONTROL_UPDATE_EMPTY", "Specify at least one project control")
        if any(value is not None and not isinstance(value, bool) for value in (autopilot_enabled, auto_continue_enabled, paused)):
            raise GPUError("LAB_CONTROL_INVALID", "Controls must be booleans")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM research_projects WHERE id=%s", (project_id,))
            if not cur.fetchone():
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            cur.execute("SELECT autopilot_enabled,auto_continue_enabled,paused FROM lab_project_controls WHERE project_id=%s FOR UPDATE", (project_id,))
            prior = cur.fetchone() or {"autopilot_enabled": False, "auto_continue_enabled": False, "paused": False}
            values = {
                "autopilot_enabled": prior["autopilot_enabled"] if autopilot_enabled is None else autopilot_enabled,
                "auto_continue_enabled": prior["auto_continue_enabled"] if auto_continue_enabled is None else auto_continue_enabled,
                "paused": prior["paused"] if paused is None else paused,
            }
            cur.execute("INSERT INTO lab_project_controls(project_id,autopilot_enabled,auto_continue_enabled,paused,updated_at) VALUES(%s,%s,%s,%s,%s) ON CONFLICT(project_id) DO UPDATE SET autopilot_enabled=EXCLUDED.autopilot_enabled,auto_continue_enabled=EXCLUDED.auto_continue_enabled,paused=EXCLUDED.paused,updated_at=EXCLUDED.updated_at", (project_id, values["autopilot_enabled"], values["auto_continue_enabled"], values["paused"], now))
            self.store._event(cur, project_id, "LAB_PROJECT_CONTROLS_UPDATED", None, {**values, "actor": "COCKPIT_OPERATOR"})
        return {"project_id": project_id, **values, "updated_at": now}

    @staticmethod
    def _conversation_url(url: str | None) -> str | None:
        if url is None:
            return None
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"chatgpt.com", "www.chatgpt.com"}:
            raise GPUError("BROWSER_CONVERSATION_URL_INVALID", "Only HTTPS chatgpt.com conversation URLs are accepted")
        return url

    def runtime_attach(self, project_id: str, worker_id: str, session_id: str,
                       conversation_url: str | None, browser_profile_key: str | None = None) -> dict[str, Any]:
        now, runtime_id = self._now(), str(uuid.uuid4())
        conversation_url = self._conversation_url(conversation_url)
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, project_id, worker_id, session_id)
            cur.execute("SELECT id FROM lab_worker_runtimes WHERE worker_session_id=%s FOR UPDATE", (session_id,))
            existing = cur.fetchone()
            if existing:
                runtime_id = str(existing["id"])
                cur.execute("UPDATE lab_worker_runtimes SET conversation_url=%s,browser_profile_key=%s,status='UNATTACHED',last_error=NULL,updated_at=%s WHERE id=%s", (conversation_url, browser_profile_key, now, runtime_id))
            else:
                cur.execute("INSERT INTO lab_worker_runtimes(id,worker_id,worker_session_id,project_id,runtime_type,status,browser_profile_key,logical_page_key,conversation_url,created_at,updated_at) VALUES(%s,%s,%s,%s,'CHATGPT_WEB_PLAYWRIGHT','UNATTACHED',%s,%s,%s,%s,%s)", (runtime_id, worker_id, session_id, project_id, browser_profile_key, f"worker-{worker_id}", conversation_url, now, now))
            self.store._event(cur, project_id, "BROWSER_RUNTIME_ATTACHED", None, {"runtime_id": runtime_id, "worker_id": worker_id, "conversation_url": conversation_url})
        return self.runtime_get(runtime_id)

    def runtime_get(self, runtime_id: str) -> dict[str, Any]:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_worker_runtimes WHERE id=%s", (runtime_id,))
            runtime = cur.fetchone()
            if not runtime:
                raise GPUError("BROWSER_RUNTIME_NOT_FOUND", runtime_id)
            return self.lab._record(runtime) or {}

    def runtime_status(self, runtime_id: str, status: str, error: str | None = None) -> dict[str, Any]:
        status = str(status).upper()
        if status not in WEB_STATUSES:
            raise GPUError("BROWSER_RUNTIME_STATUS_INVALID", status)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT project_id FROM lab_worker_runtimes WHERE id=%s FOR UPDATE", (runtime_id,))
            runtime = cur.fetchone()
            if not runtime:
                raise GPUError("BROWSER_RUNTIME_NOT_FOUND", runtime_id)
            cur.execute(
                "UPDATE lab_worker_runtimes SET status=%s,last_error=%s,last_seen_at=%s,"
                "attached_at=CASE WHEN %s THEN COALESCE(attached_at,%s) ELSE attached_at END,"
                "updated_at=%s WHERE id=%s",
                (status, (error or "")[:2000] or None, now, status in ATTACHED_WEB_STATUSES, now, now, runtime_id),
            )
            self.store._event(cur, runtime["project_id"], "BROWSER_RUNTIME_STATUS_CHANGED", runtime_id, {"status": status, "error": bool(error)})
        return self.runtime_get(runtime_id)

    def state_get(self, project_id: str, session_id: str | None = None) -> dict[str, Any]:
        """Build a compact cockpit projection from durable operational state."""
        lab_state = self.lab.state_get(project_id, session_id)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,worker_id,worker_session_id,status,conversation_url,last_seen_at,last_error "
                "FROM lab_worker_runtimes WHERE project_id=%s ORDER BY updated_at DESC",
                (project_id,),
            )
            runtimes = [self.lab._record(row) for row in cur.fetchall()]
            cur.execute(
                "SELECT id,worker_id,worker_session_id,work_item_id,outcome,summary,status,created_at "
                "FROM lab_worker_turns WHERE project_id=%s ORDER BY created_at DESC LIMIT 30",
                (project_id,),
            )
            turns = [self.lab._record(row) for row in cur.fetchall()]
            cur.execute(
                "SELECT id,worker_id,worker_session_id,work_item_id,reason,status,created_at "
                "FROM lab_worker_wake_requests WHERE project_id=%s AND status IN ('PENDING','DISPATCHED') "
                "ORDER BY created_at DESC LIMIT 30",
                (project_id,),
            )
            wakes = [self.lab._record(row) for row in cur.fetchall()]
            cur.execute(
                "SELECT id,status,data,created_at FROM research_objects WHERE project_id=%s "
                "AND kind='DiscoveryRound' ORDER BY created_at DESC LIMIT 12",
                (project_id,),
            )
            discovery_rounds = [
                {
                    "id": str(row["id"]), "status": row["status"],
                    "search_regime": row["data"].get("search_regime"),
                    "phase": row["data"].get("phase"),
                    "peer_visibility": row["data"].get("peer_visibility"),
                    "required_distance_coverage": row["data"].get("required_distance_coverage", {}),
                    "created_at": row["created_at"],
                }
                for row in cur.fetchall()
            ]
            completed_round_ids = [item["id"] for item in discovery_rounds if item["status"] == "COMPLETED"]
            discovery_archives: list[dict[str, Any]] = []
            if completed_round_ids:
                cur.execute(
                    "SELECT id,data,created_at FROM research_objects WHERE project_id=%s "
                    "AND kind='CrossWorkerQDArchive' AND data->>'discovery_round_id'=ANY(%s) "
                    "ORDER BY created_at DESC LIMIT 12",
                    (project_id, completed_round_ids),
                )
                for row in cur.fetchall():
                    archive_data = row["data"]
                    candidates = []
                    for candidate_id in archive_data.get("survivor_candidate_ids", []):
                        cur.execute("SELECT id,data FROM research_objects WHERE id=%s AND kind='DiscoveryCandidate'", (candidate_id,))
                        candidate = cur.fetchone()
                        if candidate:
                            data = candidate["data"]
                            candidates.append({
                                "id": str(candidate["id"]), "title": data.get("title"),
                                "generation_operator": data.get("generation_operator"),
                                "scientific_distance": data.get("scientific_distance"),
                                "mechanistic_niche": data.get("mechanistic_niche"),
                                "architecture_lineage": data.get("architecture_lineage"),
                                "dead_memory_status": data.get("dead_memory_status"),
                                "novelty_status": data.get("novelty_status"),
                                "falsifier": data.get("falsifier"),
                                "selected": str(candidate["id"]) == archive_data.get("selected_candidate_id"),
                                "runner_up": str(candidate["id"]) == archive_data.get("runner_up_candidate_id"),
                            })
                    discovery_archives.append({
                        "id": str(row["id"]), "round_id": archive_data.get("discovery_round_id"),
                        "coverage": archive_data.get("coverage", {}), "candidates": candidates,
                    })
            cur.execute(
                "SELECT id,status,data,created_at FROM research_objects WHERE project_id=%s "
                "AND kind='CorrectionCase' ORDER BY created_at DESC LIMIT 12",
                (project_id,),
            )
            correction_cases = [
                {
                    "id": str(row["id"]), "status": row["status"],
                    "target_id": row["data"].get("target_id"),
                    "target_type": row["data"].get("target_type"),
                    "phase": row["data"].get("correction_stage"),
                    "peer_visibility": row["data"].get("peer_visibility"),
                    "resolution": row["data"].get("resolution"),
                    "created_at": row["created_at"],
                }
                for row in cur.fetchall()
            ]
            cur.execute(
                "SELECT id,status,data,created_at FROM research_objects WHERE project_id=%s "
                "AND kind='StrategyTransferCandidate' ORDER BY created_at DESC LIMIT 12",
                (project_id,),
            )
            strategy_transfers = [
                {
                    "id": str(row["id"]), "status": row["status"],
                    "strategy_id": row["data"].get("strategy_id"),
                    "source_project_id": row["data"].get("source_project_id"),
                    "predicted_benefit": row["data"].get("predicted_benefit"),
                    "applicability": (row["data"].get("applicability_assessment") or {}).get("state"),
                    "created_at": row["created_at"],
                }
                for row in cur.fetchall()
            ]
        # v3.6 is operational data: expose coverage and worker availability
        # separately so a waiting WorkItem is never rendered as a blocked worker.
        portfolio = {
            "branch_coverage": self.lab.branch_coverage_get(project_id),
            "agenda_coverage": self.lab.agenda_coverage_get(project_id),
            "production_audit": self.lab.portfolio_production_audit(project_id),
        }
        return {
            "lab_state": lab_state,
            "live_workers_by_project": self.live_workers_by_project(),
            "controls": self.controls_get(project_id),
            "browser_runtimes": runtimes,
            "recent_turns": turns,
            "pending_wake_requests": wakes,
            # Deliberately excludes candidate content while a round is isolated.
            "discovery_rounds": discovery_rounds,
            "discovery_archives": discovery_archives,
            "correction_cases": correction_cases,
            "strategy_transfers": strategy_transfers,
            "portfolio_scheduler": portfolio,
        }

    def live_workers_by_project(self) -> list[dict[str, Any]]:
        """List heartbeat-live workers grouped by current project for safe operations."""
        threshold = self._now() - timedelta(seconds=self.lab.lease_seconds)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT p.id AS project_id,p.name AS project_name,s.id AS session_id,"
                "s.worker_id,w.display_name,s.status,s.active_role,s.current_work_item_id,"
                "s.last_heartbeat_at FROM research_worker_sessions s "
                "JOIN research_workers w ON w.id=s.worker_id "
                "JOIN research_projects p ON p.id=s.current_project_id "
                "WHERE s.status NOT IN ('DISCONNECTED','EXPIRED') "
                "AND s.last_heartbeat_at>=%s ORDER BY p.name,s.last_heartbeat_at DESC",
                (threshold,),
            )
            grouped: dict[str, dict[str, Any]] = {}
            for row in cur.fetchall():
                project_id = str(row["project_id"])
                project = grouped.setdefault(
                    project_id,
                    {
                        "project_id": project_id,
                        "project_name": row["project_name"],
                        "live_worker_count": 0,
                        "workers": [],
                    },
                )
                project["workers"].append(
                    {
                        "worker_id": str(row["worker_id"]),
                        "display_name": row["display_name"],
                        "session_id": str(row["session_id"]),
                        "status": row["status"],
                        "active_role": row["active_role"],
                        "current_work_item_id": (
                            str(row["current_work_item_id"])
                            if row["current_work_item_id"]
                            else None
                        ),
                        "last_heartbeat_at": row["last_heartbeat_at"],
                    }
                )
                project["live_worker_count"] += 1
        return list(grouped.values())

    def turn_report(self, project_id: str, worker_id: str, session_id: str, outcome: str,
                    summary: str, work_item_id: str | None = None) -> dict[str, Any]:
        outcome = str(outcome).upper()
        if outcome not in TURN_OUTCOMES:
            raise GPUError("LAB_TURN_OUTCOME_INVALID", outcome)
        now, turn_id = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            session = self._session(cur, project_id, worker_id, session_id)
            current = str(session["current_work_item_id"]) if session["current_work_item_id"] else None
            if work_item_id and current != str(work_item_id):
                raise GPUError("LAB_TURN_WORK_NOT_OWNED", work_item_id)
            work_item_id = work_item_id or current
            if outcome == "CONTINUE":
                if work_item_id:
                    cur.execute("SELECT count(*) AS count FROM lab_worker_turns WHERE worker_session_id=%s AND work_item_id=%s", (session_id, work_item_id))
                    if cur.fetchone()["count"] >= self.max_turns_per_work_item:
                        outcome = "HUMAN_REQUIRED"
                        summary = f"Continuation limit reached for work item: {summary}"
                cur.execute("SELECT outcome FROM lab_worker_turns WHERE worker_session_id=%s ORDER BY created_at DESC LIMIT %s", (session_id, self.max_consecutive_continues))
                previous = cur.fetchall()
                if len(previous) == self.max_consecutive_continues and all(item["outcome"] == "CONTINUE" for item in previous):
                    outcome = "HUMAN_REQUIRED"
                    summary = f"Consecutive continuation limit reached: {summary}"
            cur.execute("INSERT INTO lab_worker_turns(id,project_id,worker_id,worker_session_id,work_item_id,outcome,summary,status,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,'REPORTED',%s)", (turn_id, project_id, worker_id, session_id, work_item_id, outcome, summary[:4000], now))
            cur.execute("SELECT autopilot_enabled,auto_continue_enabled,paused FROM lab_project_controls WHERE project_id=%s", (project_id,))
            controls = cur.fetchone() or {"autopilot_enabled": False, "auto_continue_enabled": False, "paused": False}
            wake_id = None
            if outcome == "CONTINUE" and controls["autopilot_enabled"] and controls["auto_continue_enabled"] and not controls["paused"]:
                wake_id = str(uuid.uuid4())
                cur.execute("INSERT INTO lab_worker_wake_requests(id,project_id,worker_id,worker_session_id,work_item_id,turn_id,reason,status,created_at) VALUES(%s,%s,%s,%s,%s,%s,'CONTINUE','PENDING',%s) ON CONFLICT(worker_session_id) WHERE status IN ('PENDING','DISPATCHED') DO NOTHING RETURNING id", (wake_id, project_id, worker_id, session_id, work_item_id, turn_id, now))
                row = cur.fetchone()
                wake_id = str(row["id"]) if row else None
            status = "WAITING" if outcome == "WAITING_DEPENDENCY" else "IDLE" if outcome in {"IDLE", "HUMAN_REQUIRED", "ERROR"} else "BUSY"
            cur.execute("UPDATE research_worker_sessions SET status=%s,last_heartbeat_at=%s WHERE id=%s", (status, now, session_id))
            self.store._event(cur, project_id, "LAB_TURN_REPORTED", turn_id, {"worker_id": worker_id, "work_item_id": work_item_id, "outcome": outcome, "wake_request_id": wake_id})
        return {"turn_id": turn_id, "outcome": outcome, "wake_request_id": wake_id}

    def wake_ready_work(self, project_id: str, work_item_ids: list[str] | None = None) -> dict[str, int]:
        """Queue at most one bounded ready-work wake per eligible attached browser worker."""
        now, queued = self._now(), 0
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT autopilot_enabled,auto_continue_enabled,paused FROM lab_project_controls WHERE project_id=%s",
                (project_id,),
            )
            controls = cur.fetchone() or {"autopilot_enabled": False, "auto_continue_enabled": False, "paused": False}
            if not controls["autopilot_enabled"] or not controls["auto_continue_enabled"] or controls["paused"]:
                return {"queued": 0}
            sql = (
                "SELECT w.id FROM lab_work_items w WHERE w.project_id=%s AND w.status='READY' "
                "ORDER BY w.priority DESC,w.created_at"
            )
            args: list[Any] = [project_id]
            if work_item_ids:
                sql = sql.replace(" ORDER BY", " AND w.id=ANY(%s) ORDER BY")
                args.append(work_item_ids)
            cur.execute(sql + " LIMIT 1", args)
            work = cur.fetchone()
            if not work:
                return {"queued": 0}
            cur.execute(
                "SELECT s.id AS session_id,s.worker_id FROM research_worker_sessions s "
                "JOIN lab_worker_runtimes r ON r.worker_session_id=s.id "
                "WHERE s.current_project_id=%s AND s.status IN ('ACTIVE','IDLE') "
                "AND s.current_work_item_id IS NULL AND r.status NOT IN ('ERROR','LOGIN_REQUIRED','DISCONNECTED') "
                "ORDER BY s.last_heartbeat_at DESC FOR UPDATE SKIP LOCKED LIMIT 1",
                (project_id,),
            )
            worker = cur.fetchone()
            if not worker:
                return {"queued": 0}
            wake_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO lab_worker_wake_requests(id,project_id,worker_id,worker_session_id,work_item_id,reason,status,created_at) "
                "VALUES(%s,%s,%s,%s,%s,'READY_WORK','PENDING',%s) "
                "ON CONFLICT(worker_session_id) WHERE status IN ('PENDING','DISPATCHED') DO NOTHING RETURNING id",
                (wake_id, project_id, worker["worker_id"], worker["session_id"], work["id"], now),
            )
            if cur.fetchone():
                self.store._event(cur, project_id, "WORKER_WAKE_QUEUED", wake_id, {"work_item_id": str(work["id"]), "reason": "READY_WORK"})
                queued = 1
        return {"queued": queued}

    def wake_claim_next(self, project_id: str | None = None) -> dict[str, Any] | None:
        """Atomically hand one pending operational wake to a runtime dispatcher."""
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            sql = "SELECT * FROM lab_worker_wake_requests WHERE status='PENDING'"
            args: list[Any] = []
            if project_id:
                sql += " AND project_id=%s"
                args.append(project_id)
            sql += " ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1"
            cur.execute(sql, args)
            wake = cur.fetchone()
            if not wake:
                return None
            cur.execute("SELECT autopilot_enabled,auto_continue_enabled,paused FROM lab_project_controls WHERE project_id=%s", (wake["project_id"],))
            controls = cur.fetchone() or {"autopilot_enabled": False, "auto_continue_enabled": False, "paused": False}
            if not controls["autopilot_enabled"] or not controls["auto_continue_enabled"] or controls["paused"]:
                return None
            cur.execute("UPDATE lab_worker_wake_requests SET status='DISPATCHED',dispatched_at=%s WHERE id=%s", (now, wake["id"]))
            self.store._event(cur, wake["project_id"], "WORKER_WAKE_DISPATCHED", wake["id"], {"worker_id": str(wake["worker_id"])})
            return self.lab._record(wake)

    def wake_finish(self, wake_id: str, *, failure_reason: str | None = None) -> dict[str, Any]:
        now, status = self._now(), "FAILED" if failure_reason else "COMPLETED"
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT project_id FROM lab_worker_wake_requests WHERE id=%s FOR UPDATE", (wake_id,))
            wake = cur.fetchone()
            if not wake:
                raise GPUError("WORKER_WAKE_NOT_FOUND", wake_id)
            cur.execute("UPDATE lab_worker_wake_requests SET status=%s,completed_at=%s,failure_reason=%s WHERE id=%s", (status, now, (failure_reason or "")[:2000] or None, wake_id))
            self.store._event(cur, wake["project_id"], "WORKER_WAKE_FAILED" if failure_reason else "WORKER_WAKE_COMPLETED", wake_id, {"failure_reason": bool(failure_reason)})
        return {"wake_id": wake_id, "status": status}

    def recover_inflight_wakes(self) -> dict[str, int]:
        """Fail closed after a controller restart so an unconfirmed prompt is never duplicated."""
        now, recovered = self._now(), 0
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,project_id FROM lab_worker_wake_requests WHERE status='DISPATCHED' FOR UPDATE SKIP LOCKED")
            for wake in cur.fetchall():
                cur.execute("UPDATE lab_worker_wake_requests SET status='FAILED',completed_at=%s,failure_reason='CONTROLLER_RESTART_UNCONFIRMED_DISPATCH' WHERE id=%s", (now, wake["id"]))
                self.store._event(cur, wake["project_id"], "WORKER_WAKE_FAILED", wake["id"], {"failure_reason": "CONTROLLER_RESTART_UNCONFIRMED_DISPATCH"})
                recovered += 1
        return {"recovered": recovered}
