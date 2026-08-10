import json
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .errors import GPUError


class ResearchStore:
    """PostgreSQL canonical scientific state with immutable event history."""

    def __init__(self, url: str):
        self.url = url
        self.vector_available = False
        self._migrate()

    def _connect(self):
        return psycopg.connect(self.url, row_factory=dict_row)

    def _migrate(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.vector_available = True
            except psycopg.errors.UndefinedFile:
                conn.rollback()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_projects (
                    id UUID PRIMARY KEY, name TEXT UNIQUE NOT NULL, question TEXT NOT NULL,
                    state JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_objects (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    kind TEXT NOT NULL CHECK(kind IN ('Paper','EvidenceUnit','Claim','Mechanism','Anomaly','Contradiction','Hypothesis','Prediction','HypothesisLineage','Experiment','ExperimentRun','Artifact','Reproduction','NegativeResult','Lesson','ResearchState')),
                    status TEXT NOT NULL DEFAULT 'ACTIVE', data JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_edges (
                    source_id UUID NOT NULL REFERENCES research_objects(id), target_id UUID NOT NULL REFERENCES research_objects(id), relation TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, PRIMARY KEY(source_id,target_id,relation)
                );
                CREATE TABLE IF NOT EXISTS research_events (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    event_type TEXT NOT NULL, subject_id UUID, payload JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
                );
            """)

    def project_create(self, name: str, question: str) -> dict:
        ident, now = uuid.uuid4(), datetime.now(UTC)
        state = {"research_question": question, "established_facts": [], "active_hypotheses": [], "next_discriminating_experiments": []}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO research_projects VALUES(%s,%s,%s,%s,%s)", (ident, name, question, json.dumps(state), now))
            self._event(cur, ident, "RESEARCH_PROJECT_CREATED", None, {"name": name, "question": question})
        return {"project_id": str(ident), "name": name, "state": state}

    def object_create(self, project_id: str, kind: str, data: dict[str, Any], event_type: str) -> dict:
        ident, now = uuid.uuid4(), datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO research_objects VALUES(%s,%s,%s,%s,%s,%s)", (ident, project_id, kind, "ACTIVE", json.dumps(data), now))
            self._event(cur, project_id, event_type, ident, data)
        return {"id": str(ident), "project_id": project_id, "kind": kind, "status": "ACTIVE", "data": data}

    def state_get(self, project_id: str) -> dict:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT name,question,state FROM research_projects WHERE id=%s", (project_id,))
            row = cur.fetchone()
            if not row:
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            cur.execute("SELECT id,kind,status,data,created_at FROM research_objects WHERE project_id=%s ORDER BY created_at DESC", (project_id,))
            objects = cur.fetchall()
            by_kind = lambda kind, statuses=None: [item for item in objects if item["kind"] == kind and (not statuses or item["status"] in statuses)]
            canonical = {
                "research_question": row["question"],
                "supported_claims": by_kind("Claim", {"SUPPORTED"}),
                "weakened_claims": by_kind("Claim", {"WEAKENED"}),
                "refuted_claims": by_kind("Claim", {"REFUTED"}),
                "unresolved_claims": by_kind("Claim", {"ACTIVE"}),
                "active_anomalies": by_kind("Anomaly", {"ACTIVE"}),
                "active_hypotheses": by_kind("Hypothesis", {"ACTIVE", "SURVIVES_INITIAL_TEST"}),
                "refuted_lineages": by_kind("Hypothesis", {"REFUTED"}),
                "completed_experiments": by_kind("ExperimentRun", {"completed"}),
                "active_experiments": by_kind("ExperimentRun", {"ACTIVE", "running"}),
                "negative_results": by_kind("NegativeResult"),
            }
            return {**row, "canonical_state": canonical, "objects": objects}

    def object_get(self, object_id: str) -> dict:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,project_id,kind,status,data,created_at FROM research_objects WHERE id=%s", (object_id,))
            row = cur.fetchone()
            if not row:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", object_id)
            return row

    def edge_create(self, source_id: str, target_id: str, relation: str) -> None:
        """Record a graph relationship only when both objects share a project."""
        source, target = self.object_get(source_id), self.object_get(target_id)
        if source["project_id"] != target["project_id"]:
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "Research objects must belong to one project")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO research_edges VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (source_id, target_id, relation, datetime.now(UTC)),
            )

    def run_create(self, experiment_id: str, execution: dict) -> dict:
        experiment = self.object_get(experiment_id)
        if experiment["kind"] != "Experiment" or not experiment["data"].get("frozen"):
            raise GPUError("EXPERIMENT_NOT_PREREGISTERED", experiment_id)
        run = self.object_create(experiment["project_id"], "ExperimentRun", {"experiment_id": experiment_id, **execution}, "EXPERIMENT_STARTED")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO research_edges VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING", (experiment_id, run["id"], "HAS_RUN", datetime.now(UTC)))
        return run

    def run_update(self, run_id: str, result: dict) -> dict:
        run = self.object_get(run_id)
        if run["status"] in {"completed", "failed", "cancelled"}:
            return {"id": run_id, "status": run["status"], "data": run["data"], "already_final": True}
        data = {**run["data"], **result}
        status = result.get("status", run["status"])
        event = "EXPERIMENT_COMPLETED" if status == "completed" else "EXPERIMENT_FAILED"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, json.dumps(data), run_id))
            self._event(cur, run["project_id"], event, run_id, result)
        return {"id": run_id, "status": status, "data": data}

    def assess(self, object_id: str, status: str, rationale: str) -> dict:
        item = self.object_get(object_id)
        if status not in {"SUPPORTED", "WEAKENED", "REFUTED", "SURVIVES_INITIAL_TEST", "INCONCLUSIVE", "BLOCKED"}:
            raise GPUError("INVALID_SCIENTIFIC_STATUS", status)
        event = f"{item['kind'].upper()}_{status}"
        data = {**item["data"], "assessment_rationale": rationale}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, json.dumps(data), object_id))
            self._event(cur, item["project_id"], event, object_id, {"rationale": rationale})
        return {"id": object_id, "status": status, "rationale": rationale}

    def object_update(self, object_id: str, data_update: dict[str, Any], status: str, event_type: str) -> dict:
        """Materialize a new current view while preserving the scientific change as an event."""
        item = self.object_get(object_id)
        data = {**item["data"], **data_update}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, json.dumps(data), object_id))
            self._event(cur, item["project_id"], event_type, object_id, {"status": status, **data_update})
        return {"id": object_id, "status": status, "data": data}

    def events(self, project_id: str, limit: int = 100) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT event_type,subject_id,payload,created_at FROM research_events WHERE project_id=%s ORDER BY created_at DESC LIMIT %s", (project_id, limit))
            return cur.fetchall()

    def search(self, project_id: str, query: str, kind: str | None = None, limit: int = 25) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            sql = "SELECT id,kind,status,data,created_at FROM research_objects WHERE project_id=%s"
            args: list[Any] = [project_id]
            if kind:
                sql += " AND kind=%s"
                args.append(kind)
            sql += " AND data::text ILIKE %s ORDER BY created_at DESC LIMIT %s"
            args.extend((f"%{query}%", limit))
            cur.execute(sql, args)
            return cur.fetchall()

    @staticmethod
    def _event(cur, project_id, event_type, subject_id, payload):
        cur.execute("INSERT INTO research_events VALUES(%s,%s,%s,%s,%s,%s)", (uuid.uuid4(), project_id, event_type, subject_id, json.dumps(payload), datetime.now(UTC)))
