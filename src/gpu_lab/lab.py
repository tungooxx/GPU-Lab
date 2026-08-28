"""Durable, project-scoped coordination for multiple Research OS workers.

Lab control is deliberately operational: scientific truth continues to live in
ResearchStore objects, evidence pathways, and the v3.1 Brain.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg

from .errors import GPUError
from .research import ResearchStore

RUNTIME_TYPES = {"CHATGPT_WEB", "OPENAI_API", "CLAUDE_API", "CODEX", "LOCAL_AGENT", "OTHER"}
SESSION_STATUSES = {"ACTIVE", "BUSY", "WAITING", "IDLE", "DISCONNECTED", "EXPIRED"}
WORK_STATUSES = {
    "READY", "CLAIMED", "RUNNING", "RUNNING_DETACHED", "RESULT_READY", "WAITING_DEPENDENCY", "BLOCKED", "COMPLETED",
    "FAILED", "CANCELLED", "INVALIDATED", "SUPERSEDED", "DORMANT", "REPLAN_REQUIRED",
}
ACTIVE_WORK_STATUSES = {
    "READY", "CLAIMED", "RUNNING", "RUNNING_DETACHED", "RESULT_READY", "WAITING_DEPENDENCY", "BLOCKED",
    "REPLAN_REQUIRED",
}
EQUIVALENCE_ACTIVE_WORK_STATUSES = ACTIVE_WORK_STATUSES | {"DORMANT"}
AUTHORITY_STATUSES = {"AUTHORITATIVE", "SUPPORTING", "RECOVERY_TEMPLATE", "OBSOLETE", "SUPERSEDED"}
OBJECTIVE_STATUSES = {"ACTIVE", "BLOCKED", "RESOLVED", "SUPERSEDED", "PAUSED", "CANCELLED"}
PROPOSAL_STATUSES = {"PROPOSED", "MERGED_INTO_EXISTING", "SATISFIED_BY_TERMINAL", "MATERIALIZED", "REJECTED_DUPLICATE", "REJECTED_LOW_VALUE", "REJECTED_STALE", "SUPERSEDED"}
DEPENDENCY_SCOPES = {"WORKITEM_LOCAL", "BRANCH", "PORTFOLIO", "OBJECTIVE_GLOBAL"}
BRANCH_STATES = {"OPEN", "WAITING", "BLOCKED", "SUPPORTED_WITHIN_SCOPE", "WEAKENED", "REFUTED", "RESOLVED", "SUPERSEDED", "ABANDONED"}
SPECULATION_CLASSES = {"NON_SPECULATIVE", "SAFE_SPECULATIVE", "CONDITIONAL_SPECULATIVE", "EXPENSIVE_SPECULATIVE"}
RESOURCE_CLASSES = {"REASONING", "ENGINEERING", "GPU", "TRAINING", "LITERATURE", "REVIEW", "SYNTHESIS"}
GATE_STATUSES = {"PENDING", "PREFLIGHT_FAILED", "AWAITING_SEMANTIC_REVIEW", "PASS", "FAIL", "INVALID", "SUPERSEDED"}
PREFLIGHT_STATUSES = {"PASS", "FAIL"}
COORDINATION_VERSION = "lab-coordination-v3.2.2-gates-v1"
MESSAGE_TYPES = {
    "REQUEST_REVIEW", "REQUEST_DATA", "REQUEST_IMPLEMENTATION", "SHARE_FINDING",
    "CHALLENGE_INTERPRETATION", "HANDOFF", "BLOCKER", "CROSS_PROJECT_RELEVANCE",
    "COORDINATION", "INFORMATION", "ENGINEERING_HANDOFF",
}


class LabController:
    """Coordinate workers without becoming a second scientific truth store."""

    def __init__(self, store: ResearchStore, lease_seconds: int = 300,
                 feature_flags: dict[str, bool] | None = None):
        self.store = store
        self.lease_seconds = lease_seconds
        self.feature_flags = feature_flags or {}
        self._migrate()

    def _migrate(self) -> None:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_lab_worker_migration'))")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_workers (
                    id UUID PRIMARY KEY, display_name TEXT UNIQUE NOT NULL,
                    worker_type TEXT NOT NULL, capabilities JSONB NOT NULL DEFAULT '{}'::jsonb,
                    enabled BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_worker_sessions (
                    id UUID PRIMARY KEY, worker_id UUID NOT NULL REFERENCES research_workers(id),
                    runtime_type TEXT NOT NULL, runtime_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    current_project_id UUID REFERENCES research_projects(id), current_work_item_id UUID,
                    active_role TEXT, status TEXT NOT NULL, joined_at TIMESTAMPTZ NOT NULL,
                    last_heartbeat_at TIMESTAMPTZ NOT NULL, disconnected_at TIMESTAMPTZ,
                    active_policy_version TEXT, active_brain_version TEXT, context_version JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                CREATE INDEX IF NOT EXISTS research_worker_sessions_project_active_idx
                    ON research_worker_sessions(current_project_id,last_heartbeat_at DESC);
                CREATE TABLE IF NOT EXISTS lab_work_items (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    kind TEXT NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL,
                    scientific_role TEXT NOT NULL, status TEXT NOT NULL, priority DOUBLE PRECISION NOT NULL DEFAULT 0,
                    expected_value DOUBLE PRECISION, estimated_cost DOUBLE PRECISION,
                    created_by UUID REFERENCES research_workers(id), assigned_worker_id UUID REFERENCES research_workers(id),
                    assigned_session_id UUID REFERENCES research_worker_sessions(id), lease_id UUID,
                    parent_work_item_id UUID REFERENCES lab_work_items(id), branch_id UUID,
                    related_refs JSONB NOT NULL DEFAULT '{}'::jsonb, equivalence_key TEXT,
                    blocked_reason TEXT, invalidated_reason TEXT,
                    created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL, completed_at TIMESTAMPTZ
                );
                DROP INDEX IF EXISTS lab_work_items_active_equivalence_unique;
                CREATE INDEX IF NOT EXISTS lab_work_items_project_status_priority_idx
                    ON lab_work_items(project_id,status,priority DESC,created_at);
                CREATE TABLE IF NOT EXISTS lab_work_dependencies (
                    id UUID PRIMARY KEY, work_item_id UUID NOT NULL REFERENCES lab_work_items(id) ON DELETE CASCADE,
                    target_type TEXT NOT NULL, target_id TEXT NOT NULL, required_statuses JSONB NOT NULL DEFAULT '[]'::jsonb,
                    invalidating_statuses JSONB NOT NULL DEFAULT '[]'::jsonb, description TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL, UNIQUE(work_item_id,target_type,target_id)
                );
                CREATE TABLE IF NOT EXISTS lab_work_leases (
                    id UUID PRIMARY KEY, work_item_id UUID NOT NULL REFERENCES lab_work_items(id),
                    worker_id UUID NOT NULL REFERENCES research_workers(id),
                    worker_session_id UUID NOT NULL REFERENCES research_worker_sessions(id),
                    acquired_at TIMESTAMPTZ NOT NULL, heartbeat_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL, released_at TIMESTAMPTZ, release_reason TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS lab_work_leases_one_active_per_work
                    ON lab_work_leases(work_item_id) WHERE released_at IS NULL;
                CREATE TABLE IF NOT EXISTS lab_messages (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    from_worker_id UUID NOT NULL REFERENCES research_workers(id),
                    to_worker_id UUID REFERENCES research_workers(id), to_role TEXT, broadcast_scope TEXT,
                    message_type TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL,
                    reference_ids JSONB NOT NULL DEFAULT '[]'::jsonb, priority INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL, read_at TIMESTAMPTZ, resolved_at TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS lab_messages_recipient_idx
                    ON lab_messages(project_id,to_worker_id,created_at DESC);
                CREATE TABLE IF NOT EXISTS lab_project_budgets (
                    project_id UUID PRIMARY KEY REFERENCES research_projects(id),
                    limits JSONB NOT NULL DEFAULT '{}'::jsonb, updated_at TIMESTAMPTZ NOT NULL
                );
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS authority_key TEXT;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS gate_id UUID;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS canonical_subject_version TEXT;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS authority_status TEXT NOT NULL DEFAULT 'SUPPORTING';
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS subject_id TEXT;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS superseded_by UUID;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS invalidated_at TIMESTAMPTZ;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS work_version INTEGER NOT NULL DEFAULT 1;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS recovery_policy JSONB NOT NULL DEFAULT '{}'::jsonb;
                CREATE TABLE IF NOT EXISTS scientific_gates (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    gate_key TEXT NOT NULL, scientific_object_id TEXT NOT NULL,
                    canonical_subject_version TEXT NOT NULL, authority_key TEXT NOT NULL,
                    authoritative_work_item_id UUID REFERENCES lab_work_items(id),
                    deterministic_preflight_id UUID, semantic_review_work_item_id UUID REFERENCES lab_work_items(id),
                    semantic_review_required BOOLEAN NOT NULL DEFAULT TRUE,
                    status TEXT NOT NULL, coordination_version TEXT NOT NULL,
                    superseded_by UUID REFERENCES scientific_gates(id), invalidation_reason TEXT,
                    created_at TIMESTAMPTZ NOT NULL, resolved_at TIMESTAMPTZ,
                    UNIQUE(project_id, gate_key, scientific_object_id, canonical_subject_version)
                );
                ALTER TABLE scientific_gates ADD COLUMN IF NOT EXISTS semantic_review_required BOOLEAN NOT NULL DEFAULT TRUE;
                CREATE UNIQUE INDEX IF NOT EXISTS scientific_gates_active_authority_key_unique
                    ON scientific_gates(project_id, authority_key)
                    WHERE status NOT IN ('SUPERSEDED', 'INVALID');
                CREATE TABLE IF NOT EXISTS lab_deterministic_preflights (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    gate_id UUID NOT NULL REFERENCES scientific_gates(id),
                    scientific_object_id TEXT NOT NULL, canonical_subject_version TEXT NOT NULL,
                    subject_hash TEXT NOT NULL, checks JSONB NOT NULL, failures JSONB NOT NULL DEFAULT '[]'::jsonb,
                    warnings JSONB NOT NULL DEFAULT '[]'::jsonb, result_hash TEXT NOT NULL,
                    status TEXT NOT NULL, validator_version TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(project_id, scientific_object_id, canonical_subject_version, subject_hash, validator_version)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS lab_work_items_active_authority_unique
                    ON lab_work_items(project_id, authority_key)
                    WHERE authority_key IS NOT NULL AND authority_status='AUTHORITATIVE'
                    AND status IN ('READY','CLAIMED','RUNNING','RUNNING_DETACHED','RESULT_READY','WAITING_DEPENDENCY','BLOCKED','REPLAN_REQUIRED');
                CREATE INDEX IF NOT EXISTS lab_work_items_gate_idx ON lab_work_items(gate_id,status,created_at);
                CREATE INDEX IF NOT EXISTS scientific_gates_project_status_idx ON scientific_gates(project_id,status,created_at);
                CREATE TABLE IF NOT EXISTS canonical_objectives (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    version INTEGER NOT NULL, objective_type TEXT NOT NULL, title TEXT NOT NULL,
                    scientific_question TEXT NOT NULL, status TEXT NOT NULL, priority DOUBLE PRECISION NOT NULL DEFAULT 0,
                    current_goal TEXT NOT NULL DEFAULT '', parent_objective_id UUID REFERENCES canonical_objectives(id),
                    supersedes_objective_id UUID REFERENCES canonical_objectives(id),
                    superseded_by_objective_id UUID REFERENCES canonical_objectives(id),
                    created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(project_id, title, version)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS canonical_objectives_one_active_title
                    ON canonical_objectives(project_id,title) WHERE status='ACTIVE';
                CREATE TABLE IF NOT EXISTS hypothesis_branches (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    canonical_objective_id UUID NOT NULL REFERENCES canonical_objectives(id),
                    question_id TEXT, hypothesis_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    mechanistic_niche_id TEXT, scientific_distance TEXT, architecture_lineage_id TEXT,
                    state TEXT NOT NULL, priority DOUBLE PRECISION NOT NULL DEFAULT 0,
                    branch_dependencies JSONB NOT NULL DEFAULT '[]'::jsonb, branch_blocking_scope TEXT NOT NULL DEFAULT 'BRANCH',
                    created_at TIMESTAMPTZ NOT NULL, resolved_at TIMESTAMPTZ
                );
                CREATE TABLE IF NOT EXISTS lab_work_proposals (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    proposed_by_worker_id UUID REFERENCES research_workers(id), canonical_objective_id UUID REFERENCES canonical_objectives(id),
                    hypothesis_branch_id UUID REFERENCES hypothesis_branches(id),
                    target_id TEXT, proposed_mode TEXT NOT NULL, proposed_role TEXT NOT NULL,
                    proposed_authority_key_hint TEXT, proposed_equivalence_key_hint TEXT,
                    rationale TEXT NOT NULL, expected_scientific_value DOUBLE PRECISION,
                    dependency_refs JSONB NOT NULL DEFAULT '[]'::jsonb, status TEXT NOT NULL,
                    canonical_work_item_id UUID REFERENCES lab_work_items(id), created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lab_transactional_outbox (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id), event_type TEXT NOT NULL,
                    subject_id TEXT, payload JSONB NOT NULL DEFAULT '{}'::jsonb, idempotency_key TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL, delivered_at TIMESTAMPTZ, UNIQUE(project_id,idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS lab_consistency_conflicts (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    conflict_key TEXT NOT NULL, conflict_type TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN',
                    object_ids JSONB NOT NULL DEFAULT '[]'::jsonb, details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    detected_at TIMESTAMPTZ NOT NULL, resolved_at TIMESTAMPTZ,
                    UNIQUE(project_id,conflict_key)
                );
                CREATE TABLE IF NOT EXISTS hypothesis_portfolios (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    canonical_objective_id UUID NOT NULL REFERENCES canonical_objectives(id), objective_version INTEGER NOT NULL,
                    research_agenda_item_id TEXT, active_branch_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    branch_budget INTEGER, scientific_concurrency_budget INTEGER, gpu_concurrency_budget INTEGER,
                    training_concurrency_budget INTEGER, reasoning_concurrency_budget INTEGER,
                    search_regime TEXT, status TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(project_id,canonical_objective_id,objective_version)
                );
                CREATE TABLE IF NOT EXISTS parallel_research_plans (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id), portfolio_id UUID REFERENCES hypothesis_portfolios(id),
                    version INTEGER NOT NULL, status TEXT NOT NULL, assignments JSONB NOT NULL DEFAULT '[]'::jsonb,
                    rationale JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(project_id,portfolio_id,version)
                );
                ALTER TABLE research_workers ADD COLUMN IF NOT EXISTS availability_state TEXT NOT NULL DEFAULT 'AVAILABLE';
                ALTER TABLE research_workers ADD COLUMN IF NOT EXISTS availability_updated_at TIMESTAMPTZ;
                ALTER TABLE research_workers ADD COLUMN IF NOT EXISTS idle_reason TEXT;
                ALTER TABLE research_workers ADD COLUMN IF NOT EXISTS idle_since TIMESTAMPTZ;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS canonical_objective_id UUID REFERENCES canonical_objectives(id);
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS speculation_class TEXT NOT NULL DEFAULT 'NON_SPECULATIVE';
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS speculation_condition JSONB NOT NULL DEFAULT '{}'::jsonb;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS resource_class TEXT NOT NULL DEFAULT 'REASONING';
                DO $$ BEGIN
                    ALTER TABLE lab_work_items ADD CONSTRAINT lab_work_items_branch_fk
                        FOREIGN KEY(branch_id) REFERENCES hypothesis_branches(id) NOT VALID;
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$;
                ALTER TABLE lab_work_proposals ADD COLUMN IF NOT EXISTS hypothesis_branch_id UUID REFERENCES hypothesis_branches(id);
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS dependency_scope TEXT NOT NULL DEFAULT 'WORKITEM_LOCAL';
                ALTER TABLE lab_work_dependencies ADD COLUMN IF NOT EXISTS dependency_scope TEXT NOT NULL DEFAULT 'WORKITEM_LOCAL';
                ALTER TABLE lab_work_leases ADD COLUMN IF NOT EXISTS lease_version INTEGER NOT NULL DEFAULT 1;
                WITH ranked_equivalence AS (
                    SELECT id, first_value(id) OVER (PARTITION BY project_id,equivalence_key ORDER BY created_at,id) AS canonical_id,
                           row_number() OVER (PARTITION BY project_id,equivalence_key ORDER BY created_at,id) AS ordinal
                    FROM lab_work_items WHERE equivalence_key IS NOT NULL
                    AND status IN ('DORMANT','READY','CLAIMED','RUNNING','RUNNING_DETACHED','RESULT_READY','WAITING_DEPENDENCY','BLOCKED')
                )
                UPDATE lab_work_items AS duplicate SET status='SUPERSEDED',authority_status='SUPERSEDED',superseded_by=ranked_equivalence.canonical_id,
                    invalidated_reason='v3.5.5 migration: duplicate active equivalence key',invalidated_at=NOW(),updated_at=NOW(),work_version=work_version+1
                FROM ranked_equivalence WHERE duplicate.id=ranked_equivalence.id AND ranked_equivalence.ordinal>1;
                CREATE UNIQUE INDEX IF NOT EXISTS lab_work_items_active_equivalence_unique ON lab_work_items(project_id,equivalence_key)
                    WHERE equivalence_key IS NOT NULL AND status IN ('DORMANT','READY','CLAIMED','RUNNING','RUNNING_DETACHED','RESULT_READY','WAITING_DEPENDENCY','BLOCKED');
            """)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _validate(value: str, allowed: set[str], label: str) -> str:
        normalized = str(value).upper()
        if normalized not in allowed:
            raise GPUError(f"INVALID_{label}", normalized)
        return normalized

    @staticmethod
    def _record(row: dict | None) -> dict | None:
        if not row:
            return None
        return {key: (str(value) if isinstance(value, uuid.UUID) else value) for key, value in row.items()}

    def _event(self, cur, project_id: str, event_type: str, subject_id: str | None, payload: dict) -> None:
        self.store._event(cur, project_id, event_type, subject_id, payload)

    def _outbox(self, cur, project_id: str, event_type: str, subject_id: str | None,
                idempotency_key: str, payload: dict) -> None:
        """Durably queue a projection/wake event in the same transaction as its state change."""
        cur.execute(
            "INSERT INTO lab_transactional_outbox(id,project_id,event_type,subject_id,payload,idempotency_key,created_at) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(project_id,idempotency_key) DO NOTHING",
            (str(uuid.uuid4()), project_id, event_type, subject_id, json.dumps(payload), idempotency_key, self._now()),
        )

    @staticmethod
    def authority_key(project_id: str, scientific_object_id: str, canonical_subject_version: str, gate_key: str) -> str:
        """Return the stable identity for one current scientific transition."""
        return ":".join((str(project_id), str(scientific_object_id), str(canonical_subject_version), str(gate_key).upper()))

    def canonical_objective_create(self, project_id: str, objective_type: str, title: str,
                                   scientific_question: str, current_goal: str = "",
                                   priority: float = 0, parent_objective_id: str | None = None) -> dict:
        """Create one versioned objective; active meaning is never overwritten in place."""
        if not title.strip() or not scientific_question.strip():
            raise GPUError("CANONICAL_OBJECTIVE_FIELDS_REQUIRED", "title and scientific_question")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM research_projects WHERE id=%s", (project_id,))
            if not cur.fetchone():
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"objective:{project_id}:{title.strip()}",))
            cur.execute("SELECT * FROM canonical_objectives WHERE project_id=%s AND title=%s ORDER BY version DESC LIMIT 1 FOR UPDATE", (project_id, title.strip()))
            previous = cur.fetchone()
            if previous and previous["status"] == "ACTIVE":
                if previous["scientific_question"] == scientific_question.strip() and previous["current_goal"] == current_goal.strip():
                    return self._record(previous) or {}
                raise GPUError("CANONICAL_OBJECTIVE_VERSION_TRANSITION_REQUIRED", str(previous["id"]))
            version = int(previous["version"]) + 1 if previous else 1
            cur.execute("INSERT INTO canonical_objectives(id,project_id,version,objective_type,title,scientific_question,status,priority,current_goal,parent_objective_id,supersedes_objective_id,created_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s,'ACTIVE',%s,%s,%s,%s,%s,%s)", (ident, project_id, version, objective_type.strip().upper(), title.strip(), scientific_question.strip(), priority, current_goal.strip(), parent_objective_id, previous["id"] if previous else None, now, now))
            if previous:
                cur.execute("UPDATE canonical_objectives SET status='SUPERSEDED',superseded_by_objective_id=%s,updated_at=%s WHERE id=%s", (ident, now, previous["id"]))
            self._event(cur, project_id, "CANONICAL_OBJECTIVE_CREATED", ident, {"version": version, "supersedes": str(previous["id"]) if previous else None})
            cur.execute("SELECT * FROM canonical_objectives WHERE id=%s", (ident,))
            return self._record(cur.fetchone()) or {}

    def work_propose(self, project_id: str, worker_id: str, session_id: str, proposed_mode: str,
                     proposed_role: str, rationale: str, canonical_objective_id: str | None = None,
                     hypothesis_branch_id: str | None = None,
                     target_id: str | None = None, authority_key_hint: str | None = None,
                     equivalence_key_hint: str | None = None, expected_scientific_value: float | None = None,
                     dependency_refs: list[dict] | None = None) -> dict:
        """Persist a worker proposal and merge it into existing canonical work when possible."""
        if not proposed_mode.strip() or not proposed_role.strip() or not rationale.strip():
            raise GPUError("WORK_PROPOSAL_FIELDS_REQUIRED", "proposed_mode, proposed_role, and rationale")
        if dependency_refs is not None and not isinstance(dependency_refs, list):
            raise GPUError("INVALID_WORK_PROPOSAL_DEPENDENCIES", "dependency_refs must be a list")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            self._worker(cur, worker_id); self._session(cur, session_id, worker_id, project_id)
            if canonical_objective_id:
                cur.execute(
                    "SELECT id,status FROM canonical_objectives WHERE id=%s AND project_id=%s FOR SHARE",
                    (canonical_objective_id, project_id),
                )
                objective = cur.fetchone()
                if not objective:
                    raise GPUError("CANONICAL_OBJECTIVE_NOT_FOUND", canonical_objective_id)
                if objective["status"] != "ACTIVE":
                    raise GPUError("CANONICAL_OBJECTIVE_NOT_ACTIVE", str(objective["status"]))
            if hypothesis_branch_id:
                cur.execute(
                    "SELECT canonical_objective_id,state FROM hypothesis_branches WHERE id=%s AND project_id=%s FOR SHARE",
                    (hypothesis_branch_id, project_id),
                )
                branch = cur.fetchone()
                if not branch:
                    raise GPUError("HYPOTHESIS_BRANCH_NOT_FOUND", hypothesis_branch_id)
                if branch["state"] not in {"OPEN", "WAITING", "BLOCKED", "SUPPORTED_WITHIN_SCOPE", "WEAKENED"}:
                    raise GPUError("HYPOTHESIS_BRANCH_NOT_ACTIONABLE", str(branch["state"]))
                if canonical_objective_id and str(branch["canonical_objective_id"]) != str(canonical_objective_id):
                    raise GPUError("HYPOTHESIS_BRANCH_OBJECTIVE_MISMATCH", hypothesis_branch_id)
                canonical_objective_id = canonical_objective_id or str(branch["canonical_objective_id"])
            existing_id = None
            existing_terminal = False
            if authority_key_hint:
                cur.execute("SELECT id,status FROM lab_work_items WHERE project_id=%s AND authority_key=%s AND authority_status='AUTHORITATIVE' AND status=ANY(%s) ORDER BY created_at LIMIT 1", (project_id, authority_key_hint, list(ACTIVE_WORK_STATUSES | {"COMPLETED"})))
                row = cur.fetchone()
                existing_id = str(row["id"]) if row else None
                existing_terminal = bool(row and row["status"] == "COMPLETED")
            if not existing_id and equivalence_key_hint:
                cur.execute("SELECT id,status FROM lab_work_items WHERE project_id=%s AND equivalence_key=%s AND status=ANY(%s) ORDER BY created_at LIMIT 1", (project_id, equivalence_key_hint, list(EQUIVALENCE_ACTIVE_WORK_STATUSES | {"COMPLETED"})))
                row = cur.fetchone()
                existing_id = str(row["id"]) if row else None
                existing_terminal = bool(row and row["status"] == "COMPLETED")
            status = "SATISFIED_BY_TERMINAL" if existing_terminal else "MERGED_INTO_EXISTING" if existing_id else "PROPOSED"
            cur.execute("INSERT INTO lab_work_proposals(id,project_id,proposed_by_worker_id,canonical_objective_id,hypothesis_branch_id,target_id,proposed_mode,proposed_role,proposed_authority_key_hint,proposed_equivalence_key_hint,rationale,expected_scientific_value,dependency_refs,status,canonical_work_item_id,created_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (ident, project_id, worker_id, canonical_objective_id, hypothesis_branch_id, target_id, proposed_mode, proposed_role, authority_key_hint, equivalence_key_hint, rationale.strip(), expected_scientific_value, json.dumps(dependency_refs or []), status, existing_id, now, now))
            self._event(cur, project_id, "WORK_PROPOSAL_" + status, ident, {"canonical_work_item_id": existing_id})
            cur.execute("SELECT * FROM lab_work_proposals WHERE id=%s", (ident,))
            return self._record(cur.fetchone()) or {}

    def hypothesis_branch_create(
        self, project_id: str, canonical_objective_id: str, state: str = "OPEN",
        question_id: str | None = None, hypothesis_ids: list[str] | None = None,
        mechanistic_niche_id: str | None = None, scientific_distance: str | None = None,
        architecture_lineage_id: str | None = None, priority: float = 0,
        branch_dependencies: list[dict] | None = None, branch_blocking_scope: str = "BRANCH",
    ) -> dict:
        """Create an explicit independently-progressable branch under an active objective."""
        state = self._validate(state, BRANCH_STATES, "HYPOTHESIS_BRANCH_STATE")
        scope = self._validate(branch_blocking_scope, {"BRANCH", "PORTFOLIO", "OBJECTIVE_GLOBAL"}, "HYPOTHESIS_BRANCH_SCOPE")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM canonical_objectives WHERE id=%s AND project_id=%s FOR SHARE", (canonical_objective_id, project_id))
            objective = cur.fetchone()
            if not objective:
                raise GPUError("CANONICAL_OBJECTIVE_NOT_FOUND", canonical_objective_id)
            if objective["status"] != "ACTIVE":
                raise GPUError("CANONICAL_OBJECTIVE_NOT_ACTIVE", objective["status"])
            cur.execute(
                "INSERT INTO hypothesis_branches(id,project_id,canonical_objective_id,question_id,hypothesis_ids,mechanistic_niche_id,scientific_distance,architecture_lineage_id,state,priority,branch_dependencies,branch_blocking_scope,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (ident, project_id, canonical_objective_id, question_id, json.dumps(hypothesis_ids or []), mechanistic_niche_id,
                 scientific_distance, architecture_lineage_id, state, priority, json.dumps(branch_dependencies or []), scope, now),
            )
            self._event(cur, project_id, "HYPOTHESIS_BRANCH_CREATED", ident, {"canonical_objective_id": canonical_objective_id, "state": state})
            cur.execute("SELECT * FROM hypothesis_branches WHERE id=%s", (ident,))
            return self._record(cur.fetchone()) or {}

    @staticmethod
    def _canonical_json_hash(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    def _worker(self, cur, worker_id: str) -> dict:
        cur.execute("SELECT * FROM research_workers WHERE id=%s AND enabled=TRUE", (worker_id,))
        worker = cur.fetchone()
        if not worker:
            raise GPUError("LAB_WORKER_NOT_FOUND", worker_id)
        return worker

    def _session(self, cur, session_id: str, worker_id: str, project_id: str) -> dict:
        cur.execute(
            "SELECT * FROM research_worker_sessions WHERE id=%s AND worker_id=%s "
            "AND current_project_id=%s FOR UPDATE",
            (session_id, worker_id, project_id),
        )
        session = cur.fetchone()
        if not session or session["status"] in {"DISCONNECTED", "EXPIRED"}:
            raise GPUError("LAB_SESSION_NOT_ACTIVE", session_id)
        return session

    def budget_set(self, project_id: str, worker_id: str, session_id: str,
                   limits: dict[str, Any]) -> dict:
        allowed = {
            "max_active_workers", "max_parallel_branches", "max_concurrent_gpu_runs",
            "max_concurrent_training_runs", "max_concurrent_expensive_llm_tasks",
            "max_concurrent_expensive_speculative_runs",
            "project_compute_budget", "project_llm_budget",
        }
        if not isinstance(limits, dict) or set(limits) - allowed:
            raise GPUError("INVALID_LAB_BUDGET", "Unsupported LabBudget field")
        if any(not isinstance(value, (int, float)) or value < 0 for value in limits.values()):
            raise GPUError("INVALID_LAB_BUDGET", "Budget values must be non-negative numbers")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM research_projects WHERE id=%s", (project_id,))
            if not cur.fetchone():
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            self._worker(cur, worker_id)
            self._session(cur, session_id, worker_id, project_id)
            cur.execute(
                "INSERT INTO lab_project_budgets(project_id,limits,updated_at) VALUES(%s,%s,%s) "
                "ON CONFLICT(project_id) DO UPDATE SET limits=EXCLUDED.limits,updated_at=EXCLUDED.updated_at",
                (project_id, json.dumps(limits), now),
            )
            self._event(cur, project_id, "LAB_BUDGET_UPDATED", None, {"limits": limits})
        return {"project_id": project_id, "limits": limits, "updated_at": now}

    def _budget(self, cur, project_id: str) -> dict[str, Any]:
        cur.execute("SELECT limits FROM lab_project_budgets WHERE project_id=%s", (project_id,))
        row = cur.fetchone()
        return row["limits"] if row else {}

    def _enforce_claim_budget(self, cur, project_id: str, item: dict, session_id: str) -> None:
        limits = self._budget(cur, project_id)
        active = ("CLAIMED", "RUNNING")
        max_workers = int(limits.get("max_active_workers", 0) or 0)
        if max_workers:
            cur.execute("SELECT count(DISTINCT assigned_session_id) AS count FROM lab_work_items WHERE project_id=%s AND status=ANY(%s) AND assigned_session_id<>%s", (project_id, list(active), session_id))
            if cur.fetchone()["count"] >= max_workers:
                raise GPUError("LAB_WORKER_BUDGET_EXCEEDED", "max_active_workers")
        max_branches = int(limits.get("max_parallel_branches", 0) or 0)
        if max_branches and item.get("branch_id"):
            cur.execute(
                "SELECT count(DISTINCT branch_id) AS count FROM lab_work_items "
                "WHERE project_id=%s AND status=ANY(%s) AND branch_id IS NOT NULL AND branch_id<>%s",
                (project_id, list(active), item["branch_id"]),
            )
            if cur.fetchone()["count"] >= max_branches:
                raise GPUError("LAB_BRANCH_BUDGET_EXCEEDED", "max_parallel_branches")
        kind = item["kind"].upper()
        max_gpu = int(limits.get("max_concurrent_gpu_runs", 0) or 0)
        if max_gpu and kind in {"RUN_EXPERIMENT", "TRAINING_RUN"}:
            cur.execute("SELECT count(*) AS count FROM research_objects WHERE project_id=%s AND kind='ExperimentRun' AND status IN ('running','RESERVED','unknown')", (project_id,))
            if cur.fetchone()["count"] >= max_gpu:
                raise GPUError("LAB_GPU_BUDGET_EXCEEDED", "max_concurrent_gpu_runs")
        max_training = int(limits.get("max_concurrent_training_runs", 0) or 0)
        if max_training and kind == "TRAINING_RUN":
            cur.execute("SELECT count(*) AS count FROM lab_work_items WHERE project_id=%s AND kind='TRAINING_RUN' AND status=ANY(%s)", (project_id, list(active)))
            if cur.fetchone()["count"] >= max_training:
                raise GPUError("LAB_TRAINING_BUDGET_EXCEEDED", "max_concurrent_training_runs")
        max_llm = int(limits.get("max_concurrent_expensive_llm_tasks", 0) or 0)
        if max_llm and kind in {"EXPENSIVE_LLM_TASK", "LLM_EVALUATION", "LLM_REVIEW"}:
            cur.execute("SELECT count(*) AS count FROM lab_work_items WHERE project_id=%s AND kind=ANY(%s) AND status=ANY(%s)", (project_id, ["EXPENSIVE_LLM_TASK", "LLM_EVALUATION", "LLM_REVIEW"], list(active)))
            if cur.fetchone()["count"] >= max_llm:
                raise GPUError("LAB_LLM_BUDGET_EXCEEDED", "max_concurrent_expensive_llm_tasks")
        compute_budget = float(limits.get("project_compute_budget", 0) or 0)
        if compute_budget and item.get("estimated_cost") is not None:
            cur.execute("SELECT COALESCE(sum(estimated_cost), 0) AS cost FROM lab_work_items WHERE project_id=%s AND status=ANY(%s)", (project_id, list(active)))
            if float(cur.fetchone()["cost"]) + float(item["estimated_cost"]) > compute_budget:
                raise GPUError("LAB_COMPUTE_BUDGET_EXCEEDED", "project_compute_budget")
        max_expensive_speculative = int(limits.get("max_concurrent_expensive_speculative_runs", 0) or 0)
        if max_expensive_speculative and item.get("speculation_class") == "EXPENSIVE_SPECULATIVE":
            cur.execute(
                "SELECT count(*) AS count FROM lab_work_items WHERE project_id=%s "
                "AND speculation_class='EXPENSIVE_SPECULATIVE' AND status=ANY(%s)",
                (project_id, list(active)),
            )
            if cur.fetchone()["count"] >= max_expensive_speculative:
                raise GPUError("LAB_SPECULATIVE_BUDGET_EXCEEDED", "max_concurrent_expensive_speculative_runs")
        if item.get("canonical_objective_id"):
            cur.execute(
                "SELECT * FROM hypothesis_portfolios WHERE project_id=%s AND canonical_objective_id=%s "
                "AND status='ACTIVE' ORDER BY created_at DESC LIMIT 1 FOR SHARE",
                (project_id, item["canonical_objective_id"]),
            )
            portfolio = cur.fetchone()
            if portfolio:
                def active_count(where: str, args: list[Any]) -> int:
                    cur.execute(
                        "SELECT count(*) AS count FROM lab_work_items WHERE project_id=%s "
                        "AND canonical_objective_id=%s AND status=ANY(%s) AND " + where,
                        [project_id, item["canonical_objective_id"], list(active), *args],
                    )
                    return int(cur.fetchone()["count"])
                if portfolio["branch_budget"] and item.get("branch_id"):
                    cur.execute(
                        "SELECT count(DISTINCT branch_id) AS count FROM lab_work_items WHERE project_id=%s "
                        "AND canonical_objective_id=%s AND status=ANY(%s) AND branch_id IS NOT NULL AND branch_id<>%s",
                        (project_id, item["canonical_objective_id"], list(active), item["branch_id"]),
                    )
                    if int(cur.fetchone()["count"]) >= int(portfolio["branch_budget"]):
                        raise GPUError("PORTFOLIO_BRANCH_BUDGET_EXCEEDED", str(portfolio["id"]))
                if portfolio["scientific_concurrency_budget"] and active_count("TRUE", []) >= int(portfolio["scientific_concurrency_budget"]):
                    raise GPUError("PORTFOLIO_SCIENTIFIC_CONCURRENCY_EXCEEDED", str(portfolio["id"]))
                resource_limit = {
                    "GPU": portfolio["gpu_concurrency_budget"],
                    "TRAINING": portfolio["training_concurrency_budget"],
                    "REASONING": portfolio["reasoning_concurrency_budget"],
                }.get(item.get("resource_class"))
                if resource_limit and active_count("resource_class=%s", [item.get("resource_class")]) >= int(resource_limit):
                    raise GPUError("PORTFOLIO_RESOURCE_BUDGET_EXCEEDED", str(item.get("resource_class")))

    def join(self, worker_id: str | None, worker_name: str | None, runtime_type: str,
             project_id: str, capabilities: dict[str, Any] | None = None,
             runtime_metadata: dict[str, Any] | None = None, session_id: str | None = None) -> dict:
        runtime_type = self._validate(runtime_type, RUNTIME_TYPES, "LAB_RUNTIME_TYPE")
        if not worker_id and not worker_name:
            raise GPUError("LAB_WORKER_IDENTITY_REQUIRED", "worker_id or worker_name is required")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM research_projects WHERE id=%s", (project_id,))
            if not cur.fetchone():
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            if worker_id:
                worker = self._worker(cur, worker_id)
            else:
                cur.execute("SELECT * FROM research_workers WHERE display_name=%s FOR UPDATE", (worker_name,))
                worker = cur.fetchone()
                if not worker:
                    worker_id = str(uuid.uuid4())
                    cur.execute("INSERT INTO research_workers(id,display_name,worker_type,capabilities,enabled,created_at) VALUES(%s,%s,%s,%s,TRUE,%s)",
                                (worker_id, worker_name, runtime_type, json.dumps(capabilities or {}), now))
                    cur.execute("SELECT * FROM research_workers WHERE id=%s", (worker_id,))
                    worker = cur.fetchone()
            recovered = False
            if session_id:
                cur.execute("SELECT * FROM research_worker_sessions WHERE id=%s AND worker_id=%s FOR UPDATE", (session_id, worker["id"]))
                session = cur.fetchone()
                if session:
                    recovered = True
                    cur.execute("UPDATE research_worker_sessions SET runtime_type=%s,runtime_metadata=%s,current_project_id=%s,status='ACTIVE',last_heartbeat_at=%s,disconnected_at=NULL WHERE id=%s",
                                (runtime_type, json.dumps(runtime_metadata or {}), project_id, now, session_id))
                else:
                    session_id = None
            if not session_id:
                session_id = str(uuid.uuid4())
                cur.execute("INSERT INTO research_worker_sessions(id,worker_id,runtime_type,runtime_metadata,current_project_id,status,joined_at,last_heartbeat_at,context_version) VALUES(%s,%s,%s,%s,%s,'ACTIVE',%s,%s,'{}')",
                            (session_id, worker["id"], runtime_type, json.dumps(runtime_metadata or {}), project_id, now, now))
            self._event(cur, project_id, "WORKER_SESSION_RECOVERED" if recovered else "WORKER_JOINED", None,
                        {"worker_id": str(worker["id"]), "session_id": session_id, "runtime_type": runtime_type})
        # Joining only creates or resumes a session.  Reconciliation is a
        # sync-time responsibility; doing it here makes every new worker wait
        # behind lease/dependency scans before it can even receive its state.
        return {"worker": self._record(worker), "session_id": session_id, "recovered": recovered,
                "lab_state": self.state_get(project_id, session_id, reconcile=False)}

    def renew_session(self, session_id: str, worker_id: str, project_id: str) -> dict:
        """Restore an expired session identity without resurrecting its lease.

        Expiry remains meaningful for work ownership: this operation never
        changes a WorkItem, a lease, or an attached execution.  It only avoids
        forcing a worker to create an unnecessary successor identity before it
        can continue coordination work.
        """
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            self._worker(cur, worker_id)
            cur.execute(
                "SELECT * FROM research_worker_sessions WHERE id=%s AND worker_id=%s "
                "AND current_project_id=%s FOR UPDATE",
                (session_id, worker_id, project_id),
            )
            session = cur.fetchone()
            if not session:
                raise GPUError("LAB_SESSION_NOT_FOUND", session_id)
            if session["status"] == "DISCONNECTED":
                raise GPUError("LAB_SESSION_NOT_RENEWABLE", "Disconnected sessions require lab_join")
            cur.execute(
                "UPDATE research_worker_sessions SET status='ACTIVE',last_heartbeat_at=%s,"
                "disconnected_at=NULL,current_work_item_id=NULL,active_role=NULL WHERE id=%s",
                (now, session_id),
            )
            self._event(cur, project_id, "WORKER_SESSION_RENEWED", None, {
                "worker_id": worker_id, "session_id": session_id,
                "previous_status": session["status"], "lease_restored": False,
            })
        return {"session_id": session_id, "worker_id": worker_id, "project_id": project_id, "renewed": True, "lease_restored": False, "lab_state": self.state_get(project_id, session_id, reconcile=False)}

    def gate_ensure(
        self, project_id: str, gate_key: str, scientific_object_id: str,
        canonical_subject_version: str, worker_id: str, session_id: str,
        semantic_review_required: bool = True,
    ) -> dict:
        """Create or return one durable, version-bound scientific gate."""
        gate_key = gate_key.strip().upper()
        if not gate_key or not scientific_object_id or not canonical_subject_version:
            raise GPUError("SCIENTIFIC_GATE_IDENTITY_REQUIRED", "gate_key, scientific_object_id, and canonical_subject_version")
        authority_key = self.authority_key(project_id, scientific_object_id, canonical_subject_version, gate_key)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, session_id, worker_id, project_id)
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (authority_key,))
            cur.execute(
                "SELECT * FROM scientific_gates WHERE project_id=%s AND gate_key=%s AND scientific_object_id=%s "
                "AND canonical_subject_version=%s FOR UPDATE",
                (project_id, gate_key, scientific_object_id, canonical_subject_version),
            )
            existing = cur.fetchone()
            if existing:
                return self._record(existing) or {}
            ident = str(uuid.uuid4())
            if not isinstance(semantic_review_required, bool):
                raise GPUError("SCIENTIFIC_GATE_SEMANTIC_REVIEW_REQUIRED_INVALID", str(semantic_review_required))
            cur.execute(
                "INSERT INTO scientific_gates(id,project_id,gate_key,scientific_object_id,canonical_subject_version,"
                "authority_key,status,coordination_version,semantic_review_required,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,'PENDING',%s,%s,%s) RETURNING *",
                (ident, project_id, gate_key, scientific_object_id, canonical_subject_version, authority_key,
                 COORDINATION_VERSION, semantic_review_required, now),
            )
            gate = cur.fetchone()
            self._event(cur, project_id, "SCIENTIFIC_GATE_CREATED", ident, {
                "gate_key": gate_key, "scientific_object_id": scientific_object_id,
                "canonical_subject_version": canonical_subject_version, "authority_key": authority_key,
                "coordination_version": COORDINATION_VERSION, "semantic_review_required": semantic_review_required,
            })
            return self._record(gate) or {}

    def gate_get(self, gate_id: str) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM scientific_gates WHERE id=%s", (gate_id,))
            gate = cur.fetchone()
            if not gate:
                raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", gate_id)
            result = self._record(gate) or {}
            if gate["deterministic_preflight_id"]:
                cur.execute("SELECT * FROM lab_deterministic_preflights WHERE id=%s", (gate["deterministic_preflight_id"],))
                result["deterministic_preflight"] = self._record(cur.fetchone())
            return result

    def gate_work_ensure(
        self, gate_id: str, kind: str, title: str, description: str, scientific_role: str,
        worker_id: str, session_id: str, priority: float = 0, expected_value: float | None = None,
        estimated_cost: float | None = None, dependencies: list[dict] | None = None,
        recovery_policy: dict[str, Any] | None = None, branch_id: str | None = None,
    ) -> dict:
        """Create or reuse the sole active authoritative WorkItem for a ScientificGate."""
        gate = self.gate_get(gate_id)
        if gate["status"] in {"SUPERSEDED", "INVALID", "PASS", "FAIL"}:
            raise GPUError("SCIENTIFIC_GATE_NOT_ACTIONABLE", gate["status"])
        refs = {"scientific_object_id": gate["scientific_object_id"], "gate_id": gate_id}
        return self.create_work(
            gate["project_id"], kind, title, description, scientific_role,
            worker_id, priority, expected_value, estimated_cost, refs, dependencies,
            gate["authority_key"], None, session_id, gate["authority_key"], gate_id,
            gate["canonical_subject_version"], "AUTHORITATIVE", gate["scientific_object_id"], recovery_policy,
            canonical_objective_id=None, branch_id=branch_id,
        )

    @staticmethod
    def _normalize_preflight_checks(checks: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not checks:
            raise GPUError("PREFLIGHT_CHECKS_REQUIRED", "At least one deterministic check is required")
        normalized: dict[str, dict[str, Any]] = {}
        for name, value in checks.items():
            if not isinstance(name, str) or not name.strip():
                raise GPUError("PREFLIGHT_CHECK_NAME_INVALID", str(name))
            if isinstance(value, bool):
                normalized[name] = {"passed": value}
            elif isinstance(value, dict) and isinstance(value.get("passed"), bool):
                normalized[name] = {key: value[key] for key in sorted(value)}
            else:
                raise GPUError("PREFLIGHT_CHECK_INVALID", name)
        return {name: normalized[name] for name in sorted(normalized)}

    def preflight_run(
        self, gate_id: str, worker_id: str, session_id: str, checks: dict[str, Any],
        validator_version: str = COORDINATION_VERSION,
    ) -> dict:
        """Persist one immutable deterministic readiness result for a gate subject version."""
        normalized = self._normalize_preflight_checks(checks)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM scientific_gates WHERE id=%s FOR UPDATE", (gate_id,))
            gate = cur.fetchone()
            if not gate:
                raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", gate_id)
            self._session(cur, session_id, worker_id, str(gate["project_id"]))
            if gate["status"] in {"SUPERSEDED", "INVALID", "PASS", "FAIL"}:
                raise GPUError("SCIENTIFIC_GATE_NOT_PREFLIGHTABLE", gate["status"])
            subject = {
                "scientific_object_id": gate["scientific_object_id"],
                "canonical_subject_version": gate["canonical_subject_version"],
                "checks": normalized,
            }
            subject_hash = self._canonical_json_hash(subject)
            cur.execute(
                "SELECT * FROM lab_deterministic_preflights WHERE project_id=%s AND scientific_object_id=%s "
                "AND canonical_subject_version=%s AND subject_hash=%s AND validator_version=%s",
                (gate["project_id"], gate["scientific_object_id"], gate["canonical_subject_version"], subject_hash, validator_version),
            )
            result = cur.fetchone()
            reused = result is not None
            if not result:
                failures = [name for name, value in normalized.items() if not value["passed"]]
                warnings = [name for name, value in normalized.items() if value.get("warning")]
                status = "FAIL" if failures else "PASS"
                result_hash = self._canonical_json_hash({**subject, "status": status, "validator_version": validator_version})
                ident = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO lab_deterministic_preflights(id,project_id,gate_id,scientific_object_id,"
                    "canonical_subject_version,subject_hash,checks,failures,warnings,result_hash,status,validator_version,created_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
                    (ident, gate["project_id"], gate_id, gate["scientific_object_id"], gate["canonical_subject_version"], subject_hash,
                     json.dumps(normalized), json.dumps(failures), json.dumps(warnings), result_hash, status, validator_version, now),
                )
                result = cur.fetchone()
            next_status = "AWAITING_SEMANTIC_REVIEW" if result["status"] == "PASS" else "PREFLIGHT_FAILED"
            cur.execute(
                "UPDATE scientific_gates SET deterministic_preflight_id=%s,status=%s,resolved_at=NULL WHERE id=%s",
                (result["id"], next_status, gate_id),
            )
            self._event(cur, gate["project_id"], "DETERMINISTIC_PREFLIGHT_REUSED" if reused else "DETERMINISTIC_PREFLIGHT_COMPLETED", result["id"], {
                "gate_id": gate_id, "status": result["status"], "subject_hash": subject_hash,
                "validator_version": validator_version,
            })
            self._event(cur, gate["project_id"], "SCIENTIFIC_GATE_PREFLIGHT_FAILED" if result["status"] == "FAIL" else "SCIENTIFIC_GATE_PREFLIGHT_PASSED", gate_id, {"preflight_id": str(result["id"])})
        return {"gate": self.gate_get(gate_id), "preflight": self._record(result), "reused": reused}

    def gate_resolve(
        self, gate_id: str, worker_id: str, session_id: str, semantic_status: str,
        semantic_review_work_item_id: str | None = None, rationale: str = "",
    ) -> dict:
        """Record scientific review outcome only after deterministic preflight has passed."""
        semantic_status = self._validate(semantic_status, {"PASS", "FAIL", "INVALID"}, "SCIENTIFIC_GATE_STATUS")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM scientific_gates WHERE id=%s FOR UPDATE", (gate_id,))
            gate = cur.fetchone()
            if not gate:
                raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", gate_id)
            self._session(cur, session_id, worker_id, str(gate["project_id"]))
            if gate["status"] in {"SUPERSEDED", "INVALID"}:
                raise GPUError("SCIENTIFIC_GATE_NOT_RESOLVABLE", gate["status"])
            if gate["status"] in {"PASS", "FAIL"}:
                if gate["status"] != semantic_status:
                    raise GPUError("SCIENTIFIC_GATE_ALREADY_RESOLVED", gate["status"])
                return {
                    "gate": self._record(gate) or {},
                    "dependency_changes": {"ready": 0, "invalidated": 0},
                    "idempotent": True,
                }
            if not gate["deterministic_preflight_id"]:
                raise GPUError("SCIENTIFIC_GATE_PREFLIGHT_REQUIRED", gate_id)
            cur.execute("SELECT status FROM lab_deterministic_preflights WHERE id=%s", (gate["deterministic_preflight_id"],))
            preflight = cur.fetchone()
            if not preflight or preflight["status"] != "PASS":
                raise GPUError("SCIENTIFIC_GATE_PREFLIGHT_NOT_PASS", gate_id)
            if gate["semantic_review_required"] and not semantic_review_work_item_id:
                raise GPUError("SCIENTIFIC_GATE_SEMANTIC_REVIEW_REQUIRED", gate_id)
            if semantic_review_work_item_id:
                cur.execute("SELECT status,gate_id FROM lab_work_items WHERE id=%s AND project_id=%s", (semantic_review_work_item_id, gate["project_id"]))
                review = cur.fetchone()
                if not review or str(review["gate_id"] or "") != str(gate_id):
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_MISMATCH", semantic_review_work_item_id)
                if review["status"] != "COMPLETED":
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_NOT_COMPLETED", semantic_review_work_item_id)
            cur.execute(
                "UPDATE scientific_gates SET status=%s,semantic_review_work_item_id=COALESCE(%s,semantic_review_work_item_id),resolved_at=%s WHERE id=%s",
                (semantic_status, semantic_review_work_item_id, now, gate_id),
            )
            self._event(cur, gate["project_id"], "SCIENTIFIC_GATE_RESOLVED", gate_id, {
                "status": semantic_status, "semantic_review_work_item_id": semantic_review_work_item_id,
                "rationale": rationale[:4000],
            })
        dependency_changes = self.resolve_dependencies(str(gate["project_id"]))
        return {"gate": self.gate_get(gate_id), "dependency_changes": dependency_changes}

    def supersede_subject(
        self, project_id: str, old_subject_id: str, new_subject_id: str, rationale: str,
        worker_id: str, session_id: str, successor_gate_id: str | None = None,
    ) -> dict[str, int]:
        """Historically preserve but operationally retire work bound to a replaced subject."""
        if not old_subject_id or not new_subject_id or not rationale.strip():
            raise GPUError("SCIENTIFIC_SUPERSESSION_IDENTITY_REQUIRED", "old_subject_id, new_subject_id, and rationale")
        now, superseded = self._now(), 0
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, session_id, worker_id, project_id)
            if successor_gate_id:
                cur.execute("SELECT id FROM scientific_gates WHERE id=%s AND project_id=%s", (successor_gate_id, project_id))
                if not cur.fetchone():
                    raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", successor_gate_id)
            cur.execute(
                "SELECT * FROM lab_work_items WHERE project_id=%s AND status=ANY(%s) AND (subject_id=%s "
                "OR related_refs->>'experiment_id'=%s OR related_refs->>'experiment_run_id'=%s "
                "OR related_refs->>'run_id'=%s) FOR UPDATE",
                (project_id, list(ACTIVE_WORK_STATUSES), old_subject_id, old_subject_id, old_subject_id, old_subject_id),
            )
            for item in cur.fetchall():
                cur.execute(
                    "UPDATE lab_work_leases SET released_at=%s,release_reason='SUBJECT_SUPERSEDED' "
                    "WHERE work_item_id=%s AND released_at IS NULL",
                    (now, item["id"]),
                )
                cur.execute(
                    "UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',superseded_by=%s,"
                    "invalidated_reason=%s,invalidated_at=%s,assigned_worker_id=NULL,assigned_session_id=NULL,"
                    "lease_id=NULL,work_version=work_version+1,updated_at=%s WHERE id=%s",
                    (successor_gate_id, rationale[:4000], now, now, item["id"]),
                )
                cur.execute(
                    "UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',"
                    "last_heartbeat_at=%s WHERE current_work_item_id=%s",
                    (now, item["id"]),
                )
                self._event(cur, project_id, "WORK_ITEM_SUPERSEDED", item["id"], {
                    "old_subject_id": old_subject_id, "new_subject_id": new_subject_id,
                    "successor_gate_id": successor_gate_id, "rationale": rationale[:4000],
                })
                superseded += 1
            cur.execute(
                "UPDATE scientific_gates SET status='SUPERSEDED',superseded_by=%s,invalidation_reason=%s,resolved_at=%s "
                "WHERE project_id=%s AND scientific_object_id=%s AND status NOT IN ('SUPERSEDED','INVALID')",
                (successor_gate_id, rationale[:4000], now, project_id, old_subject_id),
            )
            self._event(cur, project_id, "SCIENTIFIC_SUBJECT_SUPERSEDED", None, {
                "old_subject_id": old_subject_id, "new_subject_id": new_subject_id,
                "successor_gate_id": successor_gate_id, "work_items_superseded": superseded,
                "rationale": rationale[:4000],
            })
        dependency_changes = self.resolve_dependencies(project_id)
        return {"work_items_superseded": superseded, **dependency_changes}

    def supersede_gate_version(
        self, old_gate_id: str, successor_gate_id: str, worker_id: str,
        session_id: str, rationale: str,
    ) -> dict[str, Any]:
        """Retire one stale gate version without retiring its scientific subject.

        A subject can legitimately receive a newer canonical version after a
        harness or deterministic-preflight repair.  Subject-level supersession
        is unsafe here because both gate versions refer to the same science.
        """
        if not rationale.strip():
            raise GPUError("SCIENTIFIC_GATE_SUPERSESSION_RATIONALE_REQUIRED", "rationale")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM scientific_gates WHERE id=ANY(%s) FOR UPDATE", ([old_gate_id, successor_gate_id],))
            gates = {str(item["id"]): item for item in cur.fetchall()}
            old, successor = gates.get(old_gate_id), gates.get(successor_gate_id)
            if not old or not successor:
                raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", "old or successor gate")
            self._session(cur, session_id, worker_id, str(old["project_id"]))
            if str(old["project_id"]) != str(successor["project_id"]):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", successor_gate_id)
            if old["scientific_object_id"] != successor["scientific_object_id"] or old["gate_key"] != successor["gate_key"]:
                raise GPUError("SCIENTIFIC_GATE_SUPERSESSION_MISMATCH", "Gate subject and key must match")
            if old["canonical_subject_version"] == successor["canonical_subject_version"]:
                raise GPUError("SCIENTIFIC_GATE_SUPERSESSION_SAME_VERSION", old_gate_id)
            if old["status"] in {"PASS", "FAIL", "INVALID"}:
                raise GPUError("SCIENTIFIC_GATE_NOT_SUPERSEDABLE", old["status"])
            cur.execute(
                "UPDATE scientific_gates SET status='SUPERSEDED',superseded_by=%s,invalidation_reason=%s,resolved_at=%s WHERE id=%s",
                (successor_gate_id, rationale.strip()[:4000], now, old_gate_id),
            )
            cur.execute(
                "UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',superseded_by=%s,"
                "invalidated_reason=%s,invalidated_at=%s,updated_at=%s WHERE gate_id=%s "
                "AND status=ANY(%s)",
                (successor_gate_id, rationale.strip()[:4000], now, now, old_gate_id, list(ACTIVE_WORK_STATUSES)),
            )
            self._event(cur, old["project_id"], "SCIENTIFIC_GATE_VERSION_SUPERSEDED", old_gate_id, {
                "successor_gate_id": successor_gate_id,
                "old_version": old["canonical_subject_version"],
                "successor_version": successor["canonical_subject_version"],
                "rationale": rationale.strip()[:4000],
            })
        dependency_changes = self.resolve_dependencies(str(old["project_id"]))
        return {"old_gate_id": old_gate_id, "successor_gate_id": successor_gate_id, "status": "SUPERSEDED", "dependency_changes": dependency_changes}

    def create_work(self, project_id: str, kind: str, title: str, description: str,
                    scientific_role: str, created_by: str | None = None, priority: float = 0,
                    expected_value: float | None = None, estimated_cost: float | None = None,
                    related_refs: dict[str, Any] | None = None, dependencies: list[dict] | None = None,
                    equivalence_key: str | None = None, parent_work_item_id: str | None = None,
                    created_session_id: str | None = None, authority_key: str | None = None,
                    gate_id: str | None = None, canonical_subject_version: str | None = None,
                    authority_status: str = "SUPPORTING", subject_id: str | None = None,
                    recovery_policy: dict[str, Any] | None = None,
                    dormant_until_dependencies: bool = False,
                    canonical_objective_id: str | None = None,
                    branch_id: str | None = None,
                    speculation_class: str = "NON_SPECULATIVE",
                    speculation_condition: dict[str, Any] | None = None,
                    resource_class: str | None = None,
                    dependency_scope: str = "WORKITEM_LOCAL") -> dict:
        now, ident = self._now(), str(uuid.uuid4())
        dependencies = dependencies or []
        authority_status = self._validate(authority_status, AUTHORITY_STATUSES, "LAB_WORK_AUTHORITY_STATUS")
        dependency_scope = self._validate(dependency_scope, DEPENDENCY_SCOPES, "LAB_WORK_DEPENDENCY_SCOPE")
        speculation_class = self._validate(speculation_class, SPECULATION_CLASSES, "LAB_WORK_SPECULATION_CLASS")
        inferred_resource = "TRAINING" if kind.upper() == "TRAINING_RUN" else "GPU" if kind.upper() == "RUN_EXPERIMENT" else "REVIEW" if kind.upper() in {"REVIEW", "CORRECTION"} else "REASONING"
        resource_class = self._validate(resource_class or inferred_resource, RESOURCE_CLASSES, "LAB_WORK_RESOURCE_CLASS")
        speculation_condition = speculation_condition or {}
        if not isinstance(speculation_condition, dict):
            raise GPUError("LAB_WORK_SPECULATION_CONDITION_INVALID", "speculation_condition must be an object")
        if speculation_class == "CONDITIONAL_SPECULATIVE":
            if not branch_id or not speculation_condition.get("condition") or not speculation_condition.get("invalidation_trigger"):
                raise GPUError("LAB_WORK_SPECULATION_CONDITION_REQUIRED", "conditional speculative work requires branch_id, condition, and invalidation_trigger")
        if speculation_class == "EXPENSIVE_SPECULATIVE":
            if not self.feature_flags.get("SPECULATIVE_WORK_POLICY"):
                raise GPUError("LAB_SPECULATIVE_WORK_POLICY_DISABLED", "Set SPECULATIVE_WORK_POLICY=true after staged review")
            if not branch_id or not related_refs or not related_refs.get("brain_approval") or expected_value is None:
                raise GPUError("LAB_EXPENSIVE_SPECULATION_APPROVAL_REQUIRED", "branch_id, related_refs.brain_approval, and expected_value")
        if dormant_until_dependencies and not dependencies:
            raise GPUError("LAB_DORMANT_WORK_DEPENDENCY_REQUIRED", "A dormant conditional branch needs dependencies")
        existing_id: str | None = None
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM research_projects WHERE id=%s", (project_id,))
            if not cur.fetchone():
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            if not created_by or not created_session_id:
                raise GPUError("LAB_WORK_CREATOR_SESSION_REQUIRED", "created_by and created_session_id")
            self._worker(cur, created_by)
            self._session(cur, created_session_id, created_by, project_id)
            if self.feature_flags.get("WORK_PROPOSAL_MODE") and authority_status == "AUTHORITATIVE" and not gate_id:
                cur.execute("SELECT worker_type FROM research_workers WHERE id=%s", (created_by,))
                creator = cur.fetchone()
                if not creator or creator["worker_type"] not in {"WORK_PLANNER", "BRAIN", "SYSTEM"}:
                    raise GPUError("LAB_WORK_PROPOSAL_REQUIRED", "normal workers must use lab_work_propose")
            if canonical_objective_id:
                cur.execute(
                    "SELECT status FROM canonical_objectives WHERE id=%s AND project_id=%s FOR SHARE",
                    (canonical_objective_id, project_id),
                )
                objective = cur.fetchone()
                if not objective:
                    raise GPUError("CANONICAL_OBJECTIVE_NOT_FOUND", canonical_objective_id)
                if objective["status"] != "ACTIVE":
                    raise GPUError("CANONICAL_OBJECTIVE_NOT_ACTIVE", objective["status"])
            elif self.feature_flags.get("CANONICAL_AUTHORITY_V355") and not branch_id:
                root_type = str((related_refs or {}).get("work_root_type", "")).upper()
                if root_type not in {"ENGINEERING_PREREQUISITE", "META_RESEARCH_OBJECTIVE", "RECOVERY_OPERATION"}:
                    raise GPUError("LAB_WORK_CANONICAL_OBJECTIVE_REQUIRED", "provide canonical_objective_id or explicit work_root_type")
            if branch_id:
                cur.execute(
                    "SELECT canonical_objective_id,state FROM hypothesis_branches WHERE id=%s AND project_id=%s FOR SHARE",
                    (branch_id, project_id),
                )
                branch = cur.fetchone()
                if not branch:
                    raise GPUError("HYPOTHESIS_BRANCH_NOT_FOUND", branch_id)
                if branch["state"] not in {"OPEN", "WAITING", "BLOCKED", "SUPPORTED_WITHIN_SCOPE", "WEAKENED"}:
                    raise GPUError("HYPOTHESIS_BRANCH_NOT_ACTIONABLE", str(branch["state"]))
                if canonical_objective_id and str(branch["canonical_objective_id"]) != str(canonical_objective_id):
                    raise GPUError("HYPOTHESIS_BRANCH_OBJECTIVE_MISMATCH", branch_id)
                canonical_objective_id = canonical_objective_id or str(branch["canonical_objective_id"])
                cur.execute("SELECT status FROM canonical_objectives WHERE id=%s FOR SHARE", (canonical_objective_id,))
                objective = cur.fetchone()
                if not objective or objective["status"] != "ACTIVE":
                    raise GPUError("CANONICAL_OBJECTIVE_NOT_ACTIVE", str(objective["status"]) if objective else canonical_objective_id)
            if gate_id:
                cur.execute("SELECT * FROM scientific_gates WHERE id=%s AND project_id=%s FOR UPDATE", (gate_id, project_id))
                gate = cur.fetchone()
                if not gate:
                    raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", gate_id)
                if gate["status"] in {"SUPERSEDED", "INVALID"}:
                    raise GPUError("SCIENTIFIC_GATE_NOT_ACTIONABLE", gate["status"])
                authority_key = authority_key or gate["authority_key"]
                canonical_subject_version = canonical_subject_version or gate["canonical_subject_version"]
                subject_id = subject_id or gate["scientific_object_id"]
            if authority_status == "AUTHORITATIVE":
                if not authority_key or not gate_id or not canonical_subject_version:
                    raise GPUError("LAB_WORK_AUTHORITY_IDENTITY_REQUIRED", "authority_key, gate_id, canonical_subject_version")
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (authority_key,))
                cur.execute(
                    "SELECT id FROM lab_work_items WHERE project_id=%s AND authority_key=%s "
                    "AND authority_status='AUTHORITATIVE' AND status=ANY(%s) FOR UPDATE",
                    (project_id, authority_key, list(ACTIVE_WORK_STATUSES)),
                )
                existing = cur.fetchone()
                if existing:
                    existing_id = str(existing["id"])
                    cur.execute(
                        "UPDATE scientific_gates SET authoritative_work_item_id=COALESCE(authoritative_work_item_id,%s) WHERE id=%s",
                        (existing_id, gate_id),
                    )
                elif equivalence_key is None:
                    equivalence_key = authority_key
            if existing_id:
                self._event(cur, project_id, "WORK_ITEM_AUTHORITY_REUSED", existing_id, {"authority_key": authority_key, "gate_id": gate_id})
                return {**self._work_record(cur, existing_id), "dedupe_result": "REUSED_ACTIVE"}
            if authority_status == "AUTHORITATIVE" and authority_key and canonical_subject_version:
                cur.execute(
                    "SELECT id FROM lab_work_items WHERE project_id=%s AND authority_key=%s "
                    "AND canonical_subject_version=%s AND authority_status='AUTHORITATIVE' "
                    "AND status='COMPLETED' AND invalidated_at IS NULL ORDER BY completed_at DESC LIMIT 1 FOR UPDATE",
                    (project_id, authority_key, canonical_subject_version),
                )
                terminal = cur.fetchone()
                if terminal:
                    existing_id = str(terminal["id"])
                    self._event(cur, project_id, "WORK_ITEM_TERMINAL_REUSED", existing_id, {"authority_key": authority_key, "canonical_subject_version": canonical_subject_version})
                    return {**self._work_record(cur, existing_id), "dedupe_result": "ALREADY_SATISFIED"}
            status = "DORMANT" if dormant_until_dependencies else ("WAITING_DEPENDENCY" if dependencies else "READY")
            try:
                cur.execute(
                    "INSERT INTO lab_work_items(id,project_id,kind,title,description,scientific_role,status,priority,expected_value,estimated_cost,"
                    "created_by,parent_work_item_id,branch_id,related_refs,equivalence_key,authority_key,gate_id,canonical_subject_version,authority_status,subject_id,recovery_policy,canonical_objective_id,speculation_class,speculation_condition,resource_class,dependency_scope,created_at,updated_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (ident, project_id, kind, title, description, scientific_role, status, priority, expected_value, estimated_cost,
                     created_by, parent_work_item_id, branch_id, json.dumps(related_refs or {}), equivalence_key, authority_key, gate_id,
                     canonical_subject_version, authority_status, subject_id, json.dumps(recovery_policy or {}),
                     canonical_objective_id, speculation_class, json.dumps(speculation_condition), resource_class,
                     dependency_scope, now, now),
                )
            except Exception as exc:
                if getattr(exc, "sqlstate", None) == "23505":
                    if authority_status == "AUTHORITATIVE" and authority_key:
                        cur.execute(
                            "SELECT id FROM lab_work_items WHERE project_id=%s AND authority_key=%s "
                            "AND authority_status='AUTHORITATIVE' AND status=ANY(%s)",
                            (project_id, authority_key, list(ACTIVE_WORK_STATUSES)),
                        )
                        existing = cur.fetchone()
                        if existing:
                            existing_id = str(existing["id"])
                            self._event(cur, project_id, "WORK_ITEM_AUTHORITY_REUSED", existing_id, {"authority_key": authority_key, "gate_id": gate_id})
                            return self._work_record(cur, existing_id)
                    raise GPUError("LAB_EQUIVALENT_WORK_ACTIVE", equivalence_key or title) from exc
                raise
            self._replace_dependencies(cur, ident, dependencies, now)
            if gate_id and authority_status == "AUTHORITATIVE":
                cur.execute("UPDATE scientific_gates SET authoritative_work_item_id=%s WHERE id=%s", (ident, gate_id))
            self._event(cur, project_id, "WORK_ITEM_CREATED", ident, {
                "kind": kind, "title": title, "status": status, "gate_id": gate_id,
                "authority_key": authority_key, "authority_status": authority_status,
                "canonical_objective_id": canonical_objective_id, "dependency_scope": dependency_scope,
                "dormant_until_dependencies": dormant_until_dependencies,
            })
        self.resolve_dependencies(project_id)
        return {**self.work_get(ident), "dedupe_result": "CREATED"}

    def _work_record(self, cur, work_item_id: str) -> dict:
        cur.execute("SELECT * FROM lab_work_items WHERE id=%s", (work_item_id,))
        item = cur.fetchone()
        if not item:
            raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
        cur.execute(
            "SELECT target_type,target_id,required_statuses,invalidating_statuses,description,dependency_scope "
            "FROM lab_work_dependencies WHERE work_item_id=%s ORDER BY created_at",
            (work_item_id,),
        )
        result = self._record(item) or {}
        result["dependencies"] = cur.fetchall()
        return result

    @staticmethod
    def _work_not_found_error(cur, work_item_id: str) -> GPUError:
        """Explain an ID-domain mismatch instead of returning a bare UUID."""
        cur.execute("SELECT kind FROM research_objects WHERE id=%s", (work_item_id,))
        object_row = cur.fetchone()
        if object_row:
            return GPUError(
                "LAB_WORK_NOT_FOUND",
                f"Expected a Lab WorkItem ID; {work_item_id} resolves to ResearchObject "
                f"kind {object_row['kind']}",
            )
        return GPUError(
            "LAB_WORK_NOT_FOUND",
            f"Expected a Lab WorkItem ID; no WorkItem exists for {work_item_id}",
        )

    def work_get(self, work_item_id: str) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            return self._work_record(cur, work_item_id)

    def work_list(self, project_id: str, statuses: list[str] | None = None, limit: int = 100) -> list[dict]:
        with self.store._connect() as conn, conn.cursor() as cur:
            sql, args = "SELECT * FROM lab_work_items WHERE project_id=%s", [project_id]
            if statuses:
                normalized = [self._validate(status, WORK_STATUSES, "LAB_WORK_STATUS") for status in statuses]
                sql += " AND status=ANY(%s)"
                args.append(normalized)
            sql += " ORDER BY priority DESC,created_at LIMIT %s"
            args.append(min(max(1, limit), 500))
            cur.execute(sql, args)
            return [self._record(row) for row in cur.fetchall()]

    def gate_list(self, project_id: str, statuses: list[str] | None = None, limit: int = 100) -> list[dict]:
        with self.store._connect() as conn, conn.cursor() as cur:
            sql, args = "SELECT * FROM scientific_gates WHERE project_id=%s", [project_id]
            if statuses:
                normalized = [self._validate(status, GATE_STATUSES, "SCIENTIFIC_GATE_STATUS") for status in statuses]
                sql += " AND status=ANY(%s)"
                args.append(normalized)
            sql += " ORDER BY created_at DESC LIMIT %s"
            args.append(min(max(1, limit), 500))
            cur.execute(sql, args)
            return [self._record(row) or {} for row in cur.fetchall()]

    def _dependency_status(self, cur, project_id: str, dependency: dict) -> tuple[bool, bool, str]:
        """Return (satisfied, invalidated, explanation) for one typed dependency."""
        target_type, target_id = dependency["target_type"], dependency["target_id"]
        if target_type == "WORK_ITEM":
            cur.execute("SELECT status FROM lab_work_items WHERE id=%s AND project_id=%s", (target_id, project_id))
        elif target_type == "SCIENTIFIC_GATE":
            cur.execute("SELECT status FROM scientific_gates WHERE id=%s AND project_id=%s", (target_id, project_id))
        elif target_type in {"RESEARCH_OBJECT", "EXPERIMENT_RUN", "ENGINEERING_RESULT", "RESEARCH_DECISION", "EVIDENCE_UNIT", "ARTIFACT"}:
            kind = {"EXPERIMENT_RUN": "ExperimentRun", "ENGINEERING_RESULT": "EngineeringResult", "RESEARCH_DECISION": "ResearchDecision", "EVIDENCE_UNIT": "EvidenceUnit", "ARTIFACT": "Artifact"}.get(target_type)
            sql = "SELECT status FROM research_objects WHERE id=%s AND project_id=%s"
            args: list[Any] = [target_id, project_id]
            if kind:
                sql += " AND kind=%s"
                args.append(kind)
            cur.execute(sql, args)
        else:
            return False, False, f"unsupported dependency type {target_type}"
        row = cur.fetchone()
        if not row:
            return False, False, f"dependency target {target_id} is unavailable"
        status = row["status"]
        invalidating = dependency["invalidating_statuses"] or []
        if status in {"SUPERSEDED", "INVALIDATED"}:
            return False, True, f"dependency {target_id} is operationally obsolete ({status})"
        if status in invalidating:
            return False, True, f"dependency {target_id} entered invalidating status {status}"
        required = dependency["required_statuses"] or []
        return (not required or status in required), False, f"dependency {target_id} is {status}"

    @staticmethod
    def _linked_experiment_run_ids(related_refs: dict[str, Any]) -> set[str]:
        """Extract explicit run links without guessing from arbitrary text."""
        values: list[Any] = [
            related_refs.get("experiment_run_id"),
            related_refs.get("run_id"),
            *(related_refs.get("experiment_run_ids") or []),
            *(related_refs.get("run_ids") or []),
        ]
        return {str(value) for value in values if isinstance(value, str) and value}

    def _replace_dependencies(
        self, cur, work_item_id: str, dependencies: list[dict], now: datetime
    ) -> None:
        cur.execute("DELETE FROM lab_work_dependencies WHERE work_item_id=%s", (work_item_id,))
        for dependency in dependencies:
            if "target_type" not in dependency or not str(dependency.get("target_type", "")).strip():
                raise GPUError("LAB_DEPENDENCY_TARGET_TYPE_REQUIRED", "Specify target_type explicitly; WorkItem UUIDs are not ResearchObjects")
            target_type = str(dependency["target_type"]).upper()
            target_id = str(dependency.get("target_id", ""))
            if not target_id:
                raise GPUError("LAB_DEPENDENCY_TARGET_REQUIRED", "target_id")
            if target_type not in {"WORK_ITEM", "SCIENTIFIC_GATE", "RESEARCH_OBJECT", "EXPERIMENT_RUN", "ENGINEERING_RESULT", "RESEARCH_DECISION", "EVIDENCE_UNIT", "ARTIFACT"}:
                raise GPUError("LAB_DEPENDENCY_TYPE_INVALID", target_type)
            required_statuses = dependency.get("required_statuses", [])
            exists_only = dependency.get("allow_exists_only", False)
            if not isinstance(required_statuses, list) or any(not isinstance(status, str) or not status.strip() for status in required_statuses):
                raise GPUError("LAB_DEPENDENCY_REQUIRED_STATUS_INVALID", target_id)
            if not required_statuses and exists_only is not True:
                raise GPUError(
                    "LAB_DEPENDENCY_REQUIRED_STATUS_REQUIRED",
                    "Specify required_statuses or explicitly set allow_exists_only=true",
                )
            if exists_only is not True and "allow_exists_only" in dependency and not isinstance(exists_only, bool):
                raise GPUError("LAB_DEPENDENCY_EXISTS_ONLY_INVALID", target_id)
            scope = self._validate(
                str(dependency.get("dependency_scope", "WORKITEM_LOCAL")), DEPENDENCY_SCOPES,
                "LAB_DEPENDENCY_SCOPE",
            )
            cur.execute(
                "INSERT INTO lab_work_dependencies(id,work_item_id,target_type,target_id,required_statuses,invalidating_statuses,description,dependency_scope,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (str(uuid.uuid4()), work_item_id, target_type, target_id, json.dumps(required_statuses),
                 json.dumps(dependency.get("invalidating_statuses", [])), str(dependency.get("description", "")), scope, now),
            )

    def resolve_dependencies(self, project_id: str) -> dict[str, int]:
        """Reconcile dependency-gated work without leaving unsafe READY items claimable."""
        changed = {"ready": 0, "waiting": 0, "invalidated": 0}
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            # A deterministic order establishes which historical duplicate is
            # canonical when dormant equivalent work becomes eligible at once.
            # The partial uniqueness constraint remains the final guard, but
            # reconciliation must never rely on a constraint exception for a
            # normal duplicate-work condition.
            cur.execute(
                "SELECT * FROM lab_work_items WHERE project_id=%s AND status IN "
                "('DORMANT','WAITING_DEPENDENCY','BLOCKED','READY','CLAIMED','RUNNING') "
                "ORDER BY created_at,id FOR UPDATE",
                (project_id,),
            )
            items = cur.fetchall()
            for item in items:
                cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s", (item["id"],))
                dependencies = cur.fetchall()
                if not dependencies:
                    continue
                outcomes = [self._dependency_status(cur, project_id, dependency) for dependency in dependencies]
                invalidations = [detail for _, invalidated, detail in outcomes if invalidated]
                if invalidations and item["status"] in ACTIVE_WORK_STATUSES:
                    cur.execute("UPDATE lab_work_items SET status='INVALIDATED',invalidated_reason=%s,updated_at=%s,completed_at=%s,work_version=work_version+1 WHERE id=%s RETURNING work_version", ("; ".join(invalidations), now, now, item["id"]))
                    version = cur.fetchone()["work_version"]
                    self._event(cur, project_id, "WORK_ITEM_INVALIDATED", item["id"], {"reason": invalidations})
                    self._outbox(cur, project_id, "WORK_ITEM_INVALIDATED", str(item["id"]), f"work-invalidated:{item['id']}:{version}", {"reason": invalidations, "work_version": version})
                    changed["invalidated"] += 1
                    continue
                unsatisfied = [detail for satisfied, _, detail in outcomes if not satisfied]
                if unsatisfied and item["status"] == "READY":
                    reason = "; ".join(unsatisfied)
                    cur.execute("UPDATE lab_work_items SET status='WAITING_DEPENDENCY',blocked_reason=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s RETURNING work_version", (reason, now, item["id"]))
                    version = cur.fetchone()["work_version"]
                    self._event(cur, project_id, "WORK_ITEM_WAITING_DEPENDENCY", item["id"], {"reason": reason})
                    self._outbox(cur, project_id, "WORK_ITEM_WAITING_DEPENDENCY", str(item["id"]), f"work-waiting:{item['id']}:{version}", {"reason": reason, "work_version": version})
                    changed["waiting"] += 1
                    continue
                if all(satisfied for satisfied, _, _ in outcomes) and item["status"] in {"DORMANT", "WAITING_DEPENDENCY", "BLOCKED"}:
                    if item["equivalence_key"]:
                        cur.execute(
                            "SELECT id FROM lab_work_items WHERE project_id=%s AND equivalence_key=%s "
                            "AND id<>%s AND status=ANY(%s) ORDER BY created_at,id LIMIT 1 FOR UPDATE",
                            (project_id, item["equivalence_key"], item["id"], list(EQUIVALENCE_ACTIVE_WORK_STATUSES)),
                        )
                        equivalent = cur.fetchone()
                        if equivalent:
                            reason = (
                                "Equivalent WorkItem became canonical before this duplicate was released: "
                                f"{equivalent['id']}"
                            )
                            cur.execute(
                                "UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',"
                                "superseded_by=%s,invalidated_reason=%s,invalidated_at=%s,updated_at=%s,"
                                "work_version=work_version+1 WHERE id=%s RETURNING work_version",
                                (equivalent["id"], reason, now, now, item["id"]),
                            )
                            version = cur.fetchone()["work_version"]
                            self._event(cur, project_id, "WORK_ITEM_EQUIVALENCE_SUPERSEDED", item["id"], {
                                "canonical_work_item_id": str(equivalent["id"]),
                                "equivalence_key": item["equivalence_key"], "reason": reason,
                            })
                            self._outbox(cur, project_id, "WORK_ITEM_SUPERSEDED", str(item["id"]), f"work-superseded:{item['id']}:{version}", {"successor_id": str(equivalent["id"]), "work_version": version})
                            changed["invalidated"] += 1
                            continue
                    cur.execute("UPDATE lab_work_items SET status='READY',blocked_reason=NULL,updated_at=%s,work_version=work_version+1 WHERE id=%s RETURNING work_version", (now, item["id"]))
                    version = cur.fetchone()["work_version"]
                    self._event(cur, project_id, "WORK_DEPENDENCY_RESOLVED", item["id"], {})
                    self._event(cur, project_id, "WORK_ITEM_READY", item["id"], {})
                    self._outbox(cur, project_id, "WORK_ITEM_READY", str(item["id"]), f"work-ready:{item['id']}:{version}", {"work_version": version})
                    changed["ready"] += 1
        return changed

    def claim_work(self, work_item_id: str, worker_id: str, session_id: str,
                   role: str | None = None, lease_seconds: int | None = None) -> dict:
        """Atomically claim one READY item; concurrent callers cannot both win."""
        now = self._now()
        expiry = now + timedelta(seconds=max(30, lease_seconds or self.lease_seconds))
        lease_id = str(uuid.uuid4())
        dependency_error: tuple[str, str] | None = None
        result: dict | None = None
        with self.store._connect() as conn, conn.cursor() as cur:
            self._worker(cur, worker_id)
            # Budget and portfolio policy use structured branch/resource/speculation
            # fields.  Claiming a partial row would silently bypass newer limits.
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            target = cur.fetchone()
            if not target:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            session = self._session(cur, session_id, worker_id, str(target["project_id"]))
            if str(session["current_project_id"]) != str(target["project_id"]):
                raise GPUError("LAB_PROJECT_MISMATCH", work_item_id)
            cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s", (work_item_id,))
            dependencies = cur.fetchall()
            if dependencies:
                outcomes = [self._dependency_status(cur, str(target["project_id"]), dependency) for dependency in dependencies]
                invalidations = [detail for _, invalidated, detail in outcomes if invalidated]
                unsatisfied = [detail for satisfied, _, detail in outcomes if not satisfied]
                if invalidations:
                    reason = "; ".join(invalidations)
                    cur.execute(
                        "UPDATE lab_work_items SET status='INVALIDATED',invalidated_reason=%s,updated_at=%s,completed_at=%s,work_version=work_version+1 WHERE id=%s AND status='READY'",
                        (reason, now, now, work_item_id),
                    )
                    self._event(cur, target["project_id"], "WORK_ITEM_INVALIDATED", work_item_id, {"reason": invalidations})
                    dependency_error = ("LAB_WORK_DEPENDENCY_INVALIDATED", reason)
                elif unsatisfied:
                    reason = "; ".join(unsatisfied)
                    cur.execute(
                        "UPDATE lab_work_items SET status='WAITING_DEPENDENCY',blocked_reason=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s AND status='READY'",
                        (reason, now, work_item_id),
                    )
                    self._event(cur, target["project_id"], "WORK_ITEM_WAITING_DEPENDENCY", work_item_id, {"reason": reason})
                    dependency_error = ("LAB_WORK_DEPENDENCY_UNSATISFIED", reason)
            if dependency_error is None:
                self._enforce_claim_budget(cur, str(target["project_id"]), target, session_id)
                cur.execute("UPDATE lab_work_items SET status='CLAIMED',assigned_worker_id=%s,assigned_session_id=%s,lease_id=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s AND status='READY' RETURNING *", (worker_id, session_id, lease_id, now, work_item_id))
                item = cur.fetchone()
                if not item:
                    raise GPUError("LAB_WORK_NOT_CLAIMABLE", work_item_id)
                cur.execute("INSERT INTO lab_work_leases(id,work_item_id,worker_id,worker_session_id,acquired_at,heartbeat_at,expires_at,lease_version) VALUES(%s,%s,%s,%s,%s,%s,%s,1)", (lease_id, work_item_id, worker_id, session_id, now, now, expiry))
                cur.execute("UPDATE research_worker_sessions SET current_work_item_id=%s,active_role=%s,status='BUSY',last_heartbeat_at=%s WHERE id=%s", (work_item_id, role or item["scientific_role"], now, session_id))
                cur.execute("UPDATE research_workers SET availability_state='ASSIGNED',availability_updated_at=%s,idle_reason=NULL,idle_since=NULL WHERE id=%s", (now, worker_id))
                self._event(cur, item["project_id"], "WORK_CLAIMED", work_item_id, {"worker_id": worker_id, "session_id": session_id, "lease_id": lease_id, "role": role or item["scientific_role"]})
                result = self._record(item)
                result["lease_id"] = lease_id
                result["lease_expires_at"] = expiry
        if dependency_error is not None:
            raise GPUError(*dependency_error)
        assert result is not None
        return result

    def heartbeat(self, session_id: str, work_item_id: str | None = None, lease_seconds: int | None = None) -> dict:
        now = self._now()
        expiry = now + timedelta(seconds=max(30, lease_seconds or self.lease_seconds))
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM research_worker_sessions WHERE id=%s FOR UPDATE", (session_id,))
            session = cur.fetchone()
            if not session:
                raise GPUError("LAB_SESSION_NOT_FOUND", session_id)
            cur.execute("UPDATE research_worker_sessions SET status='BUSY' WHERE id=%s", (session_id,)) if work_item_id else None
            cur.execute("UPDATE research_worker_sessions SET last_heartbeat_at=%s WHERE id=%s", (now, session_id))
            if work_item_id:
                cur.execute("UPDATE lab_work_leases SET heartbeat_at=%s,expires_at=%s,lease_version=lease_version+1 WHERE work_item_id=%s AND worker_session_id=%s AND released_at IS NULL RETURNING id,lease_version", (now, expiry, work_item_id, session_id))
                lease = cur.fetchone()
                if not lease:
                    raise GPUError("LAB_LEASE_NOT_OWNED", work_item_id)
            return {"session_id": session_id, "work_item_id": work_item_id, "heartbeat_at": now, "expires_at": expiry if work_item_id else None, "lease_version": lease["lease_version"] if work_item_id else None}

    def _require_fresh_work_context(self, cur, item: dict, session_id: str,
                                    expected_work_version: int | None,
                                    expected_lease_version: int | None) -> None:
        if expected_work_version is not None and item["work_version"] != expected_work_version:
            raise GPUError("STALE_WORK_CONTEXT", json.dumps({"work_item_id": str(item["id"]), "expected_work_version": expected_work_version, "current_work_version": item["work_version"]}))
        if expected_lease_version is not None:
            cur.execute("SELECT lease_version FROM lab_work_leases WHERE id=%s AND worker_session_id=%s AND released_at IS NULL", (item["lease_id"], session_id))
            lease = cur.fetchone()
            if not lease or lease["lease_version"] != expected_lease_version:
                raise GPUError("STALE_WORK_CONTEXT", json.dumps({"work_item_id": str(item["id"]), "expected_lease_version": expected_lease_version, "current_lease_version": lease["lease_version"] if lease else None}))

    def start_work(self, work_item_id: str, worker_id: str, session_id: str,
                   expected_work_version: int | None = None,
                   expected_lease_version: int | None = None) -> dict:
        """Mark an owned claim as running without changing scientific state."""
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise self._work_not_found_error(cur, work_item_id)
            self._session(cur, session_id, worker_id, str(item["project_id"]))
            if str(item["assigned_worker_id"]) != str(worker_id) or str(item["assigned_session_id"]) != str(session_id):
                raise GPUError("LAB_WORK_NOT_OWNED", work_item_id)
            self._require_fresh_work_context(cur, item, session_id, expected_work_version, expected_lease_version)
            if item["status"] not in {"CLAIMED", "RUNNING"}:
                raise GPUError("LAB_WORK_NOT_STARTABLE", work_item_id)
            cur.execute("UPDATE lab_work_items SET status='RUNNING',updated_at=%s,work_version=work_version+1 WHERE id=%s", (now, work_item_id))
            cur.execute("UPDATE research_workers SET availability_state='EXECUTING',availability_updated_at=%s,idle_reason=NULL,idle_since=NULL WHERE id=%s", (now, worker_id))
            self._event(cur, item["project_id"], "WORK_STARTED", work_item_id, {"worker_id": worker_id, "session_id": session_id})
        return self.work_get(work_item_id)

    def release_work(self, work_item_id: str, worker_id: str, session_id: str,
                     reason: str = "RELEASED", dependencies: list[dict] | None = None,
                     expected_work_version: int | None = None,
                     expected_lease_version: int | None = None) -> dict:
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            self._session(cur, session_id, worker_id, str(item["project_id"]))
            if str(item["assigned_worker_id"]) != str(worker_id) or str(item["assigned_session_id"]) != str(session_id):
                raise GPUError("LAB_WORK_NOT_OWNED", work_item_id)
            self._require_fresh_work_context(cur, item, session_id, expected_work_version, expected_lease_version)
            waiting_dependency = reason.strip().upper() == "WAITING_DEPENDENCY"
            if dependencies is not None:
                self._replace_dependencies(cur, work_item_id, dependencies, now)
            cur.execute("SELECT 1 FROM lab_work_dependencies WHERE work_item_id=%s LIMIT 1", (work_item_id,))
            has_dependencies = cur.fetchone() is not None
            next_status = "WAITING_DEPENDENCY" if waiting_dependency and has_dependencies else (
                "BLOCKED" if waiting_dependency else "READY"
            )
            blocked_reason = (
                "Waiting dependency requires an explicit dependency record."
                if waiting_dependency and not has_dependencies else None
            )
            cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason=%s,lease_version=lease_version+1 WHERE id=%s AND released_at IS NULL", (now, reason, item["lease_id"]))
            cur.execute("UPDATE lab_work_items SET status=%s,assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,blocked_reason=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s", (next_status, blocked_reason, now, work_item_id))
            cur.execute("UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s WHERE id=%s", (now, session_id))
            if waiting_dependency and self.feature_flags.get("WAITING_WORK_RELEASE"):
                cur.execute("UPDATE research_workers SET availability_state='AVAILABLE',availability_updated_at=%s,idle_reason=NULL,idle_since=NULL WHERE id=%s", (now, worker_id))
            self._event(cur, item["project_id"], "WORK_RELEASED", work_item_id, {"worker_id": worker_id, "reason": reason, "next_status": next_status})
        if waiting_dependency and has_dependencies:
            self.resolve_dependencies(str(item["project_id"]))
        return self.work_get(work_item_id)

    def block_work(self, work_item_id: str, worker_id: str, session_id: str,
                   dependencies: list[dict], reason: str = "WAITING_DEPENDENCY") -> dict:
        """Safely park owned work behind durable, typed prerequisites."""
        if not dependencies:
            raise GPUError("LAB_DEPENDENCY_REQUIRED", work_item_id)
        return self.release_work(work_item_id, worker_id, session_id, reason, dependencies)

    def attach_experiment_run(
        self, work_item_id: str, worker_id: str, session_id: str, run_id: str
    ) -> dict:
        """Detach an owned execution WorkItem onto its canonical ExperimentRun.

        Once attached, the item is intentionally not READY: execution ownership
        is represented by the immutable run, not a renewable chat-worker lease.
        """
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            self._session(cur, session_id, worker_id, str(item["project_id"]))
            if str(item["assigned_worker_id"]) != str(worker_id) or str(item["assigned_session_id"]) != str(session_id):
                raise GPUError("LAB_WORK_NOT_OWNED", work_item_id)
            if item["status"] not in {"CLAIMED", "RUNNING"}:
                raise GPUError("LAB_WORK_NOT_ATTACHABLE", item["status"])
            cur.execute("SELECT status,data FROM research_objects WHERE id=%s AND project_id=%s AND kind='ExperimentRun'", (run_id, item["project_id"]))
            run = cur.fetchone()
            if not run:
                raise GPUError("EXPERIMENT_RUN_NOT_FOUND", run_id)
            if not self.store.experiment_run_is_operationally_active(run):
                raise GPUError("EXPERIMENT_RUN_NOT_ACTIVE", run_id)
            refs = {**(item["related_refs"] or {}), "experiment_run_id": run_id}
            cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='EXPERIMENT_RUN_ATTACHED' WHERE id=%s AND released_at IS NULL", (now, item["lease_id"]))
            cur.execute("UPDATE lab_work_items SET status='RUNNING_DETACHED',related_refs=%s,assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,blocked_reason=%s,updated_at=%s WHERE id=%s", (json.dumps(refs), "Canonical ExperimentRun is executing; this WorkItem cannot be claimed.", now, work_item_id))
            cur.execute("UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s WHERE id=%s", (now, session_id))
            self._event(cur, item["project_id"], "WORK_ITEM_EXPERIMENT_ATTACHED", work_item_id, {"run_id": run_id, "worker_id": worker_id})
        return self.work_get(work_item_id)

    def experiment_run_terminal(self, run_id: str, run_status: str) -> dict[str, int]:
        """Move attached execution WorkItems to result-ready after canonical sync."""
        if run_status not in {"completed", "failed", "cancelled", "TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED"}:
            return {"result_ready": 0}
        now, changed = self._now(), 0
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM lab_work_items WHERE (related_refs->>'experiment_run_id'=%s "
                "OR related_refs->>'run_id'=%s OR related_refs->'experiment_run_ids' ? %s "
                "OR related_refs->'run_ids' ? %s) AND status IN ('CLAIMED','RUNNING','RUNNING_DETACHED','READY') FOR UPDATE",
                (run_id, run_id, run_id, run_id),
            )
            for item in cur.fetchall():
                cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='EXPERIMENT_TERMINAL' WHERE work_item_id=%s AND released_at IS NULL", (now, item["id"]))
                cur.execute("UPDATE lab_work_items SET status='RESULT_READY',assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,blocked_reason=%s,updated_at=%s WHERE id=%s", (f"Canonical ExperimentRun {run_status}; inspect before any follow-up execution.", now, item["id"]))
                self._event(cur, item["project_id"], "WORK_ITEM_RESULT_READY", item["id"], {"run_id": run_id, "run_status": run_status})
                changed += 1
        return {"result_ready": changed}

    def experiment_run_inspected(self, run_id: str) -> dict[str, int]:
        """Close detached execution work once its canonical result is inspected."""
        now, completed = self._now(), 0
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT project_id,status FROM research_objects WHERE id=%s AND kind='ExperimentRun' FOR UPDATE",
                (run_id,),
            )
            run = cur.fetchone()
            if not run:
                raise GPUError("EXPERIMENT_RUN_NOT_FOUND", run_id)
            if run["status"] != "RESULT_INSPECTED":
                return {"completed": 0}
            cur.execute(
                "SELECT * FROM lab_work_items WHERE (related_refs->>'experiment_run_id'=%s "
                "OR related_refs->>'run_id'=%s OR related_refs->'experiment_run_ids' ? %s "
                "OR related_refs->'run_ids' ? %s) AND status='RESULT_READY' FOR UPDATE",
                (run_id, run_id, run_id, run_id),
            )
            for item in cur.fetchall():
                cur.execute(
                    "UPDATE lab_work_items SET status='COMPLETED',blocked_reason=%s,updated_at=%s WHERE id=%s",
                    ("Canonical ExperimentRun inspected; execution work is complete.", now, item["id"]),
                )
                self._event(cur, item["project_id"], "WORK_ITEM_RESULT_INSPECTED_COMPLETED", item["id"], {"run_id": run_id})
                completed += 1
        return {"completed": completed}

    def repair_dependencies(
        self, work_item_id: str, worker_id: str, session_id: str, dependencies: list[dict], rationale: str
    ) -> dict:
        """Repair a legacy unblocked item without pretending it was never READY."""
        if not dependencies:
            raise GPUError("LAB_DEPENDENCY_REQUIRED", work_item_id)
        if not rationale.strip():
            raise GPUError("LAB_DEPENDENCY_REPAIR_RATIONALE_REQUIRED", work_item_id)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            self._session(cur, session_id, worker_id, str(item["project_id"]))
            if item["status"] not in {"READY", "WAITING_DEPENDENCY", "BLOCKED"}:
                raise GPUError("LAB_WORK_NOT_REPAIRABLE", item["status"])
            self._replace_dependencies(cur, work_item_id, dependencies, now)
            cur.execute("UPDATE lab_work_items SET status='WAITING_DEPENDENCY',blocked_reason=%s,updated_at=%s WHERE id=%s", (rationale.strip(), now, work_item_id))
            self._event(cur, item["project_id"], "WORK_ITEM_DEPENDENCIES_REPAIRED", work_item_id, {"rationale": rationale.strip(), "dependency_count": len(dependencies)})
        self.resolve_dependencies(str(item["project_id"]))
        return self.work_get(work_item_id)

    def complete_work(self, work_item_id: str, worker_id: str, session_id: str,
                      summary: str = "", output_object_ids: list[str] | None = None,
                      expected_work_version: int | None = None,
                      expected_lease_version: int | None = None) -> dict:
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            self._session(cur, session_id, worker_id, str(item["project_id"]))
            if str(item["assigned_worker_id"]) != str(worker_id) or str(item["assigned_session_id"]) != str(session_id):
                raise GPUError("LAB_WORK_NOT_OWNED", work_item_id)
            self._require_fresh_work_context(cur, item, session_id, expected_work_version, expected_lease_version)
            cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='COMPLETED',lease_version=lease_version+1 WHERE id=%s AND released_at IS NULL", (now, item["lease_id"]))
            cur.execute("UPDATE lab_work_items SET status='COMPLETED',completed_at=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s", (now, now, work_item_id))
            cur.execute("UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s WHERE id=%s", (now, session_id))
            cur.execute("UPDATE research_workers SET availability_state='AVAILABLE',availability_updated_at=%s,idle_reason=NULL,idle_since=NULL WHERE id=%s", (now, worker_id))
            self._event(cur, item["project_id"], "WORK_COMPLETED", work_item_id, {"worker_id": worker_id, "summary": summary[:4000], "output_object_ids": output_object_ids or []})
        self.resolve_dependencies(str(item["project_id"]))
        return self.work_get(work_item_id)

    def recover_stale_leases(self, project_id: str | None = None) -> dict[str, int]:
        """Reconcile orphaned work without cancelling external execution.

        Historical failures can leave a RUNNING/CLAIMED WorkItem with no lease
        at all.  A live session is required for such an item to remain owned;
        otherwise it is recovered just like an expired lease.
        """
        now, recovered, projection_repairs = self._now(), 0, 0
        changed_projects: set[str] = set()
        with self.store._connect() as conn, conn.cursor() as cur:
            sql = "SELECT l.*,w.project_id,w.kind,w.related_refs FROM lab_work_leases l JOIN lab_work_items w ON w.id=l.work_item_id WHERE l.released_at IS NULL AND l.expires_at<%s"
            args: list[Any] = [now]
            if project_id:
                sql += " AND w.project_id=%s"
                args.append(project_id)
            sql += " FOR UPDATE OF w SKIP LOCKED"
            cur.execute(sql, args)
            for lease in cur.fetchall():
                cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='LEASE_EXPIRED' WHERE id=%s", (now, lease["id"]))
                cur.execute("UPDATE research_worker_sessions SET status='EXPIRED',disconnected_at=%s,current_work_item_id=NULL,active_role=NULL WHERE id=%s AND last_heartbeat_at<%s", (now, lease["worker_session_id"], lease["expires_at"]))
                links = self._linked_experiment_run_ids(lease["related_refs"] or {})
                detached = False
                for run_id in links:
                    cur.execute("SELECT status,data FROM research_objects WHERE id=%s AND project_id=%s AND kind='ExperimentRun'", (run_id, lease["project_id"]))
                    run = cur.fetchone()
                    if run and self.store.experiment_run_is_operationally_active(run):
                        detached = True
                        break
                next_status = "RUNNING_DETACHED" if detached else "READY"
                # The lease expiry never cancels external execution.  A linked,
                # still-active run remains visible as detached instead of being
                # silently offered to another worker.
                cur.execute("UPDATE lab_work_items SET status=%s,assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,updated_at=%s,blocked_reason=%s WHERE id=%s AND status IN ('CLAIMED','RUNNING')", (next_status, now, "Worker lease expired; linked execution continues detached." if detached else "Previous worker lease expired.", lease["work_item_id"]))
                self._event(cur, lease["project_id"], "WORK_RELEASED", lease["work_item_id"], {"reason": "LEASE_EXPIRED", "worker_id": str(lease["worker_id"]), "next_status": next_status})
                self._event(cur, lease["project_id"], "WORKER_DISCONNECTED", None, {"worker_id": str(lease["worker_id"]), "session_id": str(lease["worker_session_id"])})
                recovered += 1
                changed_projects.add(str(lease["project_id"]))

            # A renewable lease is the ownership authority.  Repair legacy or
            # partially-written WorkItem/session projections from that source
            # before any dashboard or worker state is returned.
            sql = (
                "SELECT l.id AS live_lease_id,l.work_item_id,l.worker_id,l.worker_session_id,"
                "w.project_id,w.status,w.assigned_worker_id,w.assigned_session_id,w.lease_id "
                "FROM lab_work_leases l JOIN lab_work_items w ON w.id=l.work_item_id "
                "WHERE l.released_at IS NULL AND l.expires_at>=%s"
            )
            args = [now]
            if project_id:
                sql += " AND w.project_id=%s"
                args.append(project_id)
            sql += " FOR UPDATE OF w SKIP LOCKED"
            cur.execute(sql, args)
            for lease in cur.fetchall():
                projection_matches = (
                    str(lease["assigned_worker_id"] or "") == str(lease["worker_id"])
                    and str(lease["assigned_session_id"] or "") == str(lease["worker_session_id"])
                    and str(lease["lease_id"] or "") == str(lease["live_lease_id"])
                    and lease["status"] in {"CLAIMED", "RUNNING"}
                )
                if projection_matches:
                    continue
                status = lease["status"] if lease["status"] in {"CLAIMED", "RUNNING"} else "CLAIMED"
                cur.execute(
                    "UPDATE lab_work_items SET status=%s,assigned_worker_id=%s,assigned_session_id=%s,"
                    "lease_id=%s,updated_at=%s WHERE id=%s",
                    (status, lease["worker_id"], lease["worker_session_id"], lease["live_lease_id"], now, lease["work_item_id"]),
                )
                cur.execute(
                    "UPDATE research_worker_sessions SET status='BUSY',current_work_item_id=%s "
                    "WHERE id=%s AND status NOT IN ('DISCONNECTED','EXPIRED')",
                    (lease["work_item_id"], lease["worker_session_id"]),
                )
                cur.execute(
                    "UPDATE research_worker_sessions SET status=CASE WHEN status='BUSY' THEN 'ACTIVE' ELSE status END,"
                    "current_work_item_id=NULL WHERE current_work_item_id=%s AND id<>%s",
                    (lease["work_item_id"], lease["worker_session_id"]),
                )
                self._event(cur, lease["project_id"], "LEASE_PROJECTION_REPAIRED", lease["work_item_id"], {
                    "lease_id": str(lease["live_lease_id"]), "worker_id": str(lease["worker_id"]),
                    "session_id": str(lease["worker_session_id"]),
                })
                projection_repairs += 1
                changed_projects.add(str(lease["project_id"]))

            # Do not trust a lease to be present.  This scan is exclusively
            # for old/manual rows and partial failures where the lease write
            # was lost after the WorkItem status committed.  A live lease is
            # the ownership authority and was reconciled above, so an ACTIVE
            # (or temporarily stale-projected) session can never cause its
            # unexpired lease to be orphan-reclaimed here.
            threshold = now - timedelta(seconds=self.lease_seconds)
            sql = (
                "SELECT w.*,s.status AS session_status,s.last_heartbeat_at "
                "FROM lab_work_items w LEFT JOIN research_worker_sessions s "
                "ON s.id=w.assigned_session_id WHERE w.status IN ('CLAIMED','RUNNING')"
                " AND NOT EXISTS (SELECT 1 FROM lab_work_leases l "
                "WHERE l.work_item_id=w.id AND l.released_at IS NULL AND l.expires_at>=%s)"
            )
            args: list[Any] = [now]
            if project_id:
                sql += " AND w.project_id=%s"
                args.append(project_id)
            # Lock only the WorkItem side. PostgreSQL rejects a generic FOR
            # UPDATE on the nullable side of this LEFT JOIN.
            sql += " FOR UPDATE OF w SKIP LOCKED"
            cur.execute(sql, args)
            for item in cur.fetchall():
                session_live = (
                    item["assigned_session_id"] is not None
                    and item["session_status"] not in {None, "DISCONNECTED", "EXPIRED"}
                    and item["last_heartbeat_at"] >= threshold
                )
                if session_live:
                    continue
                links = self._linked_experiment_run_ids(item["related_refs"] or {})
                detached = False
                for run_id in links:
                    cur.execute(
                        "SELECT status,data FROM research_objects WHERE id=%s AND project_id=%s "
                        "AND kind='ExperimentRun'",
                        (run_id, item["project_id"]),
                    )
                    run = cur.fetchone()
                    if run and self.store.experiment_run_is_operationally_active(run):
                        detached = True
                        break
                next_status = "RUNNING_DETACHED" if detached else "READY"
                reason = (
                    "Worker ownership is orphaned; linked execution continues detached."
                    if detached
                    else "Worker ownership is orphaned; no heartbeat-live session remains."
                )
                cur.execute(
                    "UPDATE lab_work_leases SET released_at=%s,release_reason='ORPHANED_SESSION' "
                    "WHERE work_item_id=%s AND released_at IS NULL",
                    (now, item["id"]),
                )
                cur.execute(
                    "UPDATE lab_work_items SET status=%s,assigned_worker_id=NULL,assigned_session_id=NULL,"
                    "lease_id=NULL,updated_at=%s,blocked_reason=%s WHERE id=%s",
                    (next_status, now, reason, item["id"]),
                )
                self._event(
                    cur,
                    item["project_id"],
                    "WORK_ORPHAN_RECONCILED",
                    item["id"],
                    {"reason": reason, "next_status": next_status},
                )
                recovered += 1
                changed_projects.add(str(item["project_id"]))
        for changed_project in changed_projects:
            self.resolve_dependencies(changed_project)
        return {"recovered": recovered, "projection_repairs": projection_repairs}

    def message_send(self, project_id: str, from_worker_id: str, from_session_id: str, message_type: str,
                     subject: str, body: str, to_worker_id: str | None = None,
                     to_role: str | None = None, reference_ids: list[str] | None = None,
                     priority: int = 0, broadcast_scope: str | None = None) -> dict:
        message_type = self._validate(message_type, MESSAGE_TYPES, "LAB_MESSAGE_TYPE")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            self._worker(cur, from_worker_id)
            self._session(cur, from_session_id, from_worker_id, project_id)
            if message_type == "SHARE_FINDING":
                # During DDE v3.3 independent generation, a candidate rationale
                # sent as a LabMessage would bypass the data-access isolation.
                # Operational messages and post-freeze synthesis remain allowed.
                try:
                    cur.execute(
                        "SELECT 1 FROM discovery_round_memberships m JOIN research_objects r "
                        "ON r.id=m.discovery_round_id WHERE m.worker_session_id=%s "
                        "AND r.project_id=%s AND r.kind='DiscoveryRound' AND r.status='ACTIVE' "
                        "AND m.independent_generation=TRUE "
                        "AND r.data->>'phase'='INDEPENDENT_GENERATION' LIMIT 1",
                        (from_session_id, project_id),
                    )
                    if cur.fetchone():
                        raise GPUError(
                            "DISCOVERY_PEER_MESSAGE_BLOCKED",
                            "Current-round candidate sharing is blocked until every batch is frozen.",
                        )
                except psycopg.errors.UndefinedTable:
                    # DDE has not been initialized on this legacy database yet.
                    conn.rollback()
                    cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_lab_worker_migration'))")
            if to_worker_id:
                self._worker(cur, to_worker_id)
                cur.execute("SELECT 1 FROM research_worker_sessions WHERE worker_id=%s AND current_project_id=%s AND status NOT IN ('DISCONNECTED','EXPIRED')", (to_worker_id, project_id))
                if not cur.fetchone():
                    raise GPUError("LAB_MESSAGE_RECIPIENT_NOT_IN_PROJECT", to_worker_id)
            cur.execute("INSERT INTO lab_messages(id,project_id,from_worker_id,to_worker_id,to_role,broadcast_scope,message_type,subject,body,reference_ids,priority,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (ident, project_id, from_worker_id, to_worker_id, to_role, broadcast_scope, message_type, subject[:500], body[:8000], json.dumps(reference_ids or []), priority, now))
            # A message is advisory only. It never updates an evidence or scientific object.
            self._event(cur, project_id, "LAB_MESSAGE_SENT", None, {"message_id": ident, "from_worker_id": from_worker_id, "to_worker_id": to_worker_id, "message_type": message_type, "reference_ids": reference_ids or []})
        return {"id": ident, "project_id": project_id, "message_type": message_type, "created_at": now}

    def message_list(self, project_id: str, worker_id: str, role: str | None = None,
                     unread_only: bool = False, limit: int = 100) -> list[dict]:
        with self.store._connect() as conn, conn.cursor() as cur:
            sql = "SELECT * FROM lab_messages WHERE project_id=%s AND (to_worker_id IS NULL OR to_worker_id=%s"
            args: list[Any] = [project_id, worker_id]
            if role:
                sql += " OR to_role=%s"
                args.append(role)
            sql += ")"
            if unread_only:
                sql += " AND read_at IS NULL"
            sql += " ORDER BY priority DESC,created_at DESC LIMIT %s"
            args.append(min(max(1, limit), 500))
            cur.execute(sql, args)
            return [self._record(row) for row in cur.fetchall()]

    def message_mark_read(self, project_id: str, worker_id: str, session_id: str,
                          message_ids: list[str]) -> dict[str, int]:
        """Acknowledge only messages that are visible to this active project session."""
        if not message_ids:
            return {"marked_read": 0}
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            session = self._session(cur, session_id, worker_id, project_id)
            cur.execute("UPDATE lab_messages SET read_at=%s WHERE project_id=%s AND id=ANY(%s) AND read_at IS NULL AND (to_worker_id IS NULL OR to_worker_id=%s OR to_role=%s)", (now, project_id, message_ids, worker_id, session["active_role"]))
            return {"marked_read": cur.rowcount}

    def _active_workers(self, project_id: str) -> list[dict]:
        threshold = self._now() - timedelta(seconds=self.lease_seconds)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT s.id AS session_id,s.worker_id,w.display_name,"
                "CASE WHEN EXISTS(SELECT 1 FROM lab_work_leases l WHERE l.worker_session_id=s.id "
                "AND l.released_at IS NULL AND l.expires_at>=%s) THEN 'BUSY' "
                "WHEN s.status='BUSY' THEN 'ACTIVE' ELSE s.status END AS status,"
                "s.active_role,s.current_work_item_id,s.last_heartbeat_at "
                "FROM research_worker_sessions s JOIN research_workers w ON w.id=s.worker_id "
                "WHERE s.current_project_id=%s AND s.status NOT IN ('DISCONNECTED','EXPIRED') "
                "AND s.last_heartbeat_at>=%s ORDER BY s.last_heartbeat_at DESC",
                (self._now(), project_id, threshold),
            )
            return [self._record(row) for row in cur.fetchall()]

    def state_get(
        self,
        project_id: str,
        session_id: str | None = None,
        since: str | None = None,
        *,
        reconcile: bool = True,
    ) -> dict:
        if reconcile:
            self.recover_stale_leases(project_id)
            self.resolve_dependencies(project_id)
        state = self.store.lab_state_summary(project_id)
        events = self.store.events_summary(project_id, 30)
        if since:
            events = [event for event in events if event["created_at"].isoformat() > since]
        worker_id = None
        role = None
        if session_id:
            with self.store._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT worker_id,active_role FROM research_worker_sessions WHERE id=%s AND current_project_id=%s", (session_id, project_id))
                session = cur.fetchone()
                if session:
                    worker_id, role = str(session["worker_id"]), session["active_role"]
        return {
            "project": state["project"],
            "research_state_version": state["research_state_version"],
            "world_model_version": state["world_model_version"],
            "research_policy_version": state["research_policy_version"],
            "brain_policy_version": state["brain_policy_version"],
            "active_workers": self._active_workers(project_id),
            "scientific_gates": self.gate_list(project_id, ["PENDING", "AWAITING_SEMANTIC_REVIEW", "PREFLIGHT_FAILED"], 30),
            "ready_work_items": self.work_list(project_id, ["READY"], 30),
            "blocked_work_items": self.work_list(project_id, ["BLOCKED", "WAITING_DEPENDENCY"], 30),
            "running_work_items": self.work_list(project_id, ["CLAIMED", "RUNNING", "RUNNING_DETACHED"], 30),
            "result_ready_work_items": self.work_list(project_id, ["RESULT_READY"], 30),
            "recent_events": [{**event, "subject_id": str(event["subject_id"]) if event["subject_id"] else None} for event in events],
            "unread_messages": self.message_list(project_id, worker_id, role, True, 30) if worker_id else [],
            "gpu_activity": state["active_experiments"],
            "lab_budget": self._lab_budget_get(project_id),
        }

    def sync(self, session_id: str, project_id: str, since: str | None = None,
             current_work_item_id: str | None = None,
             expected_work_version: int | None = None) -> dict:
        self.recover_stale_leases(project_id)
        lease_state = "IDLE"
        lease_detail: dict[str, Any] = {}
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT worker_id,current_work_item_id,status FROM research_worker_sessions WHERE id=%s AND current_project_id=%s FOR UPDATE", (session_id, project_id))
            session = cur.fetchone()
            if not session:
                raise GPUError("LAB_SESSION_NOT_FOUND", session_id)
            now = self._now()
            if current_work_item_id:
                cur.execute("SELECT id,status,assigned_session_id,work_version,subject_id FROM lab_work_items WHERE id=%s AND project_id=%s", (current_work_item_id, project_id))
                item = cur.fetchone()
                if not item or str(item["assigned_session_id"] or "") != str(session_id):
                    lease_state = "LEASE_LOST"
                    lease_detail = {"work_item_id": current_work_item_id, "reason": "work is no longer owned by this session"}
                elif item["status"] not in {"CLAIMED", "RUNNING"}:
                    lease_state = "LEASE_LOST"
                    lease_detail = {"work_item_id": current_work_item_id, "reason": f"work status is {item['status']}"}
                elif expected_work_version is not None and item["work_version"] != expected_work_version:
                    lease_state = "STALE_WORK_CONTEXT"
                    lease_detail = {
                        "work_item_id": current_work_item_id, "expected_work_version": expected_work_version,
                        "current_work_version": item["work_version"], "subject_id": item["subject_id"],
                    }
                else:
                    lease_state = "OWNED"
            if session["status"] not in {"DISCONNECTED", "EXPIRED"}:
                cur.execute("UPDATE research_worker_sessions SET last_heartbeat_at=%s WHERE id=%s", (now, session_id))
        # Sync has already performed ownership reconciliation. Do not repeat
        # recovery/dependency scans inside state_get for every poll.
        self.resolve_dependencies(project_id)
        state = self.state_get(project_id, session_id, since, reconcile=False)
        old_work_reassigned = bool(current_work_item_id and current_work_item_id != str(session["current_work_item_id"] or ""))
        canonical_context = self._canonical_worker_context(project_id, session_id, state, since)
        if lease_state == "OWNED":
            sync_status = "ASSIGNED"
        elif lease_state == "LEASE_LOST":
            old = self.work_get(current_work_item_id) if current_work_item_id else None
            sync_status = "WORK_SUPERSEDED" if old and old["status"] == "SUPERSEDED" else "WORK_COMPLETED_ELSEWHERE" if old and old["status"] == "COMPLETED" else "LEASE_LOST"
        elif canonical_context["work"] and canonical_context["work"]["status"] in {"WAITING_DEPENDENCY", "BLOCKED", "DORMANT"}:
            sync_status = "WAITING"
        else:
            sync_status = "NO_WORK"
        compact = self.feature_flags.get("CANONICAL_SYNC_CONTEXT", False)
        return {
            "session_id": session_id, "old_work_reassigned": old_work_reassigned,
            "sync_status": sync_status, "lease_state": lease_state, "lease_detail": lease_detail,
            "lab_state": canonical_context if compact else state,
            "canonical_worker_context": canonical_context,
        }

    def _canonical_worker_context(self, project_id: str, session_id: str, state: dict,
                                  since: str | None = None) -> dict[str, Any]:
        """Compact coordinator projection; only IDs, versions, and actionable state are returned."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT worker_id,current_work_item_id,status FROM research_worker_sessions WHERE id=%s AND current_project_id=%s", (session_id, project_id))
            session = cur.fetchone()
            if not session:
                raise GPUError("LAB_SESSION_NOT_FOUND", session_id)
            item = None
            lease = None
            if session["current_work_item_id"]:
                cur.execute("SELECT * FROM lab_work_items WHERE id=%s AND project_id=%s", (session["current_work_item_id"], project_id))
                item = cur.fetchone()
                if item:
                    cur.execute("SELECT id,worker_id,expires_at,lease_version FROM lab_work_leases WHERE id=%s AND released_at IS NULL", (item["lease_id"],))
                    lease = cur.fetchone()
            objective = None
            objective_id = item["canonical_objective_id"] if item else None
            if objective_id:
                cur.execute("SELECT id,version,title,current_goal,status FROM canonical_objectives WHERE id=%s", (objective_id,))
                objective = cur.fetchone()
            if not objective:
                cur.execute("SELECT id,version,title,current_goal,status FROM canonical_objectives WHERE project_id=%s AND status='ACTIVE' ORDER BY priority DESC,updated_at DESC LIMIT 1", (project_id,))
                objective = cur.fetchone()
            unresolved: list[dict[str, Any]] = []
            if item:
                cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s ORDER BY created_at", (item["id"],))
                for dependency in cur.fetchall():
                    satisfied, invalidated, detail = self._dependency_status(cur, project_id, dependency)
                    if not satisfied:
                        unresolved.append({"target_type": dependency["target_type"], "target_id": dependency["target_id"], "scope": dependency["dependency_scope"], "invalidated": invalidated, "detail": detail})
            events = state.get("recent_events", [])
            if since:
                events = [event for event in events if event["created_at"].isoformat() > since]
            return {
                "context_version": "v3.5.5",
                "generated_at": self._now(),
                "worker": {"worker_id": str(session["worker_id"]), "session_id": session_id, "status": session["status"]},
                "project": {"project_id": project_id, "project_state_version": state["research_state_version"]},
                "canonical_objective": self._record(objective),
                "work": None if not item else {
                    "work_item_id": str(item["id"]), "work_item_version": item["work_version"],
                    "authority_key": item["authority_key"], "subject_id": item["subject_id"],
                    "subject_version": item["canonical_subject_version"], "status": item["status"],
                    "role": item["scientific_role"], "mode": item["kind"],
                },
                "lease": None if not lease else {"lease_id": str(lease["id"]), "version": lease["lease_version"], "owner": str(lease["worker_id"]), "expires_at": lease["expires_at"]},
                "dependencies": {"unresolved": unresolved, "newly_resolved": []},
                "invalidations": [event for event in events if "INVALIDATED" in event["event_type"] or "SUPERSEDED" in event["event_type"]],
                "relevant_changes_since_last_sync": events,
                "critical_messages": state.get("unread_messages", []),
            }

    def v355_production_audit(self, project_id: str) -> dict[str, Any]:
        """Read-only v3.5.5 consistency audit; never changes lab or science state."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE project_id=%s ORDER BY created_at,id", (project_id,))
            items = cur.fetchall()
            active = [item for item in items if item["status"] in ACTIVE_WORK_STATUSES]
            by_equivalence: dict[str, list[str]] = {}
            by_authority: dict[tuple[str, str | None], list[str]] = {}
            waiting_satisfied: list[str] = []
            for item in active:
                if item["equivalence_key"]:
                    by_equivalence.setdefault(item["equivalence_key"], []).append(str(item["id"]))
                if item["authority_key"]:
                    by_authority.setdefault((item["authority_key"], item["canonical_subject_version"]), []).append(str(item["id"]))
                if item["status"] == "WAITING_DEPENDENCY":
                    cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s", (item["id"],))
                    dependencies = cur.fetchall()
                    if dependencies and all(self._dependency_status(cur, project_id, dep)[0] for dep in dependencies):
                        waiting_satisfied.append(str(item["id"]))
            cur.execute("SELECT id,status,data FROM research_objects WHERE project_id=%s AND kind='Experiment'", (project_id,))
            experiments = cur.fetchall()
            text_state_divergence = [str(item["id"]) for item in active if "supersed" in (item["title"] + " " + item["description"]).lower()]
            cur.execute("SELECT * FROM lab_consistency_conflicts WHERE project_id=%s AND status='OPEN' ORDER BY detected_at", (project_id,))
            conflicts = [self._record(row) for row in cur.fetchall()]
            return {
                "project_id": project_id,
                "active_work_count": len(active),
                "duplicate_active_equivalence": {key: ids for key, ids in by_equivalence.items() if len(ids) > 1},
                "duplicate_active_authority": {"|".join(key): ids for key, ids in by_authority.items() if len(ids) > 1},
                "active_work_without_objective": [str(item["id"]) for item in active if item["canonical_objective_id"] is None and not item["parent_work_item_id"] and not item["recovery_policy"]],
                "active_work_without_authority": [str(item["id"]) for item in active if not item["authority_key"]],
                "active_work_without_subject_version": [str(item["id"]) for item in active if not item["canonical_subject_version"]],
                "waiting_with_satisfied_dependencies": waiting_satisfied,
                "experiments_without_explicit_version": [str(row["id"]) for row in experiments if not row["data"].get("version")],
                "text_state_divergence": text_state_divergence,
                "open_consistency_conflicts": conflicts,
            }

    def v355_shadow_projection(self, project_id: str) -> dict[str, Any]:
        """Read-only canonical projection for staged UI/scheduler rollout."""
        audit = self.v355_production_audit(project_id)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM canonical_objectives WHERE project_id=%s AND status='ACTIVE' ORDER BY priority DESC,updated_at DESC", (project_id,))
            objectives = [self._record(row) for row in cur.fetchall()]
            cur.execute("SELECT * FROM hypothesis_branches WHERE project_id=%s AND state IN ('OPEN','WAITING','BLOCKED') ORDER BY priority DESC,created_at", (project_id,))
            branches = [self._record(row) for row in cur.fetchall()]
            cur.execute("SELECT * FROM lab_work_items WHERE project_id=%s AND status=ANY(%s) AND authority_status='AUTHORITATIVE' ORDER BY priority DESC,created_at", (project_id, list(ACTIVE_WORK_STATUSES)))
            canonical_work = [self._record(row) for row in cur.fetchall()]
        return {"projection_version": "v3.5.5-shadow", "objectives": objectives, "active_branches": branches, "canonical_work_items": canonical_work, "consistency": audit}

    def outbox_list(self, project_id: str, pending_only: bool = True, limit: int = 100) -> list[dict]:
        """Inspect durable coordination events without replaying them or changing science state."""
        with self.store._connect() as conn, conn.cursor() as cur:
            sql = "SELECT * FROM lab_transactional_outbox WHERE project_id=%s"
            args: list[Any] = [project_id]
            if pending_only:
                sql += " AND delivered_at IS NULL"
            sql += " ORDER BY created_at,id LIMIT %s"
            args.append(min(max(1, limit), 500))
            cur.execute(sql, args)
            return [self._record(row) or {} for row in cur.fetchall()]

    def outbox_pending(self, limit: int = 100) -> list[dict]:
        """Return global pending operational events for restart-safe dispatcher recovery."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_transactional_outbox WHERE delivered_at IS NULL ORDER BY created_at,id LIMIT %s", (min(max(1, limit), 500),))
            return [self._record(row) or {} for row in cur.fetchall()]

    def outbox_mark_delivered(self, outbox_id: str) -> dict:
        """Idempotently acknowledge a successfully applied coordination projection."""
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE lab_transactional_outbox SET delivered_at=COALESCE(delivered_at,%s) WHERE id=%s RETURNING *", (now, outbox_id))
            row = cur.fetchone()
            if not row:
                raise GPUError("LAB_OUTBOX_EVENT_NOT_FOUND", outbox_id)
            return self._record(row) or {}

    def canonical_objective_get(self, objective_id: str) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM canonical_objectives WHERE id=%s", (objective_id,))
            row = cur.fetchone()
            if not row:
                raise GPUError("CANONICAL_OBJECTIVE_NOT_FOUND", objective_id)
            return self._record(row) or {}

    def hypothesis_branch_get(self, branch_id: str) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM hypothesis_branches WHERE id=%s", (branch_id,))
            row = cur.fetchone()
            if not row:
                raise GPUError("HYPOTHESIS_BRANCH_NOT_FOUND", branch_id)
            return self._record(row) or {}

    def hypothesis_portfolio_ensure(self, project_id: str, canonical_objective_id: str,
                                    search_regime: str = "EXPLORE", branch_budget: int | None = None,
                                    scientific_concurrency_budget: int | None = None,
                                    gpu_concurrency_budget: int | None = None,
                                    training_concurrency_budget: int | None = None,
                                    reasoning_concurrency_budget: int | None = None) -> dict:
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM canonical_objectives WHERE id=%s AND project_id=%s FOR UPDATE", (canonical_objective_id, project_id))
            objective = cur.fetchone()
            if not objective:
                raise GPUError("CANONICAL_OBJECTIVE_NOT_FOUND", canonical_objective_id)
            cur.execute("SELECT * FROM hypothesis_portfolios WHERE project_id=%s AND canonical_objective_id=%s AND objective_version=%s", (project_id, canonical_objective_id, objective["version"]))
            existing = cur.fetchone()
            if existing:
                return self._record(existing) or {}
            cur.execute("SELECT id FROM hypothesis_branches WHERE canonical_objective_id=%s AND state IN ('OPEN','WAITING','BLOCKED') ORDER BY priority DESC,created_at", (canonical_objective_id,))
            branches = [str(row["id"]) for row in cur.fetchall()]
            ident = str(uuid.uuid4())
            cur.execute("INSERT INTO hypothesis_portfolios(id,project_id,canonical_objective_id,objective_version,active_branch_ids,branch_budget,scientific_concurrency_budget,gpu_concurrency_budget,training_concurrency_budget,reasoning_concurrency_budget,search_regime,status,created_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'ACTIVE',%s,%s)", (ident, project_id, canonical_objective_id, objective["version"], json.dumps(branches), branch_budget, scientific_concurrency_budget, gpu_concurrency_budget, training_concurrency_budget, reasoning_concurrency_budget, search_regime.upper(), now, now))
            cur.execute("SELECT * FROM hypothesis_portfolios WHERE id=%s", (ident,))
            return self._record(cur.fetchone()) or {}

    def branch_coverage_get(self, project_id: str) -> list[dict]:
        """Return structured scheduler-facing coverage without creating artificial work."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM hypothesis_branches WHERE project_id=%s ORDER BY priority DESC,created_at", (project_id,))
            branches = cur.fetchall()
            result: list[dict] = []
            for branch in branches:
                cur.execute(
                    "SELECT w.*,l.worker_id AS lease_worker_id FROM lab_work_items w "
                    "LEFT JOIN lab_work_leases l ON l.id=w.lease_id AND l.released_at IS NULL "
                    "WHERE w.branch_id=%s ORDER BY w.priority DESC,w.created_at",
                    (branch["id"],),
                )
                work = cur.fetchall()
                active = [item for item in work if item["status"] in ACTIVE_WORK_STATUSES]
                executing = [item for item in work if item["status"] in {"CLAIMED", "RUNNING", "RUNNING_DETACHED", "RESULT_READY"}]
                waiting = [item for item in work if item["status"] in {"WAITING_DEPENDENCY", "BLOCKED", "DORMANT"}]
                completed = [item for item in work if item["status"] == "COMPLETED"]
                cur.execute(
                    "SELECT d.* FROM lab_work_dependencies d JOIN lab_work_items w ON w.id=d.work_item_id "
                    "WHERE w.branch_id=%s AND w.status IN ('WAITING_DEPENDENCY','BLOCKED','DORMANT') ORDER BY d.created_at",
                    (branch["id"],),
                )
                unresolved_dependencies = [self._record(item) or {} for item in cur.fetchall()]
                terminal = branch["state"] in {"RESOLVED", "REFUTED", "ABANDONED", "SUPERSEDED"}
                if terminal:
                    next_actionability, saturation = "RESOLVED", "RESOLVED"
                elif waiting and not any(item["status"] == "READY" for item in active):
                    next_actionability, saturation = "WAITING_DEPENDENCY", "BLOCKED_LOCAL"
                elif any(item["status"] == "READY" for item in active):
                    next_actionability, saturation = "READY_EXISTING_WORK", "COVERED"
                elif executing:
                    next_actionability, saturation = "EXECUTION_IN_PROGRESS", "COVERED"
                elif branch["state"] == "OPEN":
                    next_actionability, saturation = "PLANNER_EVALUATION_REQUIRED", "UNCOVERED"
                else:
                    next_actionability, saturation = "NOT_ACTIONABLE", "BLOCKED_LOCAL"
                result.append({
                    "branch_id": str(branch["id"]), "status": branch["state"], "canonical_objective_id": str(branch["canonical_objective_id"]),
                    "hypothesis_ids": branch["hypothesis_ids"], "scientific_distance": branch["scientific_distance"],
                    "branch_dependencies": branch["branch_dependencies"], "dependency_scope": branch["branch_blocking_scope"],
                    "active_work_count": len(active), "active_work_item_ids": [str(item["id"]) for item in active],
                    "executing_work_item_ids": [str(item["id"]) for item in executing],
                    "waiting_work_item_ids": [str(item["id"]) for item in waiting],
                    "completed_work_item_ids": [str(item["id"]) for item in completed],
                    "active_worker_count": len({str(item["lease_worker_id"]) for item in executing if item["lease_worker_id"]}),
                    "current_owner": str(executing[0]["lease_worker_id"]) if executing and executing[0]["lease_worker_id"] else None,
                    "waiting_dependency": bool(waiting), "unresolved_dependencies": unresolved_dependencies,
                    "current_gate_ids": [str(item["gate_id"]) for item in active if item["gate_id"]],
                    "unresolved_question": branch["question_id"], "scientific_niche": branch["mechanistic_niche_id"],
                    "next_actionability": next_actionability, "branch_saturation_state": saturation,
                    "last_scientific_progress_at": max((item["updated_at"] for item in work), default=branch["created_at"]),
                    "last_progress_at": max((item["updated_at"] for item in work), default=branch["created_at"]),
                })
            return result

    def agenda_coverage_get(self, project_id: str) -> list[dict]:
        """Read-only v3.6 agenda coverage; branch-local waits never become project-global here."""
        coverage = self.branch_coverage_get(project_id)
        by_objective: dict[str, list[dict]] = {}
        for branch in coverage:
            by_objective.setdefault(branch["canonical_objective_id"], []).append(branch)
        result = []
        for objective_id, branches in by_objective.items():
            def state(branch: dict) -> str:
                if branch["status"] in {"RESOLVED", "REFUTED", "ABANDONED", "SUPERSEDED"}:
                    return "RESOLVED"
                if branch["active_work_count"]:
                    return "COVERED_WAITING" if branch["waiting_dependency"] else "COVERED_ACTIVE"
                return "UNCOVERED_ACTIONABLE" if branch["status"] == "OPEN" else "UNCOVERED_NOT_ACTIONABLE"
            result.append({"canonical_objective_id": objective_id, "branches": [{**branch, "coverage_state": state(branch)} for branch in branches], "actionable_uncovered_count": sum(state(branch) == "UNCOVERED_ACTIONABLE" for branch in branches)})
        return result

    def work_planner_candidates(self, project_id: str, limit: int = 50) -> dict:
        """Agenda-aware planning input. It never creates work merely to occupy idle workers."""
        coverage = self.branch_coverage_get(project_id)
        ready = self.work_list(project_id, ["READY"], limit)
        active_branch_ids = {str(item["branch_id"]) for item in ready if item.get("branch_id")}
        undercovered = [branch for branch in coverage if branch["status"] in {"OPEN", "WAITING"} and branch["active_work_count"] == 0]
        return {
            "project_id": project_id,
            "ready_canonical_work": [item for item in ready if item.get("authority_status") == "AUTHORITATIVE"],
            "branch_coverage": coverage,
            "undercovered_branches": undercovered,
            "planner_action": "CLAIM_EXISTING" if ready else "CONSIDER_PROPOSAL" if undercovered else "IDLE",
            "materialization_requires": "approved WorkProposal or authorized planner/gate transition",
        }

    def portfolio_scheduler_shadow(self, project_id: str, limit: int = 50) -> dict:
        """Read-only v3.6 scheduling recommendation; existing READY canonical work always wins."""
        coverage = {item["branch_id"]: item for item in self.branch_coverage_get(project_id)}
        ready = [item for item in self.work_list(project_id, ["READY"], limit) if item.get("authority_status") == "AUTHORITATIVE"]
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,display_name,worker_type,availability_state FROM research_workers WHERE enabled=TRUE ORDER BY created_at")
            workers = [self._record(row) or {} for row in cur.fetchall()]
        assignments, used = [], set()
        for worker in workers:
            if worker["availability_state"] not in {"AVAILABLE", "IDLE"}:
                continue
            candidate = next((item for item in ready if item["id"] not in used), None)
            if candidate:
                used.add(candidate["id"])
                branch = coverage.get(str(candidate.get("branch_id") or ""), {})
                assignments.append({"worker_id": worker["id"], "suggested_work_item_id": candidate["id"], "branch_id": candidate.get("branch_id"), "reason": "EXISTING_READY_CANONICAL_WORK", "dependency_scope": candidate.get("dependency_scope"), "branch_coverage": branch})
            else:
                assignments.append({"worker_id": worker["id"], "suggested_work_item_id": None, "reason": "IDLE_NO_EXISTING_ACTIONABLE_WORK"})
        return {"projection_version": "v3.6-shadow", "assignments": assignments, "unassigned_ready_work_item_ids": [item["id"] for item in ready if item["id"] not in used], "planner_action": "DO_NOT_CREATE_WORK" if not ready else "CLAIM_EXISTING_ONLY"}

    def portfolio_production_audit(self, project_id: str) -> dict:
        """Read-only v3.6 coordination audit; no scheduler mutation or planning."""
        coverage = self.branch_coverage_get(project_id)
        ready = [item for item in self.work_list(project_id, ["READY"], 500) if item.get("authority_status") == "AUTHORITATIVE"]
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,display_name,worker_type,availability_state,availability_updated_at,idle_reason,idle_since "
                "FROM research_workers WHERE enabled=TRUE ORDER BY display_name"
            )
            workers = [self._record(row) or {} for row in cur.fetchall()]
            cur.execute(
                "SELECT w.id,w.branch_id,w.dependency_scope,d.target_type,d.target_id FROM lab_work_items w "
                "LEFT JOIN lab_work_dependencies d ON d.work_item_id=w.id "
                "WHERE w.project_id=%s AND w.status IN ('WAITING_DEPENDENCY','BLOCKED','DORMANT') ORDER BY w.created_at,d.created_at",
                (project_id,),
            )
            waiting_rows = [self._record(row) or {} for row in cur.fetchall()]
            cur.execute(
                "SELECT w.id,w.status,w.branch_id,w.scientific_role,w.resource_class,w.speculation_class,w.related_refs "
                "FROM lab_work_items w WHERE w.project_id=%s AND w.status=ANY(%s) ORDER BY w.created_at",
                (project_id, ["CLAIMED", "RUNNING", "RUNNING_DETACHED", "RESULT_READY"]),
            )
            active = [self._record(row) or {} for row in cur.fetchall()]

        available = [item for item in workers if item.get("availability_state") == "AVAILABLE"]
        idle = [item for item in workers if item.get("availability_state") == "IDLE"]
        by_scope: dict[str, int] = {}
        for item in waiting_rows:
            by_scope[item.get("dependency_scope") or "WORKITEM_LOCAL"] = by_scope.get(item.get("dependency_scope") or "WORKITEM_LOCAL", 0) + 1
        overconcentrated = [item for item in coverage if item["active_worker_count"] > 1]
        uncovered = [item for item in coverage if item["next_actionability"] == "PLANNER_EVALUATION_REQUIRED"]
        no_work = [item for item in coverage if item["status"] in {"OPEN", "WAITING"} and not item["active_work_item_ids"]]
        detached = [item for item in active if item["status"] == "RUNNING_DETACHED"]
        speculative = [item for item in active if item.get("speculation_class") != "NON_SPECULATIVE"]
        review_branches: dict[str, int] = {}
        for item in active:
            if item.get("resource_class") == "REVIEW":
                key = str(item.get("branch_id") or "UNSCOPED")
                review_branches[key] = review_branches.get(key, 0) + 1
        amplification = []
        if idle and ready:
            amplification.append("IDLE_WORKERS_WITH_EXISTING_READY_CANONICAL_WORK")
        if overconcentrated:
            amplification.append("SAME_BRANCH_OVERCONCENTRATION")
        if sum(review_branches.values()) > 1 and len(review_branches) == 1:
            amplification.append("REVIEW_CAPACITY_CONCENTRATION")
        return {
            "audit_version": "v3.6-read-only", "project_id": project_id,
            "workers": {"available": available, "idle": idle, "all": workers},
            "waiting_work_by_dependency_scope": by_scope,
            "unresolved_branches": [item for item in coverage if item["status"] in {"OPEN", "WAITING", "BLOCKED", "WEAKENED", "SUPPORTED_WITHIN_SCOPE"}],
            "actionable_uncovered_branches": uncovered,
            "independent_branches_without_work": no_work,
            "branches_with_excessive_worker_concentration": overconcentrated,
            "existing_ready_canonical_work": ready,
            "long_running_detached_work": detached,
            "speculative_work_active": speculative,
            "review_work_by_branch": review_branches,
            "potential_coordination_amplification": amplification,
            "mutated": False,
        }

    def _parallel_plan_record(self, project_id: str, portfolio_id: str | None,
                              assignments: list[dict], rationale: dict) -> dict:
        """Persist a versioned allocation decision; it contains no scientific conclusion."""
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"parallel-plan:{project_id}:{portfolio_id or 'project'}",))
            cur.execute(
                "SELECT COALESCE(MAX(version),0)+1 AS version FROM parallel_research_plans "
                "WHERE project_id=%s AND portfolio_id IS NOT DISTINCT FROM %s",
                (project_id, portfolio_id),
            )
            version = cur.fetchone()["version"]
            cur.execute(
                "INSERT INTO parallel_research_plans(id,project_id,portfolio_id,version,status,assignments,rationale,created_at) "
                "VALUES(%s,%s,%s,%s,'APPLIED',%s,%s,%s)",
                (ident, project_id, portfolio_id, version, json.dumps(assignments), json.dumps(rationale), now),
            )
            cur.execute("SELECT * FROM parallel_research_plans WHERE id=%s", (ident,))
            return self._record(cur.fetchone()) or {}

    def portfolio_assign_existing(self, project_id: str, worker_id: str, session_id: str,
                                  limit: int = 50) -> dict:
        """Feature-gated v3.6 allocation for one available worker.

        This stage intentionally never materializes work: it can only atomically
        claim an existing authoritative READY WorkItem, or persist an explicit
        idle reason.  `claim_work` remains the concurrency guard, so concurrent
        scheduler processes cannot assign the same item twice.
        """
        if not self.feature_flags.get("BRANCH_AWARE_ASSIGNMENT"):
            raise GPUError("PORTFOLIO_SCHEDULER_DISABLED", "Set BRANCH_AWARE_ASSIGNMENT=true after shadow review")
        with self.store._connect() as conn, conn.cursor() as cur:
            worker = self._worker(cur, worker_id)
            self._session(cur, session_id, worker_id, project_id)
            if worker.get("availability_state") not in {"AVAILABLE", "IDLE"}:
                raise GPUError("LAB_WORKER_NOT_AVAILABLE", str(worker.get("availability_state")))

        coverage = {entry["branch_id"]: entry for entry in self.branch_coverage_get(project_id)}
        ready = [
            item for item in self.work_list(project_id, ["READY"], limit)
            if item.get("authority_status") == "AUTHORITATIVE"
            # Deterministic reallocation never adds a second worker to an
            # already-covered branch. A separately authorized correction or
            # review can still be explicitly claimed outside this allocator.
            and not (item.get("branch_id") and coverage.get(str(item["branch_id"]), {}).get("active_worker_count", 0) > 0)
        ]

        def rank(item: dict) -> tuple:
            branch = coverage.get(str(item.get("branch_id") or ""), {})
            # Lower active-worker concentration is better; branchless work is
            # valid but ranks after a genuinely under-covered scientific branch.
            concentration = int(branch.get("active_worker_count", 0)) if branch else 10_000
            actionability = 0 if branch.get("next_actionability") == "READY_EXISTING_WORK" else 1
            return (concentration, actionability, -float(item.get("priority") or 0), str(item["id"]))

        chosen: dict | None = None
        for candidate in sorted(ready, key=rank):
            try:
                chosen = self.claim_work(candidate["id"], worker_id, session_id)
                break
            except GPUError as error:
                if error.error_type not in {
                    "LAB_WORK_NOT_CLAIMABLE", "LAB_WORK_DEPENDENCY_UNSATISFIED",
                    "LAB_WORK_DEPENDENCY_INVALIDATED",
                }:
                    raise

        if chosen:
            branch_id = chosen.get("branch_id")
            portfolio_id = None
            if branch_id:
                with self.store._connect() as conn, conn.cursor() as cur:
                    cur.execute(
                        "SELECT p.id FROM hypothesis_portfolios p JOIN hypothesis_branches b "
                        "ON b.canonical_objective_id=p.canonical_objective_id "
                        "WHERE b.id=%s AND p.project_id=%s AND p.status='ACTIVE' "
                        "ORDER BY p.created_at DESC LIMIT 1",
                        (branch_id, project_id),
                    )
                    portfolio = cur.fetchone()
                    portfolio_id = str(portfolio["id"]) if portfolio else None
            plan = self._parallel_plan_record(
                project_id, portfolio_id,
                [{"worker_id": worker_id, "session_id": session_id, "work_item_id": chosen["id"], "branch_id": branch_id}],
                {"reason": "EXISTING_READY_CANONICAL_WORK", "new_work_created": False},
            )
            return {"status": "ASSIGNED", "work_item": chosen, "plan": plan,
                    "reason": "EXISTING_READY_CANONICAL_WORK"}

        now = self._now()
        idle_reason = "NO_EXISTING_ACTIONABLE_CANONICAL_WORK"
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, session_id, worker_id, project_id)
            cur.execute(
                "UPDATE research_workers SET availability_state='IDLE',availability_updated_at=%s,idle_reason=%s,idle_since=%s WHERE id=%s",
                (now, idle_reason, now, worker_id),
            )
        plan = self._parallel_plan_record(
            project_id, None, [{"worker_id": worker_id, "session_id": session_id, "work_item_id": None}],
            {"reason": idle_reason, "new_work_created": False,
             "planner_action": "PLANNER_EVALUATION_REQUIRED" if self.feature_flags.get("PLANNER_ON_IDLE") else "IDLE"},
        )
        return {"status": "IDLE", "idle_reason": idle_reason, "plan": plan,
                "planner_action": "PLANNER_EVALUATION_REQUIRED" if self.feature_flags.get("PLANNER_ON_IDLE") else "IDLE"}

    def canonical_execution_projection(self, project_id: str) -> dict:
        """Separate current canonical execution from physically real historical runs."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE project_id=%s AND related_refs ? 'experiment_run_id' ORDER BY updated_at DESC", (project_id,))
            current, historical = [], []
            for item in cur.fetchall():
                entry = {"work_item_id": str(item["id"]), "run_id": item["related_refs"].get("experiment_run_id"), "subject_id": item["subject_id"], "subject_version": item["canonical_subject_version"], "physical_work_status": item["status"], "canonical": item["status"] not in {"SUPERSEDED", "INVALIDATED"} and item["authority_status"] == "AUTHORITATIVE", "superseded_by": str(item["superseded_by"]) if item["superseded_by"] else None}
                (current if entry["canonical"] else historical).append(entry)
            return {"current_canonical_runs": current, "historical_or_noncanonical_runs": historical}

    def work_authority_get(self, project_id: str, authority_key: str,
                           canonical_subject_version: str | None = None) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            sql = "SELECT * FROM lab_work_items WHERE project_id=%s AND authority_key=%s AND authority_status='AUTHORITATIVE'"
            args: list[Any] = [project_id, authority_key]
            if canonical_subject_version is not None:
                sql += " AND canonical_subject_version=%s"
                args.append(canonical_subject_version)
            sql += " ORDER BY CASE WHEN status=ANY(%s) THEN 0 WHEN status='COMPLETED' THEN 1 ELSE 2 END,created_at LIMIT 1"
            args.append(list(EQUIVALENCE_ACTIVE_WORK_STATUSES))
            cur.execute(sql, args)
            row = cur.fetchone()
            if not row:
                raise GPUError("LAB_WORK_AUTHORITY_NOT_FOUND", authority_key)
            return self._record(row) or {}

    def work_equivalence_lookup(self, project_id: str, equivalence_key: str,
                                canonical_subject_version: str | None = None) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            sql = "SELECT * FROM lab_work_items WHERE project_id=%s AND equivalence_key=%s"
            args: list[Any] = [project_id, equivalence_key]
            if canonical_subject_version is not None:
                sql += " AND canonical_subject_version=%s"
                args.append(canonical_subject_version)
            sql += " AND status NOT IN ('INVALIDATED','SUPERSEDED','CANCELLED','FAILED') ORDER BY CASE WHEN status=ANY(%s) THEN 0 WHEN status='COMPLETED' THEN 1 ELSE 2 END,created_at LIMIT 1"
            args.append(list(ACTIVE_WORK_STATUSES))
            cur.execute(sql, args)
            row = cur.fetchone()
            return {"result": "NOT_FOUND"} if not row else {"result": "REUSED_ACTIVE" if row["status"] in ACTIVE_WORK_STATUSES else "ALREADY_SATISFIED" if row["status"] == "COMPLETED" else "HISTORICAL", "work_item": self._record(row)}

    def dependency_status(self, project_id: str, dependency: dict) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            satisfied, invalidated, detail = self._dependency_status(cur, project_id, dependency)
            return {"satisfied": satisfied, "invalidated": invalidated, "detail": detail}

    def consistency_conflicts_get(self, project_id: str, status: str = "OPEN") -> list[dict]:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_consistency_conflicts WHERE project_id=%s AND status=%s ORDER BY detected_at", (project_id, status.upper()))
            return [self._record(row) or {} for row in cur.fetchall()]

    def _lab_budget_get(self, project_id: str) -> dict[str, Any]:
        with self.store._connect() as conn, conn.cursor() as cur:
            return self._budget(cur, project_id)
