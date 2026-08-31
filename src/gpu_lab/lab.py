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
from .research_portfolio_v36_shadow import ResearchPortfolioV36Shadow

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
WORK_AUTHORITY_CLASSES = {"AUTHORITATIVE", "SPECULATIVE", "PREPARATORY", "SUPPORTING", "INDEPENDENT_REVIEW", "RECOVERY"}
PROMOTION_STATES = {"NOT_APPLICABLE", "BLOCKED", "CONDITIONALLY_ELIGIBLE", "ELIGIBLE", "PROMOTED", "REBASE_REQUIRED", "INVALIDATED", "SUPERSEDED"}
REVIEW_VERDICTS = {"PASS", "PASS_WITH_NONMATERIAL_NOTES", "MATERIAL_REPAIR_REQUIRED", "REJECT", "INVALID_REVIEW", "INCONCLUSIVE"}
DEPENDENCY_PREDICATES = {"STATUS_IN", "REVIEW_VERDICT_EQUALS", "REVIEW_VERDICT_IN"}
GUARD_PREDICATES = {
    "MUST_BE_TRUE", "MUST_BE_FALSE", "EQUALS", "NOT_EQUALS", "IN_SET", "NOT_IN_SET",
    "EXISTS", "NOT_EXISTS", "HASH_EQUALS", "VERSION_EQUALS", "VERDICT_EQUALS", "VERDICT_IN",
    "GREATER_THAN", "GREATER_OR_EQUAL", "LESS_THAN", "LESS_OR_EQUAL", "ATTESTATION_VALID",
    "INFORMATIONAL",
}
OBJECTIVE_STATUSES = {"ACTIVE", "BLOCKED", "RESOLVED", "SUPERSEDED", "PAUSED", "CANCELLED"}
PROPOSAL_STATUSES = {"PROPOSED", "MERGED_INTO_EXISTING", "SATISFIED_BY_TERMINAL", "MATERIALIZED", "REJECTED_DUPLICATE", "REJECTED_LOW_VALUE", "REJECTED_STALE", "SUPERSEDED"}
DEPENDENCY_SCOPES = {"WORKITEM_LOCAL", "BRANCH", "PORTFOLIO", "OBJECTIVE_GLOBAL"}
BRANCH_STATES = {"OPEN", "WAITING", "BLOCKED", "SUPPORTED_WITHIN_SCOPE", "WEAKENED", "REFUTED", "RESOLVED", "SUPERSEDED", "ABANDONED"}
SPECULATION_CLASSES = {"NON_SPECULATIVE", "SAFE_SPECULATIVE", "CONDITIONAL_SPECULATIVE", "EXPENSIVE_SPECULATIVE"}
RESOURCE_CLASSES = {"REASONING", "ENGINEERING", "GPU", "TRAINING", "LITERATURE", "REVIEW", "SYNTHESIS"}
EVIDENCE_NEED_EXCEPTIONS = {
    "INDEPENDENT_REPLICATION", "INDEPENDENT_REVIEW", "CORRECTION_VERIFICATION",
    "NEW_ANALYSIS_QUESTION", "NEW_ARTIFACT_VERSION", "DIFFERENT_EVIDENCE_FAMILY", "GENERALIZATION_SCOPE",
}
GATE_STATUSES = {"PENDING", "PREFLIGHT_FAILED", "AWAITING_SEMANTIC_REVIEW", "REPAIR_REQUIRED", "PASS", "FAIL", "INVALID", "SUPERSEDED"}
GATE_REVOCATION_REASONS = {"MALFORMED_SUBJECT_VERSION", "DUPLICATE_EXECUTION_AUTHORITY", "INVALID_PROVENANCE_BINDING"}
PREFLIGHT_STATUSES = {"PASS", "FAIL"}
COORDINATION_VERSION = "lab-coordination-v3.2.2-gates-v1"
PLANNER_LEASE_SECONDS = 300
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
                CREATE TABLE IF NOT EXISTS lab_project_planner_leases (
                    project_id UUID PRIMARY KEY REFERENCES research_projects(id),
                    worker_id UUID NOT NULL REFERENCES research_workers(id),
                    session_id UUID NOT NULL REFERENCES research_worker_sessions(id),
                    acquired_at TIMESTAMPTZ NOT NULL, heartbeat_at TIMESTAMPTZ NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL, lease_version INTEGER NOT NULL DEFAULT 1
                );
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
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS preferred_worker_id UUID REFERENCES research_workers(id);
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS affinity_reason TEXT;
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS authority_class TEXT NOT NULL DEFAULT 'SUPPORTING';
                ALTER TABLE lab_work_items ADD COLUMN IF NOT EXISTS promotion_state TEXT NOT NULL DEFAULT 'NOT_APPLICABLE';
                ALTER TABLE lab_work_dependencies ADD COLUMN IF NOT EXISTS predicate_type TEXT NOT NULL DEFAULT 'STATUS_IN';
                ALTER TABLE lab_work_dependencies ADD COLUMN IF NOT EXISTS predicate_value JSONB NOT NULL DEFAULT '[]'::jsonb;
                CREATE TABLE IF NOT EXISTS lab_review_results (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    review_work_item_id UUID NOT NULL REFERENCES lab_work_items(id),
                    target_subject_id TEXT NOT NULL, target_subject_version TEXT NOT NULL,
                    review_type TEXT NOT NULL, verdict TEXT NOT NULL, findings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    materiality TEXT, reviewed_hashes JSONB NOT NULL DEFAULT '{}'::jsonb,
                    lineage_snapshot_id TEXT, created_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(project_id,review_work_item_id,review_type)
                );
                CREATE INDEX IF NOT EXISTS lab_review_results_subject_idx
                    ON lab_review_results(project_id,target_subject_id,target_subject_version,review_type,created_at DESC);
                ALTER TABLE scientific_gates ADD COLUMN IF NOT EXISTS semantic_review_result_id UUID REFERENCES lab_review_results(id);
                CREATE TABLE IF NOT EXISTS lab_speculative_bindings (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    work_item_id UUID NOT NULL UNIQUE REFERENCES lab_work_items(id),
                    parent_subject_id TEXT NOT NULL, parent_subject_version TEXT NOT NULL,
                    parent_semantic_fingerprint TEXT NOT NULL, speculation_class TEXT NOT NULL,
                    reusable_if_parent_changes BOOLEAN NOT NULL DEFAULT FALSE,
                    invalidation_conditions JSONB NOT NULL DEFAULT '[]'::jsonb,
                    rebase_policy TEXT NOT NULL, max_cost_class TEXT NOT NULL,
                    protected_data_access BOOLEAN NOT NULL DEFAULT FALSE,
                    authority_effect TEXT NOT NULL DEFAULT 'NONE', created_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lab_speculative_rebase_assessments (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    binding_id UUID NOT NULL REFERENCES lab_speculative_bindings(id),
                    old_parent_version TEXT NOT NULL, new_parent_version TEXT NOT NULL,
                    changed_semantic_fields JSONB NOT NULL DEFAULT '[]'::jsonb,
                    affected_modules JSONB NOT NULL DEFAULT '[]'::jsonb,
                    unaffected_modules JSONB NOT NULL DEFAULT '[]'::jsonb,
                    reusable_artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
                    invalidated_artifacts JSONB NOT NULL DEFAULT '[]'::jsonb,
                    required_repairs JSONB NOT NULL DEFAULT '[]'::jsonb,
                    result TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(binding_id,new_parent_version)
                );
                CREATE TABLE IF NOT EXISTS lab_evidence_needs (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    need_key TEXT NOT NULL, exception_key TEXT NOT NULL DEFAULT '',
                    empirical_origin TEXT NOT NULL, artifact_version TEXT NOT NULL,
                    scientific_question TEXT NOT NULL, inspection_role TEXT NOT NULL,
                    evidence_family_intent TEXT NOT NULL, analysis_version TEXT NOT NULL,
                    exception_type TEXT, work_item_id UUID NOT NULL REFERENCES lab_work_items(id),
                    created_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(project_id,need_key,exception_key)
                );
                CREATE INDEX IF NOT EXISTS lab_evidence_needs_project_need_idx
                    ON lab_evidence_needs(project_id,need_key);
                CREATE TABLE IF NOT EXISTS lab_review_harnesses (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    scientific_object_id TEXT NOT NULL, scientific_spec_version TEXT NOT NULL,
                    scientific_semantic_hash TEXT NOT NULL, target_interface_version TEXT NOT NULL,
                    prepared_by_worker_id UUID NOT NULL REFERENCES research_workers(id),
                    prepared_by_session_id UUID NOT NULL REFERENCES research_worker_sessions(id),
                    invariants JSONB NOT NULL DEFAULT '[]'::jsonb,
                    adversarial_cases JSONB NOT NULL DEFAULT '[]'::jsonb,
                    protected_data_access BOOLEAN NOT NULL DEFAULT FALSE,
                    implementation_peer_exposure TEXT NOT NULL DEFAULT 'NONE',
                    status TEXT NOT NULL DEFAULT 'FROZEN', created_at TIMESTAMPTZ NOT NULL,
                    UNIQUE(project_id,scientific_object_id,scientific_spec_version,scientific_semantic_hash,target_interface_version,prepared_by_worker_id)
                );
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

    def _planner_lease_elect(self, cur, project_id: str, worker_id: str, session_id: str) -> dict:
        """Acquire the one expiring planner lease for a project."""
        now = self._now()
        expiry = now + timedelta(seconds=PLANNER_LEASE_SECONDS)
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"planner:{project_id}",))
        cur.execute("SELECT * FROM lab_project_planner_leases WHERE project_id=%s FOR UPDATE", (project_id,))
        current = cur.fetchone()
        if current and current["expires_at"] > now:
            if str(current["worker_id"]) == str(worker_id) and str(current["session_id"]) == str(session_id):
                cur.execute(
                    "UPDATE lab_project_planner_leases SET heartbeat_at=%s,expires_at=%s,lease_version=lease_version+1 "
                    "WHERE project_id=%s RETURNING *", (now, expiry, project_id),
                )
                return self._record(cur.fetchone()) or {}
            raise GPUError("LAB_PROJECT_PLANNER_LEASE_HELD", str(current["worker_id"]))
        if current:
            cur.execute(
                "UPDATE lab_project_planner_leases SET worker_id=%s,session_id=%s,acquired_at=%s,heartbeat_at=%s,"
                "expires_at=%s,lease_version=lease_version+1 WHERE project_id=%s RETURNING *",
                (worker_id, session_id, now, now, expiry, project_id),
            )
        else:
            cur.execute(
                "INSERT INTO lab_project_planner_leases(project_id,worker_id,session_id,acquired_at,heartbeat_at,expires_at) "
                "VALUES(%s,%s,%s,%s,%s,%s) RETURNING *", (project_id, worker_id, session_id, now, now, expiry),
            )
        lease = self._record(cur.fetchone()) or {}
        self._event(cur, project_id, "PROJECT_PLANNER_ELECTED", None, {
            "worker_id": worker_id, "session_id": session_id, "expires_at": expiry.isoformat(),
        })
        return lease

    def planner_authorize(self, project_id: str, worker_id: str, session_id: str,
                          confirmation: str) -> dict:
        """Apply an explicit user-directed planner handoff for one project."""
        if confirmation != "AUTHORIZE_PLANNER":
            raise GPUError("LAB_PLANNER_AUTHORIZATION_CONFIRMATION_REQUIRED", "AUTHORIZE_PLANNER")
        now = self._now()
        expiry = now + timedelta(seconds=PLANNER_LEASE_SECONDS)
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, session_id, worker_id, project_id)
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"planner:{project_id}",))
            cur.execute("SELECT * FROM lab_project_planner_leases WHERE project_id=%s FOR UPDATE", (project_id,))
            previous = cur.fetchone()
            if previous:
                cur.execute(
                    "UPDATE lab_project_planner_leases SET worker_id=%s,session_id=%s,acquired_at=%s,heartbeat_at=%s,"
                    "expires_at=%s,lease_version=lease_version+1 WHERE project_id=%s RETURNING *",
                    (worker_id, session_id, now, now, expiry, project_id),
                )
            else:
                cur.execute(
                    "INSERT INTO lab_project_planner_leases(project_id,worker_id,session_id,acquired_at,heartbeat_at,expires_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s) RETURNING *",
                    (project_id, worker_id, session_id, now, now, expiry),
                )
            lease = self._record(cur.fetchone()) or {}
            self._event(cur, project_id, "PROJECT_PLANNER_AUTHORIZED", None, {
                "worker_id": worker_id, "session_id": session_id,
                "previous_worker_id": str(previous["worker_id"]) if previous else None,
                "expires_at": expiry.isoformat(),
            })
            return {"planner_lease": lease, "replaced_worker_id": str(previous["worker_id"]) if previous else None}

    def work_proposal_materialize(
        self, proposal_id: str, planner_worker_id: str, session_id: str,
        kind: str, title: str, description: str, scientific_role: str,
        priority: float = 0, expected_value: float | None = None,
        estimated_cost: float | None = None, related_refs: dict[str, Any] | None = None,
        dependencies: list[dict] | None = None, authority_key: str | None = None,
        gate_id: str | None = None, canonical_subject_version: str | None = None,
        authority_status: str = "SUPPORTING", subject_id: str | None = None,
        recovery_policy: dict[str, Any] | None = None,
        speculation_class: str = "NON_SPECULATIVE", speculation_condition: dict[str, Any] | None = None,
        resource_class: str | None = None, dependency_scope: str = "WORKITEM_LOCAL",
        required_capabilities: list[str] | None = None,
    ) -> dict:
        """Central planner-only proposal materialization; workers cannot self-allocate work."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_proposals WHERE id=%s FOR UPDATE", (proposal_id,))
            proposal = cur.fetchone()
            if not proposal:
                raise GPUError("LAB_WORK_PROPOSAL_NOT_FOUND", proposal_id)
            self._session(cur, session_id, planner_worker_id, str(proposal["project_id"]))
            self._planner_lease_elect(cur, str(proposal["project_id"]), planner_worker_id, session_id)
            if proposal["status"] != "PROPOSED":
                return {"proposal": self._record(proposal), "materialization": "NOT_ACTIONABLE"}
            proposal_data = self._record(proposal) or {}

        work = self.create_work(
            proposal_data["project_id"], kind, title, description, scientific_role,
            planner_worker_id, priority, expected_value, estimated_cost, related_refs,
            dependencies, proposal_data.get("proposed_equivalence_key_hint"), None, session_id,
            authority_key or proposal_data.get("proposed_authority_key_hint"), gate_id,
            canonical_subject_version, authority_status, subject_id, recovery_policy,
            canonical_objective_id=proposal_data.get("canonical_objective_id"),
            branch_id=proposal_data.get("hypothesis_branch_id"), speculation_class=speculation_class,
            speculation_condition=speculation_condition, resource_class=resource_class,
            dependency_scope=dependency_scope, required_capabilities=required_capabilities,
        )
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE lab_work_proposals SET status='MATERIALIZED',canonical_work_item_id=%s,updated_at=%s "
                "WHERE id=%s AND status='PROPOSED' RETURNING *",
                (work["id"], now, proposal_id),
            )
            updated = cur.fetchone()
            if updated:
                self._event(cur, proposal_data["project_id"], "WORK_PROPOSAL_MATERIALIZED", proposal_id, {"work_item_id": work["id"]})
        return {"proposal": self._record(updated) if updated else proposal_data, "work_item": work,
                "materialization": "CREATED_OR_REUSED"}

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
            self._refresh_portfolio_branches(cur, project_id, canonical_objective_id, now)
            cur.execute("SELECT * FROM hypothesis_branches WHERE id=%s", (ident,))
            return self._record(cur.fetchone()) or {}

    def _refresh_portfolio_branches(self, cur, project_id: str, canonical_objective_id: str,
                                    now: datetime) -> None:
        """Keep the persisted portfolio membership aligned with branch lifecycle.

        This is a scheduling projection, not scientific interpretation.  It is
        deliberately refreshed inside the same transaction that creates or
        transitions a branch, so controller restart cannot resurrect a branch
        that was already retired from the current objective version.
        """
        cur.execute("SELECT version FROM canonical_objectives WHERE id=%s AND project_id=%s", (canonical_objective_id, project_id))
        objective = cur.fetchone()
        if not objective:
            return
        cur.execute(
            "SELECT id FROM hypothesis_branches WHERE canonical_objective_id=%s "
            "AND state IN ('OPEN','WAITING','BLOCKED','WEAKENED','SUPPORTED_WITHIN_SCOPE') "
            "ORDER BY priority DESC,created_at",
            (canonical_objective_id,),
        )
        branch_ids = [str(row["id"]) for row in cur.fetchall()]
        cur.execute(
            "UPDATE hypothesis_portfolios SET active_branch_ids=%s,status=%s,updated_at=%s "
            "WHERE project_id=%s AND canonical_objective_id=%s AND objective_version=%s",
            (json.dumps(branch_ids), "ACTIVE" if branch_ids else "RESOLVED", now,
             project_id, canonical_objective_id, objective["version"]),
        )

    def hypothesis_branch_transition(self, branch_id: str, worker_id: str, session_id: str,
                                     state: str, rationale: str) -> dict:
        """Resolve/refute one branch and retire only its active descendants."""
        state = self._validate(state, BRANCH_STATES, "HYPOTHESIS_BRANCH_STATE")
        if state not in {"RESOLVED", "REFUTED", "ABANDONED", "SUPERSEDED", "WEAKENED", "SUPPORTED_WITHIN_SCOPE"}:
            raise GPUError("HYPOTHESIS_BRANCH_TRANSITION_INVALID", state)
        if not rationale.strip():
            raise GPUError("HYPOTHESIS_BRANCH_RATIONALE_REQUIRED", "rationale")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM hypothesis_branches WHERE id=%s FOR UPDATE", (branch_id,))
            branch = cur.fetchone()
            if not branch:
                raise GPUError("HYPOTHESIS_BRANCH_NOT_FOUND", branch_id)
            self._session(cur, session_id, worker_id, str(branch["project_id"]))
            cur.execute("UPDATE hypothesis_branches SET state=%s,resolved_at=CASE WHEN %s IN ('RESOLVED','REFUTED','ABANDONED','SUPERSEDED') THEN %s ELSE resolved_at END WHERE id=%s", (state, state, now, branch_id))
            retired = 0
            if state in {"RESOLVED", "REFUTED", "ABANDONED", "SUPERSEDED"}:
                # A terminal branch state retires *operational work ownership*, not
                # merely its display status.  Leaving a lease/session behind here
                # makes a worker look BUSY against invalidated work and can block
                # the portfolio scheduler indefinitely after a branch refutation.
                cur.execute(
                    "SELECT * FROM lab_work_items WHERE branch_id=%s AND status=ANY(%s) FOR UPDATE",
                    (branch_id, list(ACTIVE_WORK_STATUSES)),
                )
                retired_items = cur.fetchall()
                released_worker_ids: set[str] = set()
                for item in retired_items:
                    if item["assigned_worker_id"]:
                        released_worker_ids.add(str(item["assigned_worker_id"]))
                    cur.execute(
                        "UPDATE lab_work_leases SET released_at=%s,release_reason='BRANCH_RETIRED',lease_version=lease_version+1 "
                        "WHERE work_item_id=%s AND released_at IS NULL",
                        (now, item["id"]),
                    )
                    cur.execute(
                        "UPDATE lab_work_items SET status='INVALIDATED',invalidated_reason=%s,invalidated_at=%s,"
                        "completed_at=%s,assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,"
                        "updated_at=%s,work_version=work_version+1 WHERE id=%s",
                        (f"Branch {state}: {rationale.strip()}", now, now, now, item["id"]),
                    )
                    cur.execute(
                        "UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',"
                        "last_heartbeat_at=%s WHERE current_work_item_id=%s",
                        (now, item["id"]),
                    )
                    self._event(cur, branch["project_id"], "WORK_ITEM_INVALIDATED", item["id"], {
                        "reason": f"Branch {state}: {rationale.strip()}", "branch_id": branch_id,
                    })
                    retired += 1
                # A worker can have more than one live session, so publish it as
                # AVAILABLE only when no remaining session still owns work.
                if released_worker_ids:
                    cur.execute(
                        "UPDATE research_workers w SET availability_state='AVAILABLE',availability_updated_at=%s,"
                        "idle_reason=NULL,idle_since=NULL WHERE w.id=ANY(%s) AND NOT EXISTS ("
                        "SELECT 1 FROM research_worker_sessions s WHERE s.worker_id=w.id "
                        "AND s.current_work_item_id IS NOT NULL AND s.status IN ('ACTIVE','BUSY','WAITING'))",
                        (now, list(released_worker_ids)),
                    )
            self._refresh_portfolio_branches(cur, str(branch["project_id"]), str(branch["canonical_objective_id"]), now)
            self._event(cur, branch["project_id"], "HYPOTHESIS_BRANCH_TRANSITIONED", branch_id, {"state": state, "retired_descendants": retired})
        return {"branch_id": branch_id, "state": state, "retired_descendants": retired}

    @staticmethod
    def _canonical_json_hash(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    @classmethod
    def _evidence_need_normalize(cls, evidence_need: dict[str, Any]) -> dict[str, str | None]:
        """Return a stable EvidenceNeed identity, including explicit exceptions."""
        if not isinstance(evidence_need, dict):
            raise GPUError("LAB_EVIDENCE_NEED_INVALID", "evidence_need must be an object")
        fields = (
            "empirical_origin", "artifact_version", "scientific_question", "inspection_role",
            "evidence_family_intent", "analysis_version",
        )
        normalized = {field: str(evidence_need.get(field) or "").strip() for field in fields}
        if any(not value for value in normalized.values()):
            raise GPUError("LAB_EVIDENCE_NEED_FIELDS_REQUIRED", ", ".join(fields))
        exception_type = str(evidence_need.get("exception_type") or "").strip().upper() or None
        exception_key = str(evidence_need.get("exception_key") or "").strip()
        if exception_type:
            if exception_type not in EVIDENCE_NEED_EXCEPTIONS:
                raise GPUError("LAB_EVIDENCE_NEED_EXCEPTION_INVALID", exception_type)
            if not exception_key:
                raise GPUError("LAB_EVIDENCE_NEED_EXCEPTION_KEY_REQUIRED", exception_type)
        elif exception_key:
            raise GPUError("LAB_EVIDENCE_NEED_EXCEPTION_TYPE_REQUIRED", exception_key)
        need_key = cls._canonical_json_hash(normalized)
        return {**normalized, "need_key": need_key, "exception_type": exception_type, "exception_key": exception_key}

    def _worker(self, cur, worker_id: str) -> dict:
        cur.execute("SELECT * FROM research_workers WHERE id=%s AND enabled=TRUE", (worker_id,))
        worker = cur.fetchone()
        if not worker:
            raise GPUError("LAB_WORKER_NOT_FOUND", worker_id)
        return worker

    @staticmethod
    def _capability_names(capabilities: Any) -> set[str]:
        """Normalize the durable worker-capability map without trusting prose."""
        if not isinstance(capabilities, dict):
            return set()
        return {
            name.strip().upper() for name, enabled in capabilities.items()
            if isinstance(name, str) and name.strip() and enabled is True
        }

    @staticmethod
    def _required_capability_names(item: dict) -> set[str]:
        refs = item.get("related_refs") or {}
        raw = refs.get("required_capabilities", []) if isinstance(refs, dict) else []
        if not isinstance(raw, list) or any(not isinstance(value, str) or not value.strip() for value in raw):
            raise GPUError("LAB_WORK_REQUIRED_CAPABILITIES_INVALID", "related_refs.required_capabilities must be a list of strings")
        return {value.strip().upper() for value in raw}

    @classmethod
    def _worker_can_claim_item(cls, worker: dict, item: dict) -> bool:
        return cls._required_capability_names(item).issubset(cls._capability_names(worker.get("capabilities")))

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

    def _clear_stale_session_ownership(self, cur, worker_id: str, now: datetime) -> list[str]:
        """Remove session pointers which no longer represent a live lease.

        ``current_work_item_id`` is a projection for fast coordinator reads,
        not an authority source. A pointer may survive a crash or an orphan
        reconciliation, so it may block neither a fresh session nor a new
        claim unless its WorkItem and lease still belong to that exact session.
        """
        cur.execute(
            "UPDATE research_worker_sessions s SET current_work_item_id=NULL,active_role=NULL,"
            "status=CASE WHEN s.status='BUSY' THEN 'ACTIVE' ELSE s.status END,"
            "last_heartbeat_at=%s "
            "WHERE s.worker_id=%s AND s.current_work_item_id IS NOT NULL "
            "AND NOT EXISTS ("
            "SELECT 1 FROM lab_work_items wi JOIN lab_work_leases l ON l.id=wi.lease_id "
            "WHERE wi.id=s.current_work_item_id "
            "AND wi.status NOT IN ('COMPLETED','SUPERSEDED','FAILED','CANCELLED','INVALIDATED','DORMANT') "
            "AND wi.assigned_worker_id=s.worker_id AND wi.assigned_session_id=s.id "
            "AND l.worker_id=s.worker_id AND l.worker_session_id=s.id "
            "AND l.released_at IS NULL AND l.expires_at>=%s"
            ") RETURNING s.id",
            (now, worker_id, now),
        )
        return [str(row["id"]) for row in cur.fetchall()]

    def _reconcile_worker_availability(self, cur, worker_id: str, now: datetime) -> dict:
        """Repair availability projection before allocator admission.

        A worker can remain EXECUTING after its lease expired or was released,
        even when no session owns work. Availability is a projection, not the
        authority source, so correct it only when no current session has a
        live unreleased lease. Detached execution does not reserve its worker.
        """
        self._clear_stale_session_ownership(cur, worker_id, now)
        cur.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM research_worker_sessions s "
            "JOIN lab_work_items wi ON wi.id=s.current_work_item_id "
            "JOIN lab_work_leases l ON l.id=wi.lease_id "
            "WHERE s.worker_id=%s AND s.status IN ('ACTIVE','BUSY','WAITING') "
            "AND wi.status IN ('CLAIMED','RUNNING') "
            "AND wi.assigned_worker_id=s.worker_id AND wi.assigned_session_id=s.id "
            "AND l.worker_id=s.worker_id AND l.worker_session_id=s.id "
            "AND l.released_at IS NULL AND l.expires_at >= %s"
            ") AS owns_live_work",
            (worker_id, now),
        )
        if not bool(cur.fetchone()["owns_live_work"]):
            cur.execute(
                "UPDATE research_workers SET availability_state='AVAILABLE',availability_updated_at=%s,"
                "idle_reason=NULL,idle_since=NULL WHERE id=%s",
                (now, worker_id),
            )
        return self._worker(cur, worker_id)

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
            # A durable worker identity outlives individual browser/runtime
            # sessions.  Capabilities, however, describe the currently joined
            # runtime and must be refreshable on rejoin.  Treat a supplied map
            # as a patch so a reviewer can add SCIENTIFIC_REVIEW without
            # accidentally dropping unrelated durable capabilities.
            if capabilities is not None:
                if not isinstance(capabilities, dict) or any(
                    not isinstance(name, str) or not name.strip() or not isinstance(enabled, bool)
                    for name, enabled in capabilities.items()
                ):
                    raise GPUError("LAB_WORKER_CAPABILITIES_INVALID", "capabilities must map nonempty names to booleans")
                previous_capabilities = dict(worker["capabilities"] or {})
                refreshed_capabilities = {
                    **previous_capabilities,
                    **{name.strip(): enabled for name, enabled in capabilities.items()},
                }
                if refreshed_capabilities != previous_capabilities:
                    cur.execute(
                        "UPDATE research_workers SET capabilities=%s WHERE id=%s RETURNING *",
                        (json.dumps(refreshed_capabilities), worker["id"]),
                    )
                    worker = cur.fetchone()
                    self._event(cur, project_id, "WORKER_CAPABILITIES_REFRESHED", None, {
                        "worker_id": str(worker["id"]),
                        "updated_capability_names": sorted(capabilities),
                        "enabled_capability_names": sorted(self._capability_names(refreshed_capabilities)),
                    })
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
        # A canonical subject version is one immutable identity, never a
        # mutable composite such as ``release_sha+review-id``.  Review IDs
        # belong in ReviewResult bindings, not in authority identity.
        if not canonical_subject_version.strip() or "+" in canonical_subject_version or any(char.isspace() for char in canonical_subject_version):
            raise GPUError("SCIENTIFIC_GATE_VERSION_NOT_ATOMIC", canonical_subject_version)
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
            # A newer canonical subject version makes any *non-terminal* gate
            # for the same subject/key operationally obsolete.  Keep terminal
            # PASS/FAIL evidence intact, but retire old pending/preflight/review
            # gates and their READY/active work in this transaction.  Leaving
            # those cards READY is unsafe: a worker can otherwise execute a
            # version that the authority layer has already replaced.
            cur.execute(
                "SELECT id FROM scientific_gates WHERE project_id=%s AND gate_key=%s "
                "AND scientific_object_id=%s AND canonical_subject_version<>%s "
                "AND status NOT IN ('PASS','FAIL','INVALID','SUPERSEDED') FOR UPDATE",
                (project_id, gate_key, scientific_object_id, canonical_subject_version),
            )
            stale_gate_ids = [str(row["id"]) for row in cur.fetchall()]
            superseded_work = 0
            if stale_gate_ids:
                cur.execute(
                    "UPDATE scientific_gates SET status='SUPERSEDED',superseded_by=%s,"
                    "invalidation_reason=%s,resolved_at=%s WHERE id=ANY(%s)",
                    (ident, f"Replaced by canonical subject version {canonical_subject_version}", now, stale_gate_ids),
                )
                cur.execute(
                    "SELECT * FROM lab_work_items WHERE gate_id=ANY(%s) AND status=ANY(%s) FOR UPDATE",
                    (stale_gate_ids, list(ACTIVE_WORK_STATUSES)),
                )
                for stale_work in cur.fetchall():
                    cur.execute(
                        "UPDATE lab_work_leases SET released_at=%s,release_reason='CANONICAL_VERSION_SUPERSEDED',"
                        "lease_version=lease_version+1 WHERE work_item_id=%s AND released_at IS NULL",
                        (now, stale_work["id"]),
                    )
                    cur.execute(
                        "UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',superseded_by=%s,"
                        "invalidated_reason=%s,invalidated_at=%s,completed_at=%s,assigned_worker_id=NULL,"
                        "assigned_session_id=NULL,lease_id=NULL,updated_at=%s,work_version=work_version+1 WHERE id=%s",
                        (ident, f"Replaced by canonical subject version {canonical_subject_version}", now, now, now, stale_work["id"]),
                    )
                    cur.execute(
                        "UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',"
                        "last_heartbeat_at=%s WHERE current_work_item_id=%s",
                        (now, stale_work["id"]),
                    )
                    self._event(cur, project_id, "WORK_ITEM_SUPERSEDED", stale_work["id"], {
                        "successor_gate_id": ident,
                        "reason": f"Replaced by canonical subject version {canonical_subject_version}",
                    })
                    superseded_work += 1
                self._event(cur, project_id, "SCIENTIFIC_GATE_VERSION_AUTO_SUPERSEDED", ident, {
                    "superseded_gate_ids": stale_gate_ids,
                    "superseded_work_items": superseded_work,
                    "canonical_subject_version": canonical_subject_version,
                })
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
        """Create or reuse the sole active independent-review slot for a gate."""
        gate = self.gate_get(gate_id)
        if gate["status"] in {"SUPERSEDED", "INVALID", "REPAIR_REQUIRED", "PASS", "FAIL"}:
            raise GPUError("SCIENTIFIC_GATE_NOT_ACTIONABLE", gate["status"])
        # Review slots are singleton per gate/version.  They are deliberately
        # supporting work, so authority-key arbitration alone cannot reuse
        # them; reuse explicitly before creating an equivalence-key duplicate.
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM lab_work_items WHERE gate_id=%s AND authority_class='INDEPENDENT_REVIEW' "
                "AND status=ANY(%s) ORDER BY created_at LIMIT 1",
                (gate_id, list(ACTIVE_WORK_STATUSES)),
            )
            existing = cur.fetchone()
            if existing:
                return {**(self._record(existing) or {}), "dedupe_result": "REUSED_ACTIVE"}
        refs = {"scientific_object_id": gate["scientific_object_id"], "gate_id": gate_id}
        return self.create_work(
            gate["project_id"], kind, title, description, scientific_role,
            worker_id, priority, expected_value, estimated_cost, refs, dependencies,
            f"{gate['authority_key']}:review", None, session_id, gate["authority_key"], gate_id,
            gate["canonical_subject_version"], "SUPPORTING", gate["scientific_object_id"], recovery_policy,
            canonical_objective_id=None, branch_id=branch_id,
            authority_class="INDEPENDENT_REVIEW", promotion_state="NOT_APPLICABLE",
        )

    def gate_authority_work_ensure(
        self, gate_id: str, kind: str, title: str, description: str, scientific_role: str,
        worker_id: str, session_id: str, priority: float = 0, expected_value: float | None = None,
        estimated_cost: float | None = None, dependencies: list[dict] | None = None,
        recovery_policy: dict[str, Any] | None = None, branch_id: str | None = None,
    ) -> dict:
        """Create/reuse the one authoritative work slot for a scientific gate.

        Review work uses :meth:`gate_work_ensure` and is deliberately
        SUPPORTING/INDEPENDENT_REVIEW.  This separate method is the only
        gate-bound creation path that may populate
        ``authoritative_work_item_id``.
        """
        gate = self.gate_get(gate_id)
        if gate["status"] in {"SUPERSEDED", "INVALID", "REPAIR_REQUIRED", "FAIL"}:
            raise GPUError("SCIENTIFIC_GATE_NOT_ACTIONABLE", gate["status"])
        # Repair the short-lived v3.6.5 projection bug where the review slot
        # reused the gate authority's equivalence key.  That collision would
        # otherwise prevent creation of the real authority work item.  This
        # changes only operational dedupe metadata; the review remains intact.
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM lab_work_items WHERE gate_id=%s AND authority_class='INDEPENDENT_REVIEW' "
                "AND equivalence_key=%s AND status=ANY(%s) FOR UPDATE",
                (gate_id, gate["authority_key"], list(ACTIVE_WORK_STATUSES)),
            )
            for review in cur.fetchall():
                review_key = f"{gate['authority_key']}:review:{review['id']}"
                cur.execute(
                    "UPDATE lab_work_items SET equivalence_key=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s",
                    (review_key, now, review["id"]),
                )
                self._event(cur, gate["project_id"], "GATE_REVIEW_EQUIVALENCE_REPAIRED", review["id"], {
                    "old_equivalence_key": gate["authority_key"], "new_equivalence_key": review_key,
                    "reason": "Review slots may not occupy execution authority equivalence keys.",
                })
        refs = {
            "scientific_object_id": gate["scientific_object_id"], "gate_id": gate_id,
            "authority_slot": "GATE_EXECUTION_AUTHORITY",
        }
        return self.create_work(
            gate["project_id"], kind, title, description, scientific_role,
            worker_id, priority, expected_value, estimated_cost, refs, dependencies,
            gate["authority_key"], None, session_id, gate["authority_key"], gate_id,
            gate["canonical_subject_version"], "AUTHORITATIVE", gate["scientific_object_id"], recovery_policy,
            canonical_objective_id=None, branch_id=branch_id,
            authority_class="AUTHORITATIVE", promotion_state="ELIGIBLE",
        )

    @staticmethod
    def _evaluate_guard_predicate(name: str, declaration: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a typed guard without conflating an observed value with PASS.

        ``False`` is a valid passing observation for MUST_BE_FALSE.  Legacy
        ``passed`` checks remain supported only as an already-evaluated
        compatibility representation; new authority-critical callers can
        declare the predicate and its observation explicitly.
        """
        predicate = declaration.get("predicate", declaration.get("type"))
        if predicate is None:
            if isinstance(declaration.get("passed"), bool):
                return {key: declaration[key] for key in sorted(declaration)}
            raise GPUError("PREFLIGHT_GUARD_PREDICATE_REQUIRED", name)
        predicate = str(predicate).upper()
        if predicate not in GUARD_PREDICATES:
            raise GPUError("INVALID_GUARD_PREDICATE", predicate)
        observed_present = "observed" in declaration or "value" in declaration
        observed = declaration["observed"] if "observed" in declaration else declaration.get("value")
        expected = declaration.get("expected")
        if predicate == "INFORMATIONAL":
            passed = True
        elif predicate == "EXISTS":
            passed = observed_present and observed is not None
        elif predicate == "NOT_EXISTS":
            passed = not observed_present or observed is None
        else:
            if not observed_present:
                raise GPUError("PREFLIGHT_GUARD_OBSERVATION_REQUIRED", name)
            if predicate == "MUST_BE_TRUE" or predicate == "ATTESTATION_VALID":
                passed = observed is True
            elif predicate == "MUST_BE_FALSE":
                passed = observed is False
            elif predicate in {"EQUALS", "HASH_EQUALS", "VERSION_EQUALS", "VERDICT_EQUALS"}:
                passed = observed == expected
            elif predicate == "NOT_EQUALS":
                passed = observed != expected
            elif predicate in {"IN_SET", "VERDICT_IN"}:
                if not isinstance(expected, list):
                    raise GPUError("PREFLIGHT_GUARD_EXPECTED_SET_REQUIRED", name)
                passed = observed in expected
            elif predicate == "NOT_IN_SET":
                if not isinstance(expected, list):
                    raise GPUError("PREFLIGHT_GUARD_EXPECTED_SET_REQUIRED", name)
                passed = observed not in expected
            else:
                if not isinstance(observed, (int, float)) or not isinstance(expected, (int, float)):
                    raise GPUError("PREFLIGHT_GUARD_NUMERIC_EXPECTED", name)
                passed = {
                    "GREATER_THAN": observed > expected,
                    "GREATER_OR_EQUAL": observed >= expected,
                    "LESS_THAN": observed < expected,
                    "LESS_OR_EQUAL": observed <= expected,
                }[predicate]
        result = {key: declaration[key] for key in sorted(declaration) if key not in {"passed", "type"}}
        result.update({"predicate": predicate, "passed": bool(passed)})
        return result

    @classmethod
    def _normalize_preflight_checks(cls, checks: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if not checks:
            raise GPUError("PREFLIGHT_CHECKS_REQUIRED", "At least one deterministic check is required")
        normalized: dict[str, dict[str, Any]] = {}
        for name, value in checks.items():
            if not isinstance(name, str) or not name.strip():
                raise GPUError("PREFLIGHT_CHECK_NAME_INVALID", str(name))
            if isinstance(value, bool):
                normalized[name] = {"passed": value, "predicate": "LEGACY_PASSED"}
            elif isinstance(value, dict):
                normalized[name] = cls._evaluate_guard_predicate(name, value)
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
            if gate["status"] in {"SUPERSEDED", "INVALID", "REPAIR_REQUIRED", "PASS", "FAIL"}:
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
            # A gate can explicitly be classified as a technical-only delta at
            # creation time.  In that case its subject version still receives
            # an immutable deterministic preflight, but it must not be routed
            # through a second semantic review which the policy deliberately
            # declared unnecessary.  Previously both paths were sent to
            # AWAITING_SEMANTIC_REVIEW, leaving provenance/runtime-only repairs
            # in a needless review loop and requiring a manual no-op resolve.
            if result["status"] == "FAIL":
                next_status = "PREFLIGHT_FAILED"
            elif gate["semantic_review_required"]:
                next_status = "AWAITING_SEMANTIC_REVIEW"
            else:
                next_status = "PASS"
            cur.execute(
                "UPDATE scientific_gates SET deterministic_preflight_id=%s,status=%s,resolved_at=%s WHERE id=%s",
                (result["id"], next_status, now if next_status == "PASS" else None, gate_id),
            )
            self._event(cur, gate["project_id"], "DETERMINISTIC_PREFLIGHT_REUSED" if reused else "DETERMINISTIC_PREFLIGHT_COMPLETED", result["id"], {
                "gate_id": gate_id, "status": result["status"], "subject_hash": subject_hash,
                "validator_version": validator_version,
            })
            self._event(cur, gate["project_id"], "SCIENTIFIC_GATE_PREFLIGHT_FAILED" if result["status"] == "FAIL" else "SCIENTIFIC_GATE_PREFLIGHT_PASSED", gate_id, {"preflight_id": str(result["id"]), "semantic_review_required": gate["semantic_review_required"]})
            if next_status == "PASS":
                self._event(cur, gate["project_id"], "SCIENTIFIC_GATE_AUTO_RESOLVED_TECHNICAL_DELTA", gate_id, {
                    "preflight_id": str(result["id"]),
                    "rationale": "semantic_review_required=false; deterministic preflight passed",
                })
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
            if gate["status"] in {"SUPERSEDED", "INVALID", "REPAIR_REQUIRED"}:
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
            semantic_review_result_id = None
            if gate["semantic_review_required"] and self.feature_flags.get("VERDICT_AWARE_DEPENDENCIES_V365"):
                # A terminal review task is operational information only.  The
                # version-bound verdict is the authority fact a semantic gate
                # consumes.  This check deliberately happens in the same
                # gate-locking transaction as the PASS/FAIL transition.
                cur.execute(
                    "SELECT id,verdict,target_subject_id,target_subject_version FROM lab_review_results "
                    "WHERE project_id=%s AND review_work_item_id=%s ORDER BY created_at DESC LIMIT 1 FOR SHARE",
                    (gate["project_id"], semantic_review_work_item_id),
                )
                review_result = cur.fetchone()
                if not review_result:
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_VERDICT_REQUIRED", str(semantic_review_work_item_id))
                if (
                    str(review_result["target_subject_id"]) != str(gate["scientific_object_id"])
                    or str(review_result["target_subject_version"]) != str(gate["canonical_subject_version"])
                ):
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_RESULT_BINDING_MISMATCH", str(review_result["id"]))
                verdict = str(review_result["verdict"]).upper()
                passing_verdict = verdict in {"PASS", "PASS_WITH_NONMATERIAL_NOTES"}
                if semantic_status == "PASS" and not passing_verdict:
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_VERDICT_NOT_PASS", verdict)
                if semantic_status == "FAIL" and passing_verdict:
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_VERDICT_CONTRADICTS_STATUS", verdict)
                semantic_review_result_id = str(review_result["id"])
            cur.execute(
                "UPDATE scientific_gates SET status=%s,semantic_review_work_item_id=COALESCE(%s,semantic_review_work_item_id),"
                "semantic_review_result_id=COALESCE(%s,semantic_review_result_id),resolved_at=%s WHERE id=%s",
                (semantic_status, semantic_review_work_item_id, semantic_review_result_id, now, gate_id),
            )
            self._event(cur, gate["project_id"], "SCIENTIFIC_GATE_RESOLVED", gate_id, {
                "status": semantic_status, "semantic_review_work_item_id": semantic_review_work_item_id,
                "semantic_review_result_id": semantic_review_result_id,
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
            # Gate supersession must retire live operational ownership as well
            # as authority. Otherwise an old-version lease can keep its worker
            # BUSY after the replacement gate is already canonical.
            cur.execute(
                "SELECT * FROM lab_work_items WHERE gate_id=%s AND status=ANY(%s) FOR UPDATE",
                (old_gate_id, list(ACTIVE_WORK_STATUSES)),
            )
            retired_items = cur.fetchall()
            released_worker_ids: set[str] = set()
            for item in retired_items:
                if item["assigned_worker_id"]:
                    released_worker_ids.add(str(item["assigned_worker_id"]))
                cur.execute(
                    "UPDATE lab_work_leases SET released_at=%s,release_reason='GATE_SUPERSEDED',lease_version=lease_version+1 "
                    "WHERE work_item_id=%s AND released_at IS NULL",
                    (now, item["id"]),
                )
                cur.execute(
                    "UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',superseded_by=%s,"
                    "invalidated_reason=%s,invalidated_at=%s,completed_at=%s,assigned_worker_id=NULL,"
                    "assigned_session_id=NULL,lease_id=NULL,updated_at=%s,work_version=work_version+1 WHERE id=%s",
                    (successor_gate_id, rationale.strip()[:4000], now, now, now, item["id"]),
                )
                cur.execute(
                    "UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',"
                    "last_heartbeat_at=%s WHERE current_work_item_id=%s",
                    (now, item["id"]),
                )
                self._event(cur, old["project_id"], "WORK_ITEM_SUPERSEDED", item["id"], {
                    "successor_gate_id": successor_gate_id, "reason": rationale.strip()[:4000],
                })
            if released_worker_ids:
                cur.execute(
                    "UPDATE research_workers w SET availability_state='AVAILABLE',availability_updated_at=%s,"
                    "idle_reason=NULL,idle_since=NULL WHERE w.id=ANY(%s) AND NOT EXISTS ("
                    "SELECT 1 FROM research_worker_sessions s WHERE s.worker_id=w.id "
                    "AND s.current_work_item_id IS NOT NULL AND s.status IN ('ACTIVE','BUSY','WAITING'))",
                    (now, list(released_worker_ids)),
                )
            self._event(cur, old["project_id"], "SCIENTIFIC_GATE_VERSION_SUPERSEDED", old_gate_id, {
                "successor_gate_id": successor_gate_id,
                "old_version": old["canonical_subject_version"],
                "successor_version": successor["canonical_subject_version"],
                "rationale": rationale.strip()[:4000],
            })
        dependency_changes = self.resolve_dependencies(str(old["project_id"]))
        return {"old_gate_id": old_gate_id, "successor_gate_id": successor_gate_id, "status": "SUPERSEDED", "dependency_changes": dependency_changes}

    def gate_revoke_malformed_authority(
        self, gate_id: str, replacement_gate_id: str, worker_id: str, session_id: str,
        correction_reason: str, rationale: str,
    ) -> dict[str, Any]:
        """Auditably retire an erroneous terminal gate without rewriting history.

        This is deliberately narrower than a generic "change PASS" operation.
        It requires a current, passing replacement bound to the same scientific
        object and an explicit recognized identity/provenance defect.  The old
        verdict and all events remain durable; only its current authority is
        revoked and live child work is retired.
        """
        correction_reason = self._validate(
            correction_reason, GATE_REVOCATION_REASONS, "SCIENTIFIC_GATE_REVOCATION_REASON"
        )
        if not rationale.strip():
            raise GPUError("SCIENTIFIC_GATE_REVOCATION_RATIONALE_REQUIRED", "rationale")
        if gate_id == replacement_gate_id:
            raise GPUError("SCIENTIFIC_GATE_REVOCATION_REPLACEMENT_REQUIRED", gate_id)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM scientific_gates WHERE id=ANY(%s) FOR UPDATE", ([gate_id, replacement_gate_id],))
            gates = {str(item["id"]): item for item in cur.fetchall()}
            gate, replacement = gates.get(gate_id), gates.get(replacement_gate_id)
            if not gate or not replacement:
                raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", "gate or replacement")
            self._session(cur, session_id, worker_id, str(gate["project_id"]))
            if str(gate["project_id"]) != str(replacement["project_id"]):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", replacement_gate_id)
            if str(gate["scientific_object_id"]) != str(replacement["scientific_object_id"]):
                raise GPUError("SCIENTIFIC_GATE_REVOCATION_SUBJECT_MISMATCH", replacement_gate_id)
            if replacement["status"] != "PASS" or replacement["superseded_by"] or replacement["invalidation_reason"]:
                raise GPUError("SCIENTIFIC_GATE_REVOCATION_REPLACEMENT_NOT_CURRENT_PASS", replacement_gate_id)
            if gate["status"] == "SUPERSEDED":
                if str(gate["superseded_by"] or "") != str(replacement_gate_id):
                    raise GPUError("SCIENTIFIC_GATE_ALREADY_REVOKED", gate_id)
                return {"gate_id": gate_id, "replacement_gate_id": replacement_gate_id, "status": "SUPERSEDED", "idempotent": True}
            if gate["status"] != "PASS":
                raise GPUError("SCIENTIFIC_GATE_REVOCATION_PASS_REQUIRED", str(gate["status"]))
            old_version, replacement_version = str(gate["canonical_subject_version"]), str(replacement["canonical_subject_version"])
            if correction_reason == "MALFORMED_SUBJECT_VERSION":
                if "+" not in old_version or old_version.split("+", 1)[0] != replacement_version:
                    raise GPUError("SCIENTIFIC_GATE_REVOCATION_NOT_MALFORMED_VERSION", old_version)
            elif correction_reason == "DUPLICATE_EXECUTION_AUTHORITY":
                if old_version != replacement_version:
                    raise GPUError("SCIENTIFIC_GATE_REVOCATION_NOT_DUPLICATE_VERSION", old_version)
            # INVALID_PROVENANCE_BINDING is intentionally allowed across
            # versions, provided the current replacement is same-subject PASS.
            cur.execute(
                "UPDATE scientific_gates SET status='SUPERSEDED',superseded_by=%s,invalidation_reason=%s,resolved_at=%s WHERE id=%s",
                (replacement_gate_id, f"Revoked {correction_reason}: {rationale.strip()[:3600]}", now, gate_id),
            )
            cur.execute(
                "SELECT * FROM lab_work_items WHERE gate_id=%s AND status=ANY(%s) FOR UPDATE",
                (gate_id, list(ACTIVE_WORK_STATUSES)),
            )
            retired = cur.fetchall()
            retired_ids = [str(item["id"]) for item in retired]
            for item in retired:
                cur.execute(
                    "UPDATE lab_work_leases SET released_at=%s,release_reason='MALFORMED_GATE_REVOKED',lease_version=lease_version+1 "
                    "WHERE work_item_id=%s AND released_at IS NULL",
                    (now, item["id"]),
                )
                cur.execute(
                    "UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',promotion_state='SUPERSEDED',"
                    "superseded_by=%s,invalidated_reason=%s,invalidated_at=%s,assigned_worker_id=NULL,assigned_session_id=NULL,"
                    "lease_id=NULL,updated_at=%s,work_version=work_version+1 WHERE id=%s",
                    (replacement_gate_id, f"Gate revoked: {correction_reason}", now, now, item["id"]),
                )
                cur.execute(
                    "UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s "
                    "WHERE current_work_item_id=%s",
                    (now, item["id"]),
                )
            self._event(cur, gate["project_id"], "SCIENTIFIC_GATE_AUTHORITY_REVOKED", gate_id, {
                "replacement_gate_id": replacement_gate_id,
                "correction_reason": correction_reason,
                "old_canonical_subject_version": old_version,
                "replacement_canonical_subject_version": replacement_version,
                "retired_work_item_ids": retired_ids,
                "rationale": rationale.strip()[:4000],
            })
            self._outbox(cur, gate["project_id"], "SCIENTIFIC_GATE_AUTHORITY_REVOKED", gate_id,
                         f"gate-revoked:{gate_id}:{replacement_gate_id}", {
                             "gate_id": gate_id, "replacement_gate_id": replacement_gate_id,
                             "correction_reason": correction_reason,
                         })
        changes = self.resolve_dependencies(str(gate["project_id"]))
        return {
            "gate_id": gate_id, "replacement_gate_id": replacement_gate_id, "status": "SUPERSEDED",
            "correction_reason": correction_reason, "retired_work_item_ids": retired_ids,
            "dependency_changes": changes,
        }

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
                    dependency_scope: str = "WORKITEM_LOCAL",
                    preferred_worker_id: str | None = None,
                    affinity_reason: str | None = None,
                    required_capabilities: list[str] | None = None,
                    authority_class: str | None = None,
                    promotion_state: str | None = None,
                    evidence_need: dict[str, Any] | None = None) -> dict:
        now, ident = self._now(), str(uuid.uuid4())
        dependencies = dependencies or []
        normalized_evidence_need = self._evidence_need_normalize(evidence_need) if evidence_need is not None else None
        authority_status = self._validate(authority_status, AUTHORITY_STATUSES, "LAB_WORK_AUTHORITY_STATUS")
        # v3.6.6 classifies review-like labor at the scheduling boundary.  The
        # caller must name the correction layer; generic code reading is not a
        # default successor to a passing deterministic engineering result.
        related_refs = dict(related_refs or {})
        role_upper = scientific_role.strip().upper()
        correction_layer = (
            "DESIGN" if role_upper in {"SCIENTIFIC_DESIGN_ATTACK", "DESIGN_ATTACK"}
            else "IMPLEMENTATION" if role_upper in {"IMPLEMENTATION_FIDELITY_CHECK", "IMPLEMENTATION_FIDELITY"}
            else "INTERPRETATION" if role_upper in {"RESULT_INTERPRETATION_ATTACK", "RESULT_INTERPRETATION"}
            else None
        )
        if correction_layer:
            existing_layer = related_refs.get("correction_layer")
            if existing_layer and str(existing_layer).upper() != correction_layer:
                raise GPUError("LAB_CORRECTION_LAYER_CONFLICT", str(existing_layer))
            related_refs["correction_layer"] = correction_layer
        if self.feature_flags.get("LEAN_ENGINEERING_REVIEW_V366") and role_upper in {"CODE_REVIEW", "GENERIC_CODE_REVIEW", "INDEPENDENT_CODE_REVIEW"}:
            if not related_refs.get("non_deterministic_science_critical_risk"):
                raise GPUError("LAB_GENERIC_REVIEW_SUPPRESSED", "Use deterministic validation or a bounded IMPLEMENTATION_FIDELITY_CHECK")
        authority_class = self._validate(
            authority_class or ("AUTHORITATIVE" if authority_status == "AUTHORITATIVE" else "SUPPORTING"),
            WORK_AUTHORITY_CLASSES, "LAB_WORK_AUTHORITY_CLASS",
        )
        promotion_state = self._validate(
            promotion_state or ("ELIGIBLE" if authority_class == "AUTHORITATIVE" else "NOT_APPLICABLE"),
            PROMOTION_STATES, "LAB_WORK_PROMOTION_STATE",
        )
        if promotion_state == "PROMOTED":
            # Promotion is a version-fenced transaction which checks typed
            # dependencies and a current PASS gate.  Creation must never be a
            # side door around it; legacy-corruption regressions use direct DB
            # fixtures so the auditor still sees historical bad states.
            raise GPUError("LAB_PROMOTION_ATOMIC_TRANSITION_REQUIRED", "use authority_promote")
        if authority_class in {"SPECULATIVE", "PREPARATORY"} and promotion_state in {"ELIGIBLE", "PROMOTED"}:
            raise GPUError("LAB_SPECULATIVE_PROMOTION_FORBIDDEN", "Speculative/preparatory work requires explicit rebase and promotion")
        dependency_scope = self._validate(dependency_scope, DEPENDENCY_SCOPES, "LAB_WORK_DEPENDENCY_SCOPE")
        speculation_class = self._validate(speculation_class, SPECULATION_CLASSES, "LAB_WORK_SPECULATION_CLASS")
        inferred_resource = "TRAINING" if kind.upper() == "TRAINING_RUN" else "GPU" if kind.upper() == "RUN_EXPERIMENT" else "REVIEW" if kind.upper() in {"REVIEW", "CORRECTION"} else "REASONING"
        resource_class = self._validate(resource_class or inferred_resource, RESOURCE_CLASSES, "LAB_WORK_RESOURCE_CLASS")
        speculation_condition = speculation_condition or {}
        if not isinstance(speculation_condition, dict):
            raise GPUError("LAB_WORK_SPECULATION_CONDITION_INVALID", "speculation_condition must be an object")
        if required_capabilities is not None:
            if not isinstance(required_capabilities, list) or any(not isinstance(item, str) or not item.strip() for item in required_capabilities):
                raise GPUError("LAB_WORK_REQUIRED_CAPABILITIES_INVALID", "required_capabilities must be a list of nonempty strings")
            existing_required = related_refs.get("required_capabilities")
            if existing_required is not None and set(existing_required) != set(required_capabilities):
                raise GPUError("LAB_WORK_REQUIRED_CAPABILITIES_CONFLICT", "related_refs.required_capabilities")
            related_refs["required_capabilities"] = sorted(set(required_capabilities))
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
            if normalized_evidence_need:
                cur.execute(
                    "SELECT n.*,w.status FROM lab_evidence_needs n JOIN lab_work_items w ON w.id=n.work_item_id "
                    "WHERE n.project_id=%s AND n.need_key=%s AND n.exception_key=%s FOR UPDATE",
                    (project_id, normalized_evidence_need["need_key"], normalized_evidence_need["exception_key"]),
                )
                existing_need = cur.fetchone()
                if existing_need and existing_need["status"] in ACTIVE_WORK_STATUSES:
                    existing_id = str(existing_need["work_item_id"])
                    self._event(cur, project_id, "EVIDENCE_NEED_WORK_REUSED", existing_id, {
                        "need_key": normalized_evidence_need["need_key"], "reason": "ACTIVE_EVIDENCE_NEED",
                    })
                    return {**self._work_record(cur, existing_id), "dedupe_result": "REUSED_ACTIVE"}
                if existing_need and existing_need["status"] == "COMPLETED":
                    existing_id = str(existing_need["work_item_id"])
                    self._event(cur, project_id, "EVIDENCE_NEED_ALREADY_SATISFIED", existing_id, {
                        "need_key": normalized_evidence_need["need_key"], "reason": "VALID_TERMINAL_INSPECTION",
                    })
                    return {**self._work_record(cur, existing_id), "dedupe_result": "ALREADY_SATISFIED"}
            if preferred_worker_id:
                self._worker(cur, preferred_worker_id)
                if not affinity_reason or not affinity_reason.strip():
                    raise GPUError("LAB_WORK_AFFINITY_REASON_REQUIRED", "affinity_reason")
            elif affinity_reason:
                raise GPUError("LAB_WORK_AFFINITY_WORKER_REQUIRED", "preferred_worker_id")
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
                    "created_by,parent_work_item_id,branch_id,related_refs,equivalence_key,authority_key,gate_id,canonical_subject_version,authority_status,subject_id,recovery_policy,canonical_objective_id,speculation_class,speculation_condition,resource_class,dependency_scope,preferred_worker_id,affinity_reason,authority_class,promotion_state,created_at,updated_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (ident, project_id, kind, title, description, scientific_role, status, priority, expected_value, estimated_cost,
                     created_by, parent_work_item_id, branch_id, json.dumps(related_refs or {}), equivalence_key, authority_key, gate_id,
                     canonical_subject_version, authority_status, subject_id, json.dumps(recovery_policy or {}),
                     canonical_objective_id, speculation_class, json.dumps(speculation_condition), resource_class,
                     dependency_scope, preferred_worker_id, affinity_reason.strip()[:1000] if affinity_reason else None,
                     authority_class, promotion_state, now, now),
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
            if normalized_evidence_need:
                cur.execute(
                    "INSERT INTO lab_evidence_needs(id,project_id,need_key,exception_key,empirical_origin,artifact_version,scientific_question,inspection_role,evidence_family_intent,analysis_version,exception_type,work_item_id,created_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT(project_id,need_key,exception_key) DO UPDATE SET work_item_id=EXCLUDED.work_item_id,created_at=EXCLUDED.created_at "
                    "WHERE (SELECT status FROM lab_work_items WHERE id=lab_evidence_needs.work_item_id) NOT IN ('COMPLETED','READY','CLAIMED','RUNNING','RUNNING_DETACHED','RESULT_READY','WAITING_DEPENDENCY','BLOCKED','REPLAN_REQUIRED','DORMANT')",
                    (str(uuid.uuid4()), project_id, normalized_evidence_need["need_key"], normalized_evidence_need["exception_key"],
                     normalized_evidence_need["empirical_origin"], normalized_evidence_need["artifact_version"],
                     normalized_evidence_need["scientific_question"], normalized_evidence_need["inspection_role"],
                     normalized_evidence_need["evidence_family_intent"], normalized_evidence_need["analysis_version"],
                     normalized_evidence_need["exception_type"], ident, now),
                )
                self._event(cur, project_id, "EVIDENCE_NEED_BOUND", ident, {
                    "need_key": normalized_evidence_need["need_key"],
                    "exception_type": normalized_evidence_need["exception_type"],
                    "exception_key": normalized_evidence_need["exception_key"],
                })
            if gate_id and authority_status == "AUTHORITATIVE":
                cur.execute("UPDATE scientific_gates SET authoritative_work_item_id=%s WHERE id=%s", (ident, gate_id))
            self._event(cur, project_id, "WORK_ITEM_CREATED", ident, {
                "kind": kind, "title": title, "status": status, "gate_id": gate_id,
                "authority_key": authority_key, "authority_status": authority_status,
                "canonical_objective_id": canonical_objective_id, "dependency_scope": dependency_scope,
                "branch_id": branch_id, "equivalence_key": equivalence_key,
                "resource_class": resource_class, "speculation_class": speculation_class,
                "authority_class": authority_class, "promotion_state": promotion_state,
                "preferred_worker_id": preferred_worker_id,
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
            "SELECT target_type,target_id,required_statuses,invalidating_statuses,description,dependency_scope,predicate_type,predicate_value "
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

    @staticmethod
    def _validate_frozen_execution_authority_records(
        project_id: str, experiment_id: str, work_item_id: str, worker_id: str, session_id: str,
        item: dict[str, Any], gate: dict[str, Any], experiment: dict[str, Any], hypothesis: dict[str, Any],
        dependency_outcomes: list[tuple[bool, bool, str]], lease_active: bool,
        attempts: list[dict[str, Any]], execution_attempt_uuid: str | None = None,
        detached_retry: bool = False, require_promoted_authority: bool = False,
    ) -> dict[str, Any]:
        """Validate current execution authority without consulting discovery ranking."""
        if str(item.get("project_id")) != str(project_id) or str(experiment.get("project_id")) != str(project_id):
            raise GPUError("LAB_PROJECT_MISMATCH", work_item_id)
        if str(hypothesis.get("project_id")) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", "Experiment and hypothesis differ")
        refs = item.get("related_refs") or {}
        if detached_retry:
            if item.get("status") != "RUNNING_DETACHED":
                raise GPUError("LAB_EXECUTION_RETRY_NOT_DETACHED", f"WorkItem is {item.get('status')}")
            if item.get("assigned_worker_id") is not None or item.get("assigned_session_id") is not None or item.get("lease_id") is not None:
                raise GPUError("LAB_EXECUTION_RETRY_STATE_INVALID", "Detached WorkItem still carries live ownership")
            if str(refs.get("execution_owner_worker_id")) != str(worker_id) or str(refs.get("execution_owner_session_id")) != str(session_id):
                raise GPUError("LAB_WORK_OWNERSHIP_MISMATCH", work_item_id)
            if not execution_attempt_uuid or str(refs.get("execution_attempt_uuid")) != str(execution_attempt_uuid):
                raise GPUError("EXPERIMENT_ONE_SHOT_ALREADY_RESERVED", experiment_id)
            if len(attempts) != 1:
                raise GPUError("EXPERIMENT_ONE_SHOT_ALREADY_RESERVED", experiment_id)
            attempt = attempts[0]
            if (
                str(attempt.get("idempotency_key")) != str(execution_attempt_uuid)
                or str(attempt.get("run_id")) != str(refs.get("experiment_run_id"))
                or str(attempt.get("request_fingerprint")) != str(refs.get("execution_request_fingerprint"))
                or str(attempt.get("status")) != "RESERVED"
            ):
                raise GPUError("EXPERIMENT_DETACHED_RETRY_NOT_RESERVED", experiment_id)
        else:
            if item.get("status") not in {"CLAIMED", "RUNNING"}:
                raise GPUError("LAB_EXECUTION_WORK_NOT_OWNED", f"WorkItem is {item.get('status')}")
            if str(item.get("assigned_worker_id")) != str(worker_id) or str(item.get("assigned_session_id")) != str(session_id):
                raise GPUError("LAB_WORK_OWNERSHIP_MISMATCH", work_item_id)
        if item.get("authority_status") != "AUTHORITATIVE" or item.get("superseded_by") or item.get("invalidated_at"):
            raise GPUError("LAB_EXECUTION_AUTHORITY_INVALID", "WorkItem is not current authoritative work")
        if require_promoted_authority and item.get("promotion_state") != "PROMOTED":
            raise GPUError("LAB_EXECUTION_AUTHORITY_NOT_PROMOTED", str(item.get("promotion_state")))
        if str(item.get("subject_id")) != str(experiment_id):
            raise GPUError("LAB_EXECUTION_AUTHORITY_MISMATCH", "WorkItem subject differs from Experiment")
        if not item.get("gate_id") or not item.get("canonical_subject_version"):
            raise GPUError("LAB_EXECUTION_AUTHORITY_INCOMPLETE", "WorkItem lacks gate/subject binding")
        if any(invalidated for _, invalidated, _ in dependency_outcomes):
            raise GPUError("LAB_WORK_DEPENDENCY_INVALIDATED", "; ".join(d for _, i, d in dependency_outcomes if i))
        if any(not satisfied for satisfied, _, _ in dependency_outcomes):
            raise GPUError("LAB_WORK_DEPENDENCY_UNSATISFIED", "; ".join(d for s, _, d in dependency_outcomes if not s))
        if not detached_retry and not lease_active:
            raise GPUError("LAB_LEASE_NOT_ACTIVE", work_item_id)
        if gate.get("status") != "PASS" or gate.get("superseded_by") or gate.get("invalidation_reason"):
            raise GPUError("SCIENTIFIC_GATE_NOT_EXECUTABLE", str(gate.get("status")))
        if str(gate.get("project_id")) != str(project_id) or str(gate.get("scientific_object_id")) != str(experiment_id):
            raise GPUError("SCIENTIFIC_GATE_AUTHORITY_MISMATCH", str(gate.get("id")))
        if str(gate.get("authoritative_work_item_id")) != str(work_item_id):
            raise GPUError("SCIENTIFIC_GATE_AUTHORITY_MISMATCH", "Gate points to another WorkItem")
        if str(gate.get("canonical_subject_version")) != str(item.get("canonical_subject_version")):
            raise GPUError("SCIENTIFIC_GATE_SUBJECT_MISMATCH", str(gate.get("id")))
        if str(item.get("gate_id")) != str(gate.get("id")):
            raise GPUError("SCIENTIFIC_GATE_AUTHORITY_MISMATCH", "WorkItem points to another gate")
        if experiment.get("kind") != "Experiment" or experiment.get("status") != "ACTIVE" or experiment.get("data", {}).get("frozen") is not True:
            raise GPUError("EXPERIMENT_NOT_EXECUTABLE", str(experiment.get("status")))
        hypothesis_id = str(experiment.get("data", {}).get("hypothesis_id") or "")
        if not hypothesis_id or hypothesis.get("kind") != "Hypothesis" or str(hypothesis.get("id")) != hypothesis_id:
            raise GPUError("EXPERIMENT_PLAN_INCOMPLETE", "Experiment/Hypothesis binding is invalid")
        recovery = item.get("recovery_policy") or {}
        one_shot = bool(recovery.get("one_shot") or recovery.get("requires_no_existing_run"))
        if one_shot and attempts:
            def pre_submission_technical_cancellation(row: dict[str, Any]) -> bool:
                abandonment = (row.get("run_data") or {}).get("legacy_abandonment") or {}
                provenance = abandonment.get("provenance") or {}
                return (
                    row.get("status") == "cancelled"
                    and row.get("run_status") == "cancelled"
                    and abandonment.get("verified_missing_backing_job") is True
                    and provenance.get("technical_non_scientific") is True
                )

            blocking_attempts = [row for row in attempts if not pre_submission_technical_cancellation(row)]
            keys = {str(row.get("idempotency_key")) for row in blocking_attempts}
            if blocking_attempts and (
                not execution_attempt_uuid or keys != {str(execution_attempt_uuid)} or len(blocking_attempts) != 1
            ):
                raise GPUError("EXPERIMENT_ONE_SHOT_ALREADY_RESERVED", experiment_id)
        return {
            "validated": True,
            "project_id": str(project_id),
            "experiment_id": str(experiment_id),
            "hypothesis_id": hypothesis_id,
            "gate_id": str(gate["id"]),
            "work_item_id": str(work_item_id),
            "canonical_subject_version": str(item["canonical_subject_version"]),
            "authority_key": item.get("authority_key"),
            "one_shot": one_shot,
            "execution_attempt_uuid": str(execution_attempt_uuid) if execution_attempt_uuid else None,
            "detached_retry": detached_retry,
        }

    def frozen_execution_authority_validate(
        self, project_id: str, experiment_id: str, work_item_id: str, worker_id: str, session_id: str,
        execution_attempt_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Validate a current gated frozen-experiment execution owner, fail closed."""
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            self._worker(cur, worker_id)
            session = self._session(cur, session_id, worker_id, project_id)
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s AND project_id=%s FOR UPDATE", (work_item_id, project_id))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            detached_retry = item.get("status") == "RUNNING_DETACHED"
            if not detached_retry and str(session.get("current_work_item_id")) != str(work_item_id):
                raise GPUError("LAB_WORK_OWNERSHIP_MISMATCH", work_item_id)
            cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s", (work_item_id,))
            dependencies = cur.fetchall()
            outcomes = [self._dependency_status(cur, project_id, dep) for dep in dependencies]
            cur.execute("SELECT * FROM scientific_gates WHERE id=%s AND project_id=%s FOR SHARE", (item.get("gate_id"), project_id))
            gate = cur.fetchone()
            if not gate:
                raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", str(item.get("gate_id")))
            cur.execute("SELECT * FROM research_objects WHERE id=%s AND project_id=%s", (experiment_id, project_id))
            experiment = cur.fetchone()
            if not experiment:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", experiment_id)
            hypothesis_id = str((experiment.get("data") or {}).get("hypothesis_id") or "")
            cur.execute("SELECT * FROM research_objects WHERE id=%s AND project_id=%s", (hypothesis_id, project_id))
            hypothesis = cur.fetchone()
            if not hypothesis:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", hypothesis_id)
            cur.execute(
                "SELECT id FROM lab_work_leases WHERE id=%s AND work_item_id=%s AND worker_id=%s "
                "AND worker_session_id=%s AND released_at IS NULL AND expires_at>=%s",
                (item.get("lease_id"), work_item_id, worker_id, session_id, now),
            )
            lease_active = cur.fetchone() is not None
            cur.execute(
                "SELECT a.run_id,a.job_id,a.idempotency_key,a.request_fingerprint,a.status,"
                "r.status AS run_status,r.data AS run_data "
                "FROM research_execution_attempts a JOIN research_objects r ON r.id=a.run_id "
                "WHERE a.experiment_id=%s ORDER BY a.created_at,a.run_id",
                (experiment_id,),
            )
            attempts = cur.fetchall()
            return self._validate_frozen_execution_authority_records(
                project_id, experiment_id, work_item_id, worker_id, session_id,
                self._record(item) or {}, self._record(gate) or {}, dict(experiment),
                dict(hypothesis), outcomes, lease_active,
                [dict(row) for row in attempts], execution_attempt_uuid, detached_retry,
                bool(self.feature_flags.get("AUTHORITY_PROMOTION_V365")),
            )

    def recover_unsubmitted_execution_authority(
        self, work_item_id: str, worker_id: str, session_id: str,
        replacement_attempt_uuid: str | None,
    ) -> dict[str, Any]:
        """Restore a detached execution WorkItem after a proven non-dispatch.

        A canonical run is normally detached from a renewable worker lease so
        only its exact reserved UUID can be retried.  If that reservation is
        terminally cancelled with server evidence that no process ever existed,
        retaining the detached pointer turns bookkeeping into a permanent
        one-shot deadlock.  Only the same recorded owner may reclaim it here;
        the subsequent frozen-authority validation still enforces the gate,
        promoted authority, dependencies, and any non-recoverable attempts.
        """
        if not replacement_attempt_uuid:
            return {"recovered": False, "reason": "NO_REPLACEMENT_ATTEMPT"}
        now = self._now()
        expiry = now + timedelta(seconds=self.lease_seconds)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            self._session(cur, session_id, worker_id, str(item["project_id"]))
            if item["status"] != "RUNNING_DETACHED":
                return {"recovered": False, "reason": "NOT_DETACHED", "status": item["status"]}
            refs = item.get("related_refs") or {}
            old_attempt_uuid = str(refs.get("execution_attempt_uuid") or "")
            if str(replacement_attempt_uuid) == old_attempt_uuid:
                return {"recovered": False, "reason": "EXACT_RETRY", "status": item["status"]}
            if (
                str(refs.get("execution_owner_worker_id") or "") != str(worker_id)
                or str(refs.get("execution_owner_session_id") or "") != str(session_id)
            ):
                raise GPUError("LAB_WORK_OWNERSHIP_MISMATCH", work_item_id)
            old_run_id = refs.get("experiment_run_id")
            if not old_run_id:
                raise GPUError("LAB_EXECUTION_RECOVERY_RUN_MISSING", work_item_id)
            cur.execute(
                "SELECT r.status AS run_status,r.data AS run_data,a.status AS attempt_status "
                "FROM research_objects r JOIN research_execution_attempts a ON a.run_id=r.id "
                "WHERE r.id=%s FOR UPDATE",
                (old_run_id,),
            )
            run = cur.fetchone()
            abandonment = ((run or {}).get("run_data") or {}).get("legacy_abandonment") or {}
            provenance = abandonment.get("provenance") or {}
            recoverable = (
                run is not None
                and run.get("run_status") == "cancelled"
                and run.get("attempt_status") == "cancelled"
                and abandonment.get("verified_missing_backing_job") is True
                and provenance.get("technical_non_scientific") is True
            )
            if not recoverable:
                raise GPUError("LAB_EXECUTION_RECOVERY_NOT_UNSUBMITTED", str(old_run_id))
            lease_id = str(uuid.uuid4())
            historical_ref = {
                "run_id": str(old_run_id), "attempt_uuid": old_attempt_uuid,
                "reason": "VERIFIED_UNSUBMITTED_TECHNICAL_CANCELLATION",
                "recovered_at": now.isoformat(),
            }
            updated_refs = {
                **refs,
                "unsubmitted_execution_recovery": historical_ref,
            }
            for key in (
                "experiment_run_id", "execution_attempt_uuid", "execution_request_fingerprint",
                "execution_owner_worker_id", "execution_owner_session_id",
            ):
                updated_refs.pop(key, None)
            cur.execute(
                "UPDATE lab_work_items SET status='RUNNING',assigned_worker_id=%s,assigned_session_id=%s,"
                "lease_id=%s,related_refs=%s,blocked_reason=NULL,updated_at=%s,work_version=work_version+1 "
                "WHERE id=%s",
                (worker_id, session_id, lease_id, json.dumps(updated_refs), now, work_item_id),
            )
            cur.execute(
                "INSERT INTO lab_work_leases(id,work_item_id,worker_id,worker_session_id,acquired_at,heartbeat_at,expires_at,lease_version) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,1)",
                (lease_id, work_item_id, worker_id, session_id, now, now, expiry),
            )
            cur.execute(
                "UPDATE research_worker_sessions SET current_work_item_id=%s,active_role=%s,status='BUSY',last_heartbeat_at=%s WHERE id=%s",
                (work_item_id, item["scientific_role"], now, session_id),
            )
            cur.execute(
                "UPDATE research_workers SET availability_state='EXECUTING',availability_updated_at=%s,idle_reason=NULL,idle_since=NULL WHERE id=%s",
                (now, worker_id),
            )
            self._event(cur, item["project_id"], "WORK_ITEM_UNSUBMITTED_EXECUTION_RECOVERED", work_item_id, {
                "previous_run_id": str(old_run_id), "previous_attempt_uuid": old_attempt_uuid,
                "replacement_attempt_uuid": str(replacement_attempt_uuid), "worker_id": str(worker_id),
            })
        return {"recovered": True, "work_item_id": str(work_item_id), "lease_id": lease_id,
                "replacement_attempt_uuid": str(replacement_attempt_uuid)}

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

    def _discovery_work_round_is_current(self, item: dict) -> bool:
        """Discovery work is schedulable only while its frozen round is live."""
        if item.get("kind") not in {"DISCOVERY_GENERATION", "DISCOVERY_SYNTHESIS"}:
            return True
        refs = item.get("related_refs") or {}
        round_id = str(refs.get("discovery_round_id") or "")
        if not round_id:
            return False
        try:
            round_ = self.store.object_get(round_id)
        except GPUError:
            return False
        data = round_.get("data") or {}
        if (
            str(round_.get("project_id")) != str(item.get("project_id"))
            or round_.get("kind") != "DiscoveryRound"
            or round_.get("status") != "ACTIVE"
            or bool(data.get("stale"))
        ):
            return False
        phase = data.get("phase")
        if item.get("kind") == "DISCOVERY_GENERATION":
            return phase == "INDEPENDENT_GENERATION"
        return phase in {"GENERATION_FROZEN", "COVERAGE_CONTINUED"}

    def _unversioned_review_has_bound_successor(self, item: dict) -> bool:
        """Keep a legacy review from pre-empting a bound current review."""
        if (
            item.get("resource_class") != "REVIEW"
            or item.get("canonical_subject_version")
            or not item.get("subject_id")
        ):
            return False
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM lab_work_items WHERE project_id=%s AND subject_id=%s "
                "AND id<>%s AND status=ANY(%s) AND canonical_subject_version IS NOT NULL "
                "AND superseded_by IS NULL AND invalidated_at IS NULL LIMIT 1",
                (item["project_id"], item["subject_id"], item["id"], list(ACTIVE_WORK_STATUSES)),
            )
            return bool(cur.fetchone())

    def _scheduler_actionable_work(self, item: dict) -> bool:
        """Return whether a READY item is safe for portfolio assignment.

        ``AUTHORITATIVE`` remains mandatory for execution authority. It is not
        required to assign outcome-independent design, analysis, or review
        work. Treating every ``SUPPORTING`` item as invisible made a
        materialized PRE_SCIENCE_DESIGN_FREEZE discoverable by equivalence
        lookup but impossible for the scheduler to assign.
        """
        if item.get("status") != "READY" or item.get("superseded_by") or item.get("invalidated_at"):
            return False
        if not self._discovery_work_round_is_current(item):
            return False
        if self._unversioned_review_has_bound_successor(item):
            return False
        if item.get("authority_status") == "AUTHORITATIVE":
            return True
        if item.get("authority_status") != "SUPPORTING":
            return False
        # Supporting work is normally restricted to outcome-independent
        # reasoning/review. Independent discovery generation is also
        # outcome-independent: it is bound to a frozen DiscoveryRound and
        # claim_work re-validates/rebinds its batch atomically. Hiding this
        # kind leaves a valid fresh lane stranded after an expired claimant.
        if item.get("kind") == "DISCOVERY_GENERATION":
            return item.get("resource_class") == "REASONING"
        return (
            item.get("resource_class") in {"REASONING", "REVIEW"}
            and item.get("kind") not in {"RUN_EXPERIMENT", "TRAINING_RUN"}
        )

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
        predicate_type = str(dependency.get("predicate_type") or "STATUS_IN").upper()
        predicate_value = dependency.get("predicate_value") or dependency.get("required_statuses") or []
        if predicate_type in {"REVIEW_VERDICT_EQUALS", "REVIEW_VERDICT_IN"}:
            if target_type != "REVIEW_RESULT":
                return False, True, "review-verdict predicate requires REVIEW_RESULT"
            expected = [str(item).upper() for item in (predicate_value if isinstance(predicate_value, list) else [predicate_value])]
            cur.execute(
                "SELECT verdict FROM lab_review_results WHERE id=%s AND project_id=%s",
                (target_id, project_id),
            )
            row = cur.fetchone()
            if not row:
                return False, False, f"review result {target_id} is unavailable"
            verdict = str(row["verdict"]).upper()
            return verdict in expected, verdict in {"REJECT", "INVALID_REVIEW"}, f"review result {target_id} verdict is {verdict}"
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
        required = predicate_value
        return (not required or status in required), False, f"dependency {target_id} is {status}"

    def _branch_dependency_status(self, cur, project_id: str, dependency: dict) -> tuple[bool, bool, str]:
        """Evaluate typed dependencies and legacy branch metadata safely."""
        if dependency.get("target_type") and dependency.get("target_id"):
            return self._dependency_status(
                cur, project_id, {"required_statuses": [], "invalidating_statuses": [], **dependency}
            )
        legacy_status = str(dependency.get("status", "")).upper()
        legacy_type = str(dependency.get("type", "LEGACY_BRANCH_DEPENDENCY"))
        if legacy_status in {"SATISFIED", "COMPLETED", "PASS"}:
            return True, False, f"legacy branch dependency {legacy_type} is {legacy_status}"
        if legacy_status in {"INVALID", "INVALIDATED", "SUPERSEDED"}:
            return False, True, f"legacy branch dependency {legacy_type} is {legacy_status}"
        return False, False, f"legacy branch dependency {legacy_type} is {legacy_status or 'UNRESOLVED'}"

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
            if target_type not in {"WORK_ITEM", "SCIENTIFIC_GATE", "RESEARCH_OBJECT", "EXPERIMENT_RUN", "ENGINEERING_RESULT", "RESEARCH_DECISION", "EVIDENCE_UNIT", "ARTIFACT", "REVIEW_RESULT"}:
                raise GPUError("LAB_DEPENDENCY_TYPE_INVALID", target_type)
            predicate_type = self._validate(
                str(dependency.get("predicate_type", "STATUS_IN")), DEPENDENCY_PREDICATES,
                "LAB_DEPENDENCY_PREDICATE",
            )
            predicate_value = dependency.get("predicate_value", dependency.get("required_statuses", []))
            if predicate_type == "REVIEW_VERDICT_EQUALS":
                predicate_value = [predicate_value] if isinstance(predicate_value, str) else predicate_value
                if not isinstance(predicate_value, list) or len(predicate_value) != 1:
                    raise GPUError("LAB_REVIEW_VERDICT_PREDICATE_INVALID", target_id)
            if predicate_type == "REVIEW_VERDICT_IN" and not isinstance(predicate_value, list):
                raise GPUError("LAB_REVIEW_VERDICT_PREDICATE_INVALID", target_id)
            if predicate_type.startswith("REVIEW_VERDICT"):
                if target_type != "REVIEW_RESULT" or any(str(item).upper() not in REVIEW_VERDICTS for item in predicate_value):
                    raise GPUError("LAB_REVIEW_VERDICT_PREDICATE_INVALID", target_id)
            required_statuses = dependency.get("required_statuses", [])
            exists_only = dependency.get("allow_exists_only", False)
            if not isinstance(required_statuses, list) or any(not isinstance(status, str) or not status.strip() for status in required_statuses):
                raise GPUError("LAB_DEPENDENCY_REQUIRED_STATUS_INVALID", target_id)
            if predicate_type == "STATUS_IN" and not required_statuses and exists_only is not True:
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
                "INSERT INTO lab_work_dependencies(id,work_item_id,target_type,target_id,required_statuses,invalidating_statuses,description,dependency_scope,predicate_type,predicate_value,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (str(uuid.uuid4()), work_item_id, target_type, target_id, json.dumps(required_statuses),
                 json.dumps(dependency.get("invalidating_statuses", [])), str(dependency.get("description", "")), scope,
                 predicate_type, json.dumps(predicate_value), now),
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
            worker = self._worker(cur, worker_id)
            # Serialize all claims for one worker identity before taking work or
            # session row locks; this also avoids cross-session lock inversion.
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"lab-worker-claim:{worker_id}",))
            # Budget and portfolio policy use structured branch/resource/speculation
            # fields.  Claiming a partial row would silently bypass newer limits.
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            target = cur.fetchone()
            if not target:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            required_capabilities = self._required_capability_names(target)
            missing_capabilities = sorted(required_capabilities - self._capability_names(worker.get("capabilities")))
            if missing_capabilities:
                raise GPUError("LAB_WORKER_CAPABILITY_REQUIRED", ", ".join(missing_capabilities))
            session = self._session(cur, session_id, worker_id, str(target["project_id"]))
            self._clear_stale_session_ownership(cur, worker_id, now)
            # The reconciliation can clear this session's stale projection;
            # reload the locked row before applying the ownership guard.
            session = self._session(cur, session_id, worker_id, str(target["project_id"]))
            if str(session["current_project_id"]) != str(target["project_id"]):
                raise GPUError("LAB_PROJECT_MISMATCH", work_item_id)
            # An independent DiscoveryBatch is cryptographically/provenance
            # bound to its original generator session.  Reassigning only the
            # operational WorkItem would otherwise leave the new worker unable
            # to submit (or, worse, let it inherit another worker's lane).
            # Replace the expired lane atomically before the new claim.  The
            # old batch remains immutable historical record and is excluded
            # from synthesis; the new batch is a fresh independent lane.
            target = self._fresh_discovery_batch_for_reassigned_claim(
                cur, target, worker_id, session_id, now
            )
            # `_session(... FOR UPDATE)` serializes competing schedulers for one
            # runtime, but the second caller must not be allowed to replace the
            # first caller's fresh assignment after it obtains that lock.
            if session["current_work_item_id"] and str(session["current_work_item_id"]) != str(work_item_id):
                raise GPUError("LAB_WORKER_ALREADY_ASSIGNED", str(session["current_work_item_id"]))
            # Worker capacity belongs to the identity, not to a browser/session
            # row. Without this guard, the same ChatGPT/Codex worker could be
            # scheduled concurrently through two project sessions.
            cur.execute(
                "SELECT current_work_item_id FROM research_worker_sessions WHERE worker_id=%s AND id<>%s "
                "AND current_work_item_id IS NOT NULL AND status IN ('ACTIVE','BUSY','WAITING')",
                (worker_id, session_id),
            )
            other_assignment = cur.fetchone()
            if other_assignment:
                raise GPUError("LAB_WORKER_ALREADY_ASSIGNED", str(other_assignment["current_work_item_id"]))
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
        self, work_item_id: str, worker_id: str, session_id: str, run_id: str,
        expected_execution_authority: dict[str, Any] | None = None,
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
            refs = item.get("related_refs") or {}
            detached_retry = item["status"] == "RUNNING_DETACHED"
            if detached_retry:
                if expected_execution_authority is None or expected_execution_authority.get("detached_retry") is not True:
                    raise GPUError("LAB_WORK_NOT_ATTACHABLE", item["status"])
                if str(refs.get("execution_owner_worker_id")) != str(worker_id) or str(refs.get("execution_owner_session_id")) != str(session_id):
                    raise GPUError("LAB_WORK_NOT_OWNED", work_item_id)
                if str(refs.get("experiment_run_id")) != str(run_id):
                    raise GPUError("FROZEN_EXECUTION_AUTHORITY_CHANGED", "Detached retry points to another run")
                if str(refs.get("execution_attempt_uuid")) != str(expected_execution_authority.get("execution_attempt_uuid")):
                    raise GPUError("FROZEN_EXECUTION_AUTHORITY_CHANGED", "Detached retry points to another attempt")
            else:
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
            if expected_execution_authority is not None:
                if expected_execution_authority.get("validated") is not True:
                    raise GPUError("FROZEN_EXECUTION_AUTHORITY_INVALID", "Attachment authority was not validated")
                if item.get("authority_status") != "AUTHORITATIVE" or item.get("superseded_by") or item.get("invalidated_at"):
                    raise GPUError("LAB_EXECUTION_AUTHORITY_INVALID", "WorkItem changed before run attachment")
                for field, observed in (
                    ("work_item_id", item.get("id")),
                    ("gate_id", item.get("gate_id")),
                    ("canonical_subject_version", item.get("canonical_subject_version")),
                    ("experiment_id", item.get("subject_id")),
                ):
                    if str(expected_execution_authority.get(field)) != str(observed):
                        raise GPUError("FROZEN_EXECUTION_AUTHORITY_CHANGED", f"Authority field {field} changed before attachment")
                if str((run.get("data") or {}).get("experiment_id")) != str(item.get("subject_id")):
                    raise GPUError("LAB_EXECUTION_AUTHORITY_MISMATCH", "Reserved run belongs to another Experiment")
                cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s", (work_item_id,))
                outcomes = [self._dependency_status(cur, str(item["project_id"]), dep) for dep in cur.fetchall()]
                if any(invalidated for _, invalidated, _ in outcomes):
                    raise GPUError("LAB_WORK_DEPENDENCY_INVALIDATED", "; ".join(d for _, i, d in outcomes if i))
                if any(not satisfied for satisfied, _, _ in outcomes):
                    raise GPUError("LAB_WORK_DEPENDENCY_UNSATISFIED", "; ".join(d for s, _, d in outcomes if not s))
                cur.execute("SELECT * FROM scientific_gates WHERE id=%s AND project_id=%s FOR SHARE", (item.get("gate_id"), item["project_id"]))
                current_gate = cur.fetchone()
                if (
                    not current_gate
                    or current_gate.get("status") != "PASS"
                    or current_gate.get("superseded_by")
                    or current_gate.get("invalidation_reason")
                    or str(current_gate.get("authoritative_work_item_id")) != str(work_item_id)
                    or str(current_gate.get("canonical_subject_version")) != str(item.get("canonical_subject_version"))
                ):
                    raise GPUError("SCIENTIFIC_GATE_NOT_EXECUTABLE", str(item.get("gate_id")))
            if detached_retry:
                # Exact recovery is idempotent: keep the WorkItem detached and do
                # not recreate a lease or ownership.  The server may now retry
                # submission of this same RESERVED canonical attempt only.
                return self._record(item) or {}
            refs = {
                **(item["related_refs"] or {}),
                "experiment_run_id": str(run_id),
                "execution_owner_worker_id": str(worker_id),
                "execution_owner_session_id": str(session_id),
                "execution_attempt_uuid": (expected_execution_authority or {}).get("execution_attempt_uuid"),
                "execution_request_fingerprint": (run.get("data") or {}).get("request_fingerprint"),
                "execution_authority_gate_id": str(item.get("gate_id")) if item.get("gate_id") is not None else None,
                "execution_authority_subject_version": str(item.get("canonical_subject_version")) if item.get("canonical_subject_version") is not None else None,
            }
            cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='EXPERIMENT_RUN_ATTACHED' WHERE id=%s AND released_at IS NULL", (now, item["lease_id"]))
            cur.execute("UPDATE lab_work_items SET status='RUNNING_DETACHED',related_refs=%s,assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,blocked_reason=%s,updated_at=%s WHERE id=%s", (json.dumps(refs), "Canonical ExperimentRun is executing or reserved for exact retry; this WorkItem cannot be claimed.", now, work_item_id))
            cur.execute("UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s WHERE id=%s", (now, session_id))
            cur.execute(
                "UPDATE research_workers w SET availability_state='AVAILABLE',availability_updated_at=%s,"
                "idle_reason=NULL,idle_since=NULL WHERE w.id=%s AND NOT EXISTS ("
                "SELECT 1 FROM research_worker_sessions s WHERE s.worker_id=w.id "
                "AND s.current_work_item_id IS NOT NULL AND s.status IN ('ACTIVE','BUSY','WAITING'))",
                (now, worker_id),
            )
            self._event(cur, item["project_id"], "WORK_ITEM_EXPERIMENT_ATTACHED", work_item_id, {"run_id": run_id, "worker_id": worker_id})
        return self.work_get(work_item_id)

    def review_result_record(
        self, project_id: str, review_work_item_id: str, target_subject_id: str,
        target_subject_version: str, review_type: str, verdict: str,
        findings: list[dict[str, Any]] | None = None, materiality: str | None = None,
        reviewed_hashes: dict[str, str] | None = None, lineage_snapshot_id: str | None = None,
    ) -> dict:
        """Persist a review verdict independently of its operational WorkItem.

        A completed reviewer task says only that reviewing ended.  Every
        promotion/certification dependency must target this immutable verdict.
        """
        verdict = self._validate(verdict, REVIEW_VERDICTS, "REVIEW_VERDICT")
        if not target_subject_id or not target_subject_version or not review_type.strip():
            raise GPUError("LAB_REVIEW_RESULT_BINDING_REQUIRED", "subject id, version, and review type")
        if not isinstance(findings or [], list) or not isinstance(reviewed_hashes or {}, dict):
            raise GPUError("LAB_REVIEW_RESULT_FIELDS_INVALID", "findings and reviewed_hashes")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT project_id,status,authority_class FROM lab_work_items WHERE id=%s FOR UPDATE", (review_work_item_id,))
            work = cur.fetchone()
            if not work or str(work["project_id"]) != str(project_id):
                raise GPUError("LAB_REVIEW_WORK_NOT_FOUND", review_work_item_id)
            if self.feature_flags.get("VERDICT_AWARE_DEPENDENCIES_V365"):
                if work["authority_class"] != "INDEPENDENT_REVIEW":
                    raise GPUError("LAB_REVIEW_RESULT_REVIEWER_CLASS_REQUIRED", review_work_item_id)
                if work["status"] != "COMPLETED":
                    raise GPUError("LAB_REVIEW_RESULT_WORK_NOT_COMPLETED", review_work_item_id)
            cur.execute(
                "SELECT * FROM lab_review_results WHERE project_id=%s AND review_work_item_id=%s AND review_type=%s FOR UPDATE",
                (project_id, review_work_item_id, review_type.strip()),
            )
            result = cur.fetchone()
            immutable_payload = {
                "target_subject_id": target_subject_id, "target_subject_version": target_subject_version,
                "verdict": verdict, "findings": findings or [], "materiality": materiality,
                "reviewed_hashes": reviewed_hashes or {}, "lineage_snapshot_id": lineage_snapshot_id,
            }
            if result:
                existing_payload = {
                    "target_subject_id": result["target_subject_id"], "target_subject_version": result["target_subject_version"],
                    "verdict": result["verdict"], "findings": result["findings"] or [], "materiality": result["materiality"],
                    "reviewed_hashes": result["reviewed_hashes"] or {}, "lineage_snapshot_id": result["lineage_snapshot_id"],
                }
                if existing_payload != immutable_payload:
                    raise GPUError("LAB_REVIEW_RESULT_IMMUTABLE", review_work_item_id)
            else:
                cur.execute(
                    "INSERT INTO lab_review_results(id,project_id,review_work_item_id,target_subject_id,target_subject_version,review_type,verdict,findings,materiality,reviewed_hashes,lineage_snapshot_id,created_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *",
                    (ident, project_id, review_work_item_id, target_subject_id, target_subject_version, review_type.strip(), verdict,
                     json.dumps(findings or []), materiality, json.dumps(reviewed_hashes or {}), lineage_snapshot_id, now),
                )
                result = cur.fetchone()
            self._event(cur, project_id, "REVIEW_RESULT_RECORDED", str(result["id"]), {
                "review_work_item_id": review_work_item_id, "review_type": review_type.strip(),
                "verdict": verdict, "target_subject_id": target_subject_id,
                "target_subject_version": target_subject_version,
            })
        self.resolve_dependencies(project_id)
        return self._record(result) or {}

    def work_permissions_get(self, work_item_id: str) -> dict[str, Any]:
        """Derive start, promotion, execution, and evidence permissions from facts."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s", (work_item_id,))
            outcomes = [self._dependency_status(cur, str(item["project_id"]), dep) for dep in cur.fetchall()]
            all_dependencies = all(satisfied for satisfied, _, _ in outcomes)
            blocking = [detail for satisfied, _, detail in outcomes if not satisfied]
            authority_class = item["authority_class"]
            promotion_state = item["promotion_state"]
            may_start = item["status"] == "READY" and not item.get("superseded_by") and not item.get("invalidated_at")
            # Preparatory/speculative labor may start with their own bounded
            # contract; unresolved authority is not a hidden work-start gate.
            may_promote = (
                authority_class == "AUTHORITATIVE" and promotion_state in {"ELIGIBLE", "CONDITIONALLY_ELIGIBLE"}
                and all_dependencies and not item.get("superseded_by") and not item.get("invalidated_at")
            )
            gate_pass = False
            if item.get("gate_id"):
                cur.execute(
                    "SELECT status,superseded_by,invalidation_reason FROM scientific_gates WHERE id=%s AND project_id=%s",
                    (item["gate_id"], item["project_id"]),
                )
                gate = cur.fetchone()
                gate_pass = bool(gate and gate["status"] == "PASS" and not gate["superseded_by"] and not gate["invalidation_reason"])
                if not gate_pass:
                    blocking.append("scientific gate is not a current PASS")
            else:
                blocking.append("work item has no scientific gate")
            # Eligibility says an atomic promotion may be requested.  It is
            # deliberately insufficient for an execution permission: exact
            # authority must first be promoted in a version-fenced mutation.
            may_execute = (
                authority_class == "AUTHORITATIVE" and promotion_state == "PROMOTED"
                and all_dependencies and gate_pass and item.get("status") in {"CLAIMED", "RUNNING"}
            )
            return {
                "work_item_id": str(item["id"]), "may_start_work": may_start,
                "may_continue_work": item["status"] in {"CLAIMED", "RUNNING"},
                "may_finalize_subject": may_promote, "may_certify": may_promote and gate_pass,
                "may_promote_authority": may_promote, "may_execute_science": may_execute,
                "may_access_confirmation": False, "may_admit_evidence": False,
                "blocking_conditions": blocking, "satisfied_conditions": [detail for ok, _, detail in outcomes if ok],
                "permission_snapshot_version": f"{item['work_version']}:{promotion_state}:{authority_class}",
            }

    def authority_promote(self, work_item_id: str, expected_work_version: int | None = None) -> dict:
        """Atomically mark exact, certified authority as promoted.

        This is deliberately separate from work completion and cannot promote a
        speculative/preparatory binding.
        """
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            item = cur.fetchone()
            if not item:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            if expected_work_version is not None and int(item["work_version"]) != int(expected_work_version):
                raise GPUError("LAB_WORK_VERSION_STALE", work_item_id)
            if item["authority_class"] != "AUTHORITATIVE" or item["authority_status"] != "AUTHORITATIVE":
                raise GPUError("LAB_PROMOTION_AUTHORITY_CLASS_REQUIRED", work_item_id)
            if item["promotion_state"] == "PROMOTED":
                return self._record(item) or {}
            if item["promotion_state"] not in {"ELIGIBLE", "CONDITIONALLY_ELIGIBLE"}:
                raise GPUError("LAB_PROMOTION_NOT_ELIGIBLE", str(item["promotion_state"]))
            cur.execute("SELECT * FROM lab_work_dependencies WHERE work_item_id=%s", (work_item_id,))
            outcomes = [self._dependency_status(cur, str(item["project_id"]), dep) for dep in cur.fetchall()]
            if any(invalidated for _, invalidated, _ in outcomes):
                raise GPUError("LAB_PROMOTION_DEPENDENCY_INVALIDATED", "; ".join(detail for _, invalidated, detail in outcomes if invalidated))
            if any(not satisfied for satisfied, _, _ in outcomes):
                raise GPUError("LAB_PROMOTION_DEPENDENCY_UNSATISFIED", "; ".join(detail for satisfied, _, detail in outcomes if not satisfied))
            if item.get("gate_id"):
                cur.execute("SELECT status,superseded_by,invalidation_reason FROM scientific_gates WHERE id=%s FOR SHARE", (item["gate_id"],))
                gate = cur.fetchone()
                if not gate or gate["status"] != "PASS" or gate["superseded_by"] or gate["invalidation_reason"]:
                    raise GPUError("LAB_PROMOTION_GATE_NOT_PASS", work_item_id)
            cur.execute("UPDATE lab_work_items SET promotion_state='PROMOTED',updated_at=%s,work_version=work_version+1 WHERE id=%s RETURNING *", (now, work_item_id))
            promoted = cur.fetchone()
            self._event(cur, item["project_id"], "WORK_AUTHORITY_PROMOTED", work_item_id, {"work_version": promoted["work_version"]})
            self._outbox(cur, item["project_id"], "WORK_AUTHORITY_PROMOTED", work_item_id, f"work-promoted:{work_item_id}:{promoted['work_version']}", {"work_item_id": work_item_id})
        return self._record(promoted) or {}

    def speculative_binding_create(
        self, work_item_id: str, parent_subject_id: str, parent_subject_version: str,
        parent_semantic_fingerprint: str, reusable_if_parent_changes: bool,
        invalidation_conditions: list[dict[str, Any]], rebase_policy: str,
        max_cost_class: str, protected_data_access: bool = False,
    ) -> dict:
        """Bind non-authoritative labor to an exact semantic parent/version."""
        if not parent_subject_id or not parent_subject_version or not parent_semantic_fingerprint:
            raise GPUError("LAB_SPECULATIVE_BINDING_PARENT_REQUIRED", "parent subject, version, and fingerprint")
        if not isinstance(invalidation_conditions, list) or not isinstance(protected_data_access, bool):
            raise GPUError("LAB_SPECULATIVE_BINDING_FIELDS_INVALID", "invalidation_conditions/protected_data_access")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (work_item_id,))
            work = cur.fetchone()
            if not work:
                raise GPUError("LAB_WORK_NOT_FOUND", work_item_id)
            if work["authority_class"] not in {"SPECULATIVE", "PREPARATORY"}:
                raise GPUError("LAB_SPECULATIVE_BINDING_CLASS_REQUIRED", str(work["authority_class"]))
            if work["authority_status"] == "AUTHORITATIVE" or work["promotion_state"] == "PROMOTED":
                raise GPUError("LAB_SPECULATIVE_AUTHORITY_EFFECT_FORBIDDEN", work_item_id)
            if protected_data_access:
                raise GPUError("LAB_SPECULATIVE_CONFIRMATION_ACCESS_FORBIDDEN", work_item_id)
            cur.execute(
                "INSERT INTO lab_speculative_bindings(id,project_id,work_item_id,parent_subject_id,parent_subject_version,parent_semantic_fingerprint,speculation_class,reusable_if_parent_changes,invalidation_conditions,rebase_policy,max_cost_class,protected_data_access,authority_effect,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,'NONE',%s) "
                "ON CONFLICT(work_item_id) DO UPDATE SET parent_subject_id=EXCLUDED.parent_subject_id,parent_subject_version=EXCLUDED.parent_subject_version,parent_semantic_fingerprint=EXCLUDED.parent_semantic_fingerprint,reusable_if_parent_changes=EXCLUDED.reusable_if_parent_changes,invalidation_conditions=EXCLUDED.invalidation_conditions,rebase_policy=EXCLUDED.rebase_policy,max_cost_class=EXCLUDED.max_cost_class "
                "RETURNING *",
                (ident, work["project_id"], work_item_id, parent_subject_id, parent_subject_version,
                 parent_semantic_fingerprint, work["speculation_class"], reusable_if_parent_changes,
                 json.dumps(invalidation_conditions), rebase_policy.strip()[:200], max_cost_class.strip()[:80], now),
            )
            binding = cur.fetchone()
            self._event(cur, work["project_id"], "SPECULATIVE_BINDING_CREATED", work_item_id, {
                "binding_id": str(binding["id"]), "parent_subject_id": parent_subject_id,
                "parent_subject_version": parent_subject_version, "authority_effect": "NONE",
            })
        return self._record(binding) or {}

    def review_harness_prepare(
        self, project_id: str, scientific_object_id: str, scientific_spec_version: str,
        scientific_semantic_hash: str, target_interface_version: str, worker_id: str,
        session_id: str, invariants: list[str], adversarial_cases: list[dict[str, Any]],
        protected_data_access: bool = False,
    ) -> dict[str, Any]:
        """Freeze an independent reviewer oracle before mutable bytes are exposed.

        A harness is preparation, not certification: it cannot pass a gate or
        promote authority.  It is deliberately confirmation-safe and binds to
        the frozen scientific interface/semantic identity rather than code.
        """
        required = (scientific_object_id, scientific_spec_version, scientific_semantic_hash, target_interface_version)
        if any(not str(value).strip() for value in required):
            raise GPUError("REVIEW_HARNESS_BINDING_REQUIRED", "object, spec version, semantic hash, interface version")
        if protected_data_access:
            raise GPUError("REVIEW_HARNESS_CONFIRMATION_ACCESS_FORBIDDEN", scientific_object_id)
        if not isinstance(invariants, list) or not invariants or any(not isinstance(value, str) or not value.strip() for value in invariants):
            raise GPUError("REVIEW_HARNESS_INVARIANTS_REQUIRED", "nonempty invariant strings")
        if not isinstance(adversarial_cases, list) or any(not isinstance(value, dict) for value in adversarial_cases):
            raise GPUError("REVIEW_HARNESS_CASES_INVALID", "adversarial_cases")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, session_id, worker_id, project_id)
            cur.execute(
                "SELECT * FROM lab_review_harnesses WHERE project_id=%s AND scientific_object_id=%s "
                "AND scientific_spec_version=%s AND scientific_semantic_hash=%s AND target_interface_version=%s "
                "AND prepared_by_worker_id=%s FOR UPDATE",
                (project_id, scientific_object_id, scientific_spec_version, scientific_semantic_hash,
                 target_interface_version, worker_id),
            )
            existing = cur.fetchone()
            if existing:
                return {**(self._record(existing) or {}), "dedupe_result": "REUSED_FROZEN"}
            cur.execute(
                "INSERT INTO lab_review_harnesses(id,project_id,scientific_object_id,scientific_spec_version,scientific_semantic_hash,target_interface_version,prepared_by_worker_id,prepared_by_session_id,invariants,adversarial_cases,protected_data_access,implementation_peer_exposure,status,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE,'NONE','FROZEN',%s) RETURNING *",
                (ident, project_id, scientific_object_id, scientific_spec_version, scientific_semantic_hash,
                 target_interface_version, worker_id, session_id, json.dumps(invariants), json.dumps(adversarial_cases), now),
            )
            harness = cur.fetchone()
            self._event(cur, project_id, "REVIEW_HARNESS_PREPARED", ident, {
                "scientific_object_id": scientific_object_id,
                "scientific_spec_version": scientific_spec_version,
                "scientific_semantic_hash": scientific_semantic_hash,
                "target_interface_version": target_interface_version,
                "implementation_peer_exposure": "NONE",
            })
        return {**(self._record(harness) or {}), "dedupe_result": "CREATED"}

    def review_harness_list(self, project_id: str, scientific_object_id: str | None = None) -> list[dict]:
        with self.store._connect() as conn, conn.cursor() as cur:
            sql, args = "SELECT * FROM lab_review_harnesses WHERE project_id=%s", [project_id]
            if scientific_object_id:
                sql += " AND scientific_object_id=%s"
                args.append(scientific_object_id)
            sql += " ORDER BY created_at DESC"
            cur.execute(sql, args)
            return [self._record(row) or {} for row in cur.fetchall()]

    def speculative_rebase_assess(
        self, binding_id: str, new_parent_version: str, changed_semantic_fields: list[str],
        affected_modules: list[str], unaffected_modules: list[str], reusable_artifacts: list[str],
        invalidated_artifacts: list[str], required_repairs: list[str], result: str,
    ) -> dict:
        """Record an engineering rebase decision without manufacturing authority."""
        result = self._validate(result, {"REUSABLE_AS_IS", "PARTIAL_REBASE", "FULL_REBASE", "INVALIDATED"}, "SPECULATIVE_REBASE_RESULT")
        values = (changed_semantic_fields, affected_modules, unaffected_modules, reusable_artifacts, invalidated_artifacts, required_repairs)
        if not new_parent_version or any(not isinstance(value, list) or any(not isinstance(item, str) for item in value) for value in values):
            raise GPUError("LAB_SPECULATIVE_REBASE_FIELDS_INVALID", "versions and artifact lists")
        now, ident = self._now(), str(uuid.uuid4())
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT b.*,w.status,w.project_id FROM lab_speculative_bindings b JOIN lab_work_items w ON w.id=b.work_item_id WHERE b.id=%s FOR UPDATE", (binding_id,))
            binding = cur.fetchone()
            if not binding:
                raise GPUError("LAB_SPECULATIVE_BINDING_NOT_FOUND", binding_id)
            next_promotion = "BLOCKED" if result == "REUSABLE_AS_IS" else "REBASE_REQUIRED" if result in {"PARTIAL_REBASE", "FULL_REBASE"} else "INVALIDATED"
            next_status = "INVALIDATED" if result == "INVALIDATED" else binding["status"]
            cur.execute(
                "INSERT INTO lab_speculative_rebase_assessments(id,project_id,binding_id,old_parent_version,new_parent_version,changed_semantic_fields,affected_modules,unaffected_modules,reusable_artifacts,invalidated_artifacts,required_repairs,result,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(binding_id,new_parent_version) DO UPDATE SET changed_semantic_fields=EXCLUDED.changed_semantic_fields,affected_modules=EXCLUDED.affected_modules,unaffected_modules=EXCLUDED.unaffected_modules,reusable_artifacts=EXCLUDED.reusable_artifacts,invalidated_artifacts=EXCLUDED.invalidated_artifacts,required_repairs=EXCLUDED.required_repairs,result=EXCLUDED.result "
                "RETURNING *",
                (ident, binding["project_id"], binding_id, binding["parent_subject_version"], new_parent_version,
                 *(json.dumps(value) for value in values), result, now),
            )
            assessment = cur.fetchone()
            cur.execute("UPDATE lab_work_items SET promotion_state=%s,status=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s", (next_promotion, next_status, now, binding["work_item_id"]))
            self._event(cur, binding["project_id"], "SPECULATIVE_REBASE_ASSESSED", binding["work_item_id"], {
                "binding_id": binding_id, "result": result, "new_parent_version": new_parent_version,
                "promotion_state": next_promotion,
            })
        return self._record(assessment) or {}

    def experiment_run_terminal(self, run_id: str, run_status: str) -> dict[str, int]:
        """Project terminal runs without treating technical failure as evidence.

        RESULT_READY is reserved for a canonical experimental outcome that may
        be scientifically inspected.  A cancelled/orphaned technical run has
        no such outcome and must instead remain visibly blocked pending its
        engineering reconciliation.
        """
        if run_status not in {"completed", "failed", "cancelled", "TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED"}:
            return {"result_ready": 0, "technical_blocked": 0}
        now, changed, technical_blocked = self._now(), 0, 0
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM lab_work_items WHERE (related_refs->>'experiment_run_id'=%s "
                "OR related_refs->>'run_id'=%s OR related_refs->'experiment_run_ids' ? %s "
                "OR related_refs->'run_ids' ? %s) AND status IN ('CLAIMED','RUNNING','RUNNING_DETACHED','READY') FOR UPDATE",
                (run_id, run_id, run_id, run_id),
            )
            for item in cur.fetchall():
                cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='EXPERIMENT_TERMINAL' WHERE work_item_id=%s AND released_at IS NULL", (now, item["id"]))
                if run_status in {"TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED"}:
                    cur.execute(
                        "UPDATE lab_work_items SET status='BLOCKED',assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,"
                        "blocked_reason=%s,updated_at=%s WHERE id=%s",
                        (f"Canonical ExperimentRun {run_status}; technical reconciliation is required and no scientific result is available.", now, item["id"]),
                    )
                    self._event(cur, item["project_id"], "WORK_ITEM_TECHNICAL_RUN_BLOCKED", item["id"], {
                        "run_id": run_id, "run_status": run_status, "scientific_result": "NOT_ASSESSED",
                    })
                    technical_blocked += 1
                else:
                    cur.execute("UPDATE lab_work_items SET status='RESULT_READY',assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,blocked_reason=%s,updated_at=%s WHERE id=%s", (f"Canonical ExperimentRun {run_status}; inspect before any follow-up execution.", now, item["id"]))
                    self._event(cur, item["project_id"], "WORK_ITEM_RESULT_READY", item["id"], {"run_id": run_id, "run_status": run_status})
                    changed += 1
        return {"result_ready": changed, "technical_blocked": technical_blocked}

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

    def review_pass_transition(
        self, review_work_item_id: str, gate_id: str, worker_id: str, session_id: str,
        successor_spec: dict[str, Any], rationale: str = "", review_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically close a passing review and materialize its known successor.

        This is deterministic closure, not a second worker task.  A successor
        whose specification still requires human or confirmation authorization
        is persisted as WAITING_DEPENDENCY; it is never made claimable merely
        because a release review passed.
        """
        if not isinstance(successor_spec, dict):
            raise GPUError("LAB_SUCCESSOR_SPEC_INVALID", "successor_spec must be an object")
        required = ("kind", "title", "description", "scientific_role", "equivalence_key")
        if any(not str(successor_spec.get(field, "")).strip() for field in required):
            raise GPUError("LAB_SUCCESSOR_SPEC_REQUIRED", ", ".join(required))
        if not isinstance(successor_spec.get("related_refs", {}), dict):
            raise GPUError("LAB_SUCCESSOR_REFS_INVALID", "related_refs must be an object")
        if not isinstance(successor_spec.get("dependencies", []), list):
            raise GPUError("LAB_SUCCESSOR_DEPENDENCIES_INVALID", "dependencies must be a list")
        if not isinstance(successor_spec.get("recovery_policy", {}), dict):
            raise GPUError("LAB_SUCCESSOR_RECOVERY_POLICY_INVALID", "recovery_policy must be an object")
        hard_gate_requested = bool(
            successor_spec.get("human_approval_required")
            or successor_spec.get("confirmation_authorization_required")
        )
        # Waiting without a durable dependency is merely a cosmetic state:
        # resolve_dependencies would make it READY again.  Refuse that unsafe
        # shape rather than allowing a final review to bypass an approval.
        if hard_gate_requested and not successor_spec.get("dependencies"):
            raise GPUError("LAB_SUCCESSOR_HARD_GATE_DEPENDENCY_REQUIRED", "human/confirmation approval requires typed dependencies")
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (review_work_item_id,))
            review = cur.fetchone()
            if not review:
                raise GPUError("LAB_WORK_NOT_FOUND", review_work_item_id)
            self._session(cur, session_id, worker_id, str(review["project_id"]))
            if str(review["assigned_worker_id"] or "") != str(worker_id) or str(review["assigned_session_id"] or "") != str(session_id):
                raise GPUError("LAB_WORK_NOT_OWNED", review_work_item_id)
            if str(review["gate_id"] or "") != str(gate_id):
                raise GPUError("SCIENTIFIC_GATE_REVIEW_MISMATCH", review_work_item_id)
            cur.execute("SELECT * FROM scientific_gates WHERE id=%s AND project_id=%s FOR UPDATE", (gate_id, review["project_id"]))
            gate = cur.fetchone()
            if not gate:
                raise GPUError("SCIENTIFIC_GATE_NOT_FOUND", gate_id)
            if gate["status"] not in {"PENDING", "AWAITING_SEMANTIC_REVIEW"}:
                raise GPUError("SCIENTIFIC_GATE_NOT_RESOLVABLE", gate["status"])
            if not gate["deterministic_preflight_id"]:
                raise GPUError("SCIENTIFIC_GATE_PREFLIGHT_REQUIRED", gate_id)
            cur.execute("SELECT status FROM lab_deterministic_preflights WHERE id=%s", (gate["deterministic_preflight_id"],))
            if (cur.fetchone() or {}).get("status") != "PASS":
                raise GPUError("SCIENTIFIC_GATE_PREFLIGHT_NOT_PASS", gate_id)
            review_result_id = None
            if self.feature_flags.get("VERDICT_AWARE_DEPENDENCIES_V365"):
                if review["authority_class"] != "INDEPENDENT_REVIEW":
                    raise GPUError("SCIENTIFIC_GATE_REVIEWER_CLASS_REQUIRED", review_work_item_id)
                if not isinstance(review_result, dict):
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_VERDICT_REQUIRED", review_work_item_id)
                target_subject_id = str(review_result.get("target_subject_id") or "")
                target_subject_version = str(review_result.get("target_subject_version") or "")
                review_type = str(review_result.get("review_type") or "").strip()
                verdict = self._validate(str(review_result.get("verdict") or ""), REVIEW_VERDICTS, "REVIEW_VERDICT")
                findings = review_result.get("findings") or []
                reviewed_hashes = review_result.get("reviewed_hashes") or {}
                if not target_subject_id or not target_subject_version or not review_type:
                    raise GPUError("LAB_REVIEW_RESULT_BINDING_REQUIRED", "subject id, version, and review type")
                if not isinstance(findings, list) or not isinstance(reviewed_hashes, dict):
                    raise GPUError("LAB_REVIEW_RESULT_FIELDS_INVALID", "findings and reviewed_hashes")
                if target_subject_id != str(gate["scientific_object_id"]) or target_subject_version != str(gate["canonical_subject_version"]):
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_RESULT_BINDING_MISMATCH", review_work_item_id)
                if verdict not in {"PASS", "PASS_WITH_NONMATERIAL_NOTES"}:
                    raise GPUError("SCIENTIFIC_GATE_REVIEW_VERDICT_NOT_PASS", verdict)
                immutable_payload = {
                    "target_subject_id": target_subject_id, "target_subject_version": target_subject_version,
                    "verdict": verdict, "findings": findings, "materiality": review_result.get("materiality"),
                    "reviewed_hashes": reviewed_hashes, "lineage_snapshot_id": review_result.get("lineage_snapshot_id"),
                }
                cur.execute(
                    "SELECT * FROM lab_review_results WHERE project_id=%s AND review_work_item_id=%s AND review_type=%s FOR UPDATE",
                    (review["project_id"], review_work_item_id, review_type),
                )
                stored_result = cur.fetchone()
                if stored_result:
                    stored_payload = {
                        "target_subject_id": stored_result["target_subject_id"], "target_subject_version": stored_result["target_subject_version"],
                        "verdict": stored_result["verdict"], "findings": stored_result["findings"] or [], "materiality": stored_result["materiality"],
                        "reviewed_hashes": stored_result["reviewed_hashes"] or {}, "lineage_snapshot_id": stored_result["lineage_snapshot_id"],
                    }
                    if stored_payload != immutable_payload:
                        raise GPUError("LAB_REVIEW_RESULT_IMMUTABLE", review_work_item_id)
                    review_result_id = str(stored_result["id"])
                else:
                    review_result_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO lab_review_results(id,project_id,review_work_item_id,target_subject_id,target_subject_version,review_type,verdict,findings,materiality,reviewed_hashes,lineage_snapshot_id,created_at) "
                        "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (review_result_id, review["project_id"], review_work_item_id, target_subject_id,
                         target_subject_version, review_type, verdict, json.dumps(findings), review_result.get("materiality"),
                         json.dumps(reviewed_hashes), review_result.get("lineage_snapshot_id"), now),
                    )
                    self._event(cur, review["project_id"], "REVIEW_RESULT_RECORDED", review_result_id, {
                        "review_work_item_id": review_work_item_id, "review_type": review_type,
                        "verdict": verdict, "target_subject_id": target_subject_id,
                        "target_subject_version": target_subject_version,
                    })

            # Create/reuse the successor while the gate/review authority is
            # locked.  The partial unique equivalence index arbitrates races.
            equivalence_key = str(successor_spec["equivalence_key"])
            cur.execute(
                "SELECT * FROM lab_work_items WHERE project_id=%s AND equivalence_key=%s "
                "AND status=ANY(%s) ORDER BY created_at LIMIT 1 FOR UPDATE",
                (review["project_id"], equivalence_key, list(EQUIVALENCE_ACTIVE_WORK_STATUSES | {"COMPLETED"})),
            )
            successor = cur.fetchone()
            reused = successor is not None
            if not successor:
                successor_id = str(uuid.uuid4())
                hard_gate = hard_gate_requested
                status = "WAITING_DEPENDENCY" if hard_gate else "READY"
                refs = dict(successor_spec.get("related_refs") or {})
                refs.update({"predecessor_review_work_item_id": review_work_item_id, "gate_id": gate_id,
                             "deterministic_successor": True})
                cur.execute(
                    "INSERT INTO lab_work_items(id,project_id,kind,title,description,scientific_role,status,priority,expected_value,estimated_cost,"
                    "created_by,parent_work_item_id,related_refs,equivalence_key,authority_key,gate_id,canonical_subject_version,authority_status,subject_id,"
                    "recovery_policy,canonical_objective_id,speculation_class,speculation_condition,resource_class,dependency_scope,created_at,updated_at) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (successor_id, review["project_id"], str(successor_spec["kind"]), str(successor_spec["title"]),
                     str(successor_spec["description"]), str(successor_spec["scientific_role"]), status,
                     float(successor_spec.get("priority", 0)), successor_spec.get("expected_value"), successor_spec.get("estimated_cost"),
                     worker_id, review_work_item_id, json.dumps(refs), equivalence_key,
                     successor_spec.get("authority_key"), successor_spec.get("gate_id"), successor_spec.get("canonical_subject_version"),
                     str(successor_spec.get("authority_status", "SUPPORTING")), successor_spec.get("subject_id"),
                     json.dumps(successor_spec.get("recovery_policy") or {}), successor_spec.get("canonical_objective_id"),
                     "NON_SPECULATIVE", json.dumps({}), str(successor_spec.get("resource_class", "REASONING")),
                     "WORKITEM_LOCAL", now, now),
                )
                self._replace_dependencies(cur, successor_id, successor_spec.get("dependencies") or [], now)
                cur.execute("SELECT * FROM lab_work_items WHERE id=%s", (successor_id,))
                successor = cur.fetchone()
                self._event(cur, review["project_id"], "WORK_ITEM_SUCCESSOR_MATERIALIZED", successor_id, {
                    "predecessor_review_work_item_id": review_work_item_id, "gate_id": gate_id,
                    "hard_gate": hard_gate, "equivalence_key": equivalence_key,
                })
            cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='REVIEW_PASSED',lease_version=lease_version+1 WHERE id=%s AND released_at IS NULL", (now, review["lease_id"]))
            cur.execute("UPDATE lab_work_items SET status='COMPLETED',completed_at=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s", (now, now, review_work_item_id))
            cur.execute(
                "UPDATE scientific_gates SET status='PASS',semantic_review_work_item_id=%s,"
                "semantic_review_result_id=COALESCE(%s,semantic_review_result_id),resolved_at=%s WHERE id=%s",
                (review_work_item_id, review_result_id, now, gate_id),
            )
            cur.execute("UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s WHERE id=%s", (now, session_id))
            cur.execute("UPDATE research_workers SET availability_state='AVAILABLE',availability_updated_at=%s,idle_reason=NULL,idle_since=NULL WHERE id=%s", (now, worker_id))
            self._event(cur, review["project_id"], "SCIENTIFIC_GATE_REVIEW_PASSED_ATOMIC", gate_id, {
                "review_work_item_id": review_work_item_id, "successor_work_item_id": str(successor["id"]),
                "successor_reused": reused, "review_result_id": review_result_id, "rationale": rationale[:4000],
            })
            result = {"gate": self._record(gate) or {}, "review_work_item_id": review_work_item_id,
                      "successor": self._record(successor) or {}, "successor_reused": reused}
        result["dependency_changes"] = self.resolve_dependencies(str(review["project_id"]))
        result["gate"] = self.gate_get(gate_id)
        result["successor"] = self.work_get(str(result["successor"]["id"]))
        result["turn_status"] = (
            "CONTINUE" if result["successor"]["status"] == "READY"
            else "WAITING_DEPENDENCY" if result["successor"]["status"] in {"WAITING_DEPENDENCY", "DORMANT", "BLOCKED"}
            else "COMPLETED"
        )
        return result

    def _fresh_discovery_batch_for_reassigned_claim(
        self, cur, item: dict, worker_id: str, session_id: str, now: datetime
    ) -> dict:
        """Replace an expired DDE generation lane when its WorkItem is claimed.

        Candidate batches are owned by a worker *session*, not merely by a
        WorkItem.  A new claimant must never inherit an open independent batch
        from the expired session.  This transaction retires that lane and
        creates an equivalent fresh lane with the same frozen operator and
        distance reservation before ownership of the WorkItem changes.
        """
        if item.get("kind") != "DISCOVERY_GENERATION":
            return item
        refs = dict(item.get("related_refs") or {})
        round_id = str(refs.get("discovery_round_id") or "")
        batch_id = str(refs.get("candidate_batch_id") or "")
        if not round_id or not batch_id:
            raise GPUError("DISCOVERY_GENERATION_BINDING_REQUIRED", str(item["id"]))
        cur.execute(
            "SELECT * FROM research_objects WHERE id=%s AND project_id=%s AND kind='DiscoveryRound' FOR UPDATE",
            (round_id, item["project_id"]),
        )
        round_ = cur.fetchone()
        if not round_:
            raise GPUError("DISCOVERY_ROUND_NOT_FOUND", round_id)
        round_data = dict(round_["data"] or {})
        if (
            round_["status"] != "ACTIVE"
            or round_data.get("phase") != "INDEPENDENT_GENERATION"
            or bool(round_data.get("stale"))
        ):
            if bool(round_data.get("stale")):
                raise GPUError("DISCOVERY_ROUND_STALE", "Cannot reassign a generation lane from a stale round.")
            raise GPUError("DISCOVERY_ROUND_NOT_GENERATING", "Cannot replace a generation lane outside independent generation.")
        cur.execute(
            "SELECT * FROM discovery_round_memberships WHERE discovery_round_id=%s AND candidate_batch_id=%s FOR UPDATE",
            (round_id, batch_id),
        )
        member = cur.fetchone()
        if not member:
            raise GPUError("DISCOVERY_BATCH_NOT_FOUND", batch_id)
        if str(member["worker_id"]) == str(worker_id) and str(member["worker_session_id"]) == str(session_id):
            return item
        if member["status"] not in {"JOINED", "EXPIRED"}:
            raise GPUError("DISCOVERY_BATCH_REASSIGNMENT_NOT_ALLOWED", str(member["status"]))
        cur.execute(
            "SELECT 1 FROM discovery_round_memberships WHERE discovery_round_id=%s AND worker_session_id=%s FOR UPDATE",
            (round_id, session_id),
        )
        if cur.fetchone():
            raise GPUError("DISCOVERY_WORKER_ALREADY_ASSIGNED", session_id)
        cur.execute(
            "SELECT * FROM research_objects WHERE id=%s AND project_id=%s AND kind='CandidateBatch' FOR UPDATE",
            (batch_id, item["project_id"]),
        )
        old_batch = cur.fetchone()
        if not old_batch:
            raise GPUError("DISCOVERY_BATCH_NOT_FOUND", batch_id)
        if member["status"] == "JOINED":
            old_data = {
                **dict(old_batch["data"] or {}),
                "status": "EXPIRED",
                "expired_at": now.isoformat(),
                "expiration_reason": "WORKITEM_REASSIGNED_FRESH_LANE",
            }
            cur.execute(
                "UPDATE research_objects SET status='EXPIRED',data=%s WHERE id=%s",
                (json.dumps(old_data, default=str), batch_id),
            )
            cur.execute(
                "UPDATE discovery_round_memberships SET status='EXPIRED',frozen_at=%s,"
                "failure_reason='WORKITEM_REASSIGNED_FRESH_LANE' WHERE discovery_round_id=%s AND candidate_batch_id=%s",
                (now, round_id, batch_id),
            )
            self._event(cur, item["project_id"], "DISCOVERY_BATCH_EXPIRED", batch_id, {
                "round_id": round_id, "reason": "WORKITEM_REASSIGNED_FRESH_LANE",
            })
        replacement_batch_id = str(uuid.uuid4())
        replacement_data = {
            "discovery_round_id": round_id,
            "worker_id": str(worker_id),
            "worker_session_id": str(session_id),
            "generation_operator": member["generation_operator"],
            "requested_distance": member["requested_distance"],
            "candidate_ids": [],
            "status": "JOINED",
            "immutable_after_freeze": True,
            "created_at": now.isoformat(),
            "replacement_for_batch_id": batch_id,
            "replacement_reason": "EXPIRED_WORKITEM_REASSIGNED",
        }
        cur.execute(
            "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
            "VALUES(%s,%s,'CandidateBatch','ACTIVE',%s,%s)",
            (replacement_batch_id, item["project_id"], json.dumps(replacement_data), now),
        )
        cur.execute(
            "INSERT INTO discovery_round_memberships(discovery_round_id,worker_id,worker_session_id,"
            "candidate_batch_id,generation_operator,requested_distance,status,independent_generation,joined_at) "
            "VALUES(%s,%s,%s,%s,%s,%s,'JOINED',%s,%s)",
            (round_id, worker_id, session_id, replacement_batch_id, member["generation_operator"],
             member["requested_distance"], member["independent_generation"], now),
        )
        refs.update({
            "candidate_batch_id": replacement_batch_id,
            "replaced_candidate_batch_id": batch_id,
            "discovery_batch_recovery": "FRESH_INDEPENDENT_REPLACEMENT",
        })
        cur.execute(
            "UPDATE lab_work_items SET related_refs=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s RETURNING *",
            (json.dumps(refs), now, item["id"]),
        )
        updated = cur.fetchone()
        self._event(cur, item["project_id"], "DISCOVERY_BATCH_REPLACEMENT_JOINED", replacement_batch_id, {
            "round_id": round_id, "replaced_batch_id": batch_id, "work_item_id": str(item["id"]),
            "worker_id": str(worker_id), "session_id": str(session_id),
            "generation_operator": member["generation_operator"], "requested_distance": member["requested_distance"],
        })
        self._event(cur, item["project_id"], "DISCOVERY_GENERATION_WORK_REBOUND", str(item["id"]), {
            "old_batch_id": batch_id, "new_batch_id": replacement_batch_id,
            "mode": "FRESH_INDEPENDENT_REPLACEMENT",
        })
        return updated

    def review_material_repair_transition(
        self, review_work_item_id: str, gate_id: str, worker_id: str, session_id: str,
        repair_successor_spec: dict[str, Any], review_result: dict[str, Any], rationale: str = "",
    ) -> dict[str, Any]:
        """Atomically close an MRR review and open a non-promoted repair card."""
        if not isinstance(repair_successor_spec, dict) or not isinstance(review_result, dict):
            raise GPUError("LAB_REPAIR_TRANSITION_FIELDS_INVALID", "repair_successor_spec and review_result must be objects")
        required = ("kind", "title", "description", "scientific_role", "equivalence_key", "new_parent_version")
        if any(not str(repair_successor_spec.get(field, "")).strip() for field in required):
            raise GPUError("LAB_REPAIR_SUCCESSOR_SPEC_REQUIRED", ", ".join(required))
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE id=%s FOR UPDATE", (review_work_item_id,))
            review = cur.fetchone()
            if not review:
                raise GPUError("LAB_WORK_NOT_FOUND", review_work_item_id)
            self._session(cur, session_id, worker_id, str(review["project_id"]))
            if str(review["assigned_worker_id"] or "") != str(worker_id) or str(review["assigned_session_id"] or "") != str(session_id):
                raise GPUError("LAB_WORK_NOT_OWNED", review_work_item_id)
            if str(review["gate_id"] or "") != str(gate_id):
                raise GPUError("SCIENTIFIC_GATE_REVIEW_MISMATCH", review_work_item_id)
            if self.feature_flags.get("VERDICT_AWARE_DEPENDENCIES_V365") and review["authority_class"] != "INDEPENDENT_REVIEW":
                raise GPUError("SCIENTIFIC_GATE_REVIEWER_CLASS_REQUIRED", review_work_item_id)
            cur.execute("SELECT * FROM scientific_gates WHERE id=%s AND project_id=%s FOR UPDATE", (gate_id, review["project_id"]))
            gate = cur.fetchone()
            if not gate or gate["status"] not in {"PENDING", "AWAITING_SEMANTIC_REVIEW"}:
                raise GPUError("SCIENTIFIC_GATE_NOT_RESOLVABLE", str(gate["status"]) if gate else gate_id)
            target_id = str(review_result.get("target_subject_id") or "")
            target_version = str(review_result.get("target_subject_version") or "")
            review_type = str(review_result.get("review_type") or "").strip()
            verdict = self._validate(str(review_result.get("verdict") or ""), REVIEW_VERDICTS, "REVIEW_VERDICT")
            findings, hashes = review_result.get("findings") or [], review_result.get("reviewed_hashes") or {}
            if not target_id or not target_version or not review_type or not isinstance(findings, list) or not isinstance(hashes, dict):
                raise GPUError("LAB_REVIEW_RESULT_BINDING_REQUIRED", "bound review result fields")
            if target_id != str(gate["scientific_object_id"]) or target_version != str(gate["canonical_subject_version"]):
                raise GPUError("SCIENTIFIC_GATE_REVIEW_RESULT_BINDING_MISMATCH", review_work_item_id)
            if verdict != "MATERIAL_REPAIR_REQUIRED":
                raise GPUError("SCIENTIFIC_GATE_REPAIR_VERDICT_REQUIRED", verdict)
            cur.execute("SELECT * FROM lab_review_results WHERE project_id=%s AND review_work_item_id=%s AND review_type=%s FOR UPDATE", (review["project_id"], review_work_item_id, review_type))
            stored = cur.fetchone()
            if stored:
                if str(stored["verdict"]) != verdict or str(stored["target_subject_id"]) != target_id or str(stored["target_subject_version"]) != target_version:
                    raise GPUError("LAB_REVIEW_RESULT_IMMUTABLE", review_work_item_id)
                result_id = str(stored["id"])
            else:
                result_id = str(uuid.uuid4())
                cur.execute("INSERT INTO lab_review_results(id,project_id,review_work_item_id,target_subject_id,target_subject_version,review_type,verdict,findings,materiality,reviewed_hashes,lineage_snapshot_id,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (result_id, review["project_id"], review_work_item_id, target_id, target_version, review_type, verdict, json.dumps(findings), review_result.get("materiality"), json.dumps(hashes), review_result.get("lineage_snapshot_id"), now))
                self._event(cur, review["project_id"], "REVIEW_RESULT_RECORDED", result_id, {"review_work_item_id": review_work_item_id, "review_type": review_type, "verdict": verdict, "target_subject_id": target_id, "target_subject_version": target_version})
            # Old promoted-candidate paths cannot remain runnable after MRR.
            cur.execute("SELECT id FROM lab_work_items WHERE gate_id=%s AND id<>%s AND authority_class='AUTHORITATIVE' AND status=ANY(%s) FOR UPDATE", (gate_id, review_work_item_id, list(ACTIVE_WORK_STATUSES)))
            superseded_ids = [str(row["id"]) for row in cur.fetchall()]
            for work_id in superseded_ids:
                cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='SEMANTIC_REPAIR_REQUIRED',lease_version=lease_version+1 WHERE work_item_id=%s AND released_at IS NULL", (now, work_id))
                cur.execute("UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',promotion_state='SUPERSEDED',assigned_worker_id=NULL,assigned_session_id=NULL,lease_id=NULL,invalidated_at=%s,invalidated_reason=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s", (now, "Semantic review requires material repair", now, work_id))
            cur.execute("UPDATE lab_work_items w SET promotion_state='REBASE_REQUIRED',blocked_reason=%s,updated_at=%s,work_version=work_version+1 FROM lab_speculative_bindings b WHERE b.work_item_id=w.id AND b.project_id=%s AND b.parent_subject_id=%s AND b.parent_subject_version=%s AND w.status=ANY(%s)", ("Upstream semantic review requires rebase assessment.", now, review["project_id"], gate["scientific_object_id"], gate["canonical_subject_version"], list(ACTIVE_WORK_STATUSES)))
            rebased_count = cur.rowcount
            key = str(repair_successor_spec["equivalence_key"])
            cur.execute("SELECT * FROM lab_work_items WHERE project_id=%s AND equivalence_key=%s AND status=ANY(%s) ORDER BY created_at LIMIT 1 FOR UPDATE", (review["project_id"], key, list(EQUIVALENCE_ACTIVE_WORK_STATUSES | {"COMPLETED"})))
            repair = cur.fetchone()
            reused = repair is not None
            if not repair:
                repair_id = str(uuid.uuid4())
                refs = {**dict(repair_successor_spec.get("related_refs") or {}), "predecessor_review_work_item_id": review_work_item_id, "review_result_id": result_id, "repair_required": True}
                cur.execute("INSERT INTO lab_work_items(id,project_id,kind,title,description,scientific_role,status,priority,expected_value,estimated_cost,created_by,parent_work_item_id,related_refs,equivalence_key,canonical_subject_version,authority_status,subject_id,recovery_policy,speculation_class,speculation_condition,resource_class,dependency_scope,authority_class,promotion_state,created_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s,'READY',%s,%s,%s,%s,%s,%s,%s,%s,'SUPPORTING',%s,%s,'NON_SPECULATIVE','{}'::jsonb,%s,'WORKITEM_LOCAL','RECOVERY','NOT_APPLICABLE',%s,%s)", (repair_id, review["project_id"], str(repair_successor_spec["kind"]), str(repair_successor_spec["title"]), str(repair_successor_spec["description"]), str(repair_successor_spec["scientific_role"]), float(repair_successor_spec.get("priority", 0)), repair_successor_spec.get("expected_value"), repair_successor_spec.get("estimated_cost"), worker_id, review_work_item_id, json.dumps(refs), key, str(repair_successor_spec["new_parent_version"]), gate["scientific_object_id"], json.dumps(repair_successor_spec.get("recovery_policy") or {}), str(repair_successor_spec.get("resource_class", "REASONING")), now, now))
                self._replace_dependencies(cur, repair_id, repair_successor_spec.get("dependencies") or [], now)
                cur.execute("SELECT * FROM lab_work_items WHERE id=%s", (repair_id,))
                repair = cur.fetchone()
                self._event(cur, review["project_id"], "WORK_ITEM_REPAIR_MATERIALIZED", repair_id, {"gate_id": gate_id, "review_result_id": result_id, "equivalence_key": key})
            cur.execute("UPDATE lab_work_leases SET released_at=%s,release_reason='REVIEW_MATERIAL_REPAIR_REQUIRED',lease_version=lease_version+1 WHERE id=%s AND released_at IS NULL", (now, review["lease_id"]))
            cur.execute("UPDATE lab_work_items SET status='COMPLETED',completed_at=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s", (now, now, review_work_item_id))
            cur.execute("UPDATE scientific_gates SET status='REPAIR_REQUIRED',semantic_review_work_item_id=%s,semantic_review_result_id=%s,resolved_at=%s WHERE id=%s", (review_work_item_id, result_id, now, gate_id))
            cur.execute("UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s WHERE id=%s", (now, session_id))
            cur.execute("UPDATE research_workers SET availability_state='AVAILABLE',availability_updated_at=%s,idle_reason=NULL,idle_since=NULL WHERE id=%s", (now, worker_id))
            self._event(cur, review["project_id"], "SCIENTIFIC_GATE_REPAIR_REQUIRED_ATOMIC", gate_id, {"review_work_item_id": review_work_item_id, "review_result_id": result_id, "repair_work_item_id": str(repair["id"]), "superseded_work_item_ids": superseded_ids, "speculative_rebase_required_count": rebased_count, "rationale": rationale[:4000]})
        return {"gate": self.gate_get(gate_id), "review_result_id": result_id, "repair_work_item": self.work_get(str(repair["id"])), "repair_reused": reused, "superseded_work_item_ids": superseded_ids, "speculative_rebase_required_count": rebased_count, "dependency_changes": self.resolve_dependencies(str(review["project_id"])), "turn_status": "CONTINUE"}

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

    def _project_scheduler_workers(self, project_id: str) -> list[dict]:
        """Project-scoped worker projection for audit and shadow scheduling.

        Worker identities are global, but assignment capacity is not.  A worker
        joined to another project must never appear as capacity in this one.
        """
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (w.id) w.id,w.display_name,w.worker_type,"
                "CASE WHEN s.current_work_item_id IS NOT NULL THEN 'ASSIGNED' ELSE w.availability_state END AS availability_state,"
                "w.availability_updated_at,w.idle_reason,w.idle_since,s.id AS session_id,s.status AS session_status "
                "FROM research_workers w JOIN research_worker_sessions s ON s.worker_id=w.id "
                "WHERE w.enabled=TRUE AND s.current_project_id=%s AND s.status NOT IN ('DISCONNECTED','EXPIRED') "
                "ORDER BY w.id,s.last_heartbeat_at DESC",
                (project_id,),
            )
            return [self._record(row) or {} for row in cur.fetchall()]

    def policy_invariants_reconcile(self, project_id: str) -> dict[str, int]:
        """Repair mechanically provable v3.6.4 control-plane contradictions.

        This central reconciler deliberately performs no scientific inference:
        it only retires obsolete authority projections, prevents technical
        terminal runs from masquerading as inspectable results, and returns an
        idle worker to the available pool when canonical READY work exists.
        """
        now = self._now()
        superseded = technical_blocked = false_idle = 0
        active = list(ACTIVE_WORK_STATUSES | {"DORMANT"})
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT w.*,g.status AS gate_status,g.canonical_subject_version AS gate_version,"
                "g.superseded_by AS gate_superseded_by,g.invalidation_reason AS gate_invalidation_reason "
                "FROM lab_work_items w JOIN scientific_gates g ON g.id=w.gate_id "
                "WHERE w.project_id=%s AND w.status=ANY(%s) AND ("
                "g.status IN ('SUPERSEDED','INVALID') OR g.superseded_by IS NOT NULL "
                "OR g.invalidation_reason IS NOT NULL OR w.canonical_subject_version IS DISTINCT FROM g.canonical_subject_version) "
                "FOR UPDATE OF w",
                (project_id, active),
            )
            for item in cur.fetchall():
                cur.execute(
                    "UPDATE lab_work_leases SET released_at=%s,release_reason='OBSOLETE_SUBJECT_VERSION',lease_version=lease_version+1 "
                    "WHERE work_item_id=%s AND released_at IS NULL",
                    (now, item["id"]),
                )
                cur.execute(
                    "UPDATE lab_work_items SET status='SUPERSEDED',authority_status='SUPERSEDED',assigned_worker_id=NULL,"
                    "assigned_session_id=NULL,lease_id=NULL,invalidated_at=%s,invalidated_reason=%s,completed_at=%s,"
                    "updated_at=%s,work_version=work_version+1 WHERE id=%s",
                    (now, "Canonical gate/version is obsolete.", now, now, item["id"]),
                )
                cur.execute(
                    "UPDATE research_worker_sessions SET current_work_item_id=NULL,active_role=NULL,status='ACTIVE',last_heartbeat_at=%s "
                    "WHERE current_work_item_id=%s",
                    (now, item["id"]),
                )
                self._event(cur, project_id, "OBSOLETE_READY_WORK", item["id"], {
                    "gate_id": str(item["gate_id"]), "work_version": item["canonical_subject_version"],
                    "gate_version": item["gate_version"], "gate_status": item["gate_status"],
                })
                superseded += 1

            cur.execute(
                "SELECT w.id,w.related_refs,r.id AS run_id,r.status AS run_status FROM lab_work_items w "
                "JOIN research_objects r ON r.id::text=(w.related_refs->>'experiment_run_id') "
                "WHERE w.project_id=%s AND w.status='RESULT_READY' AND r.kind='ExperimentRun' "
                "AND r.status IN ('TECHNICAL_CANCELLED','TECHNICAL_ORPHANED') FOR UPDATE OF w",
                (project_id,),
            )
            for item in cur.fetchall():
                cur.execute(
                    "UPDATE lab_work_items SET status='BLOCKED',blocked_reason=%s,updated_at=%s,work_version=work_version+1 WHERE id=%s",
                    (f"ExperimentRun {item['run_status']} is technical only; reconciliation required.", now, item["id"]),
                )
                self._event(cur, project_id, "TECHNICAL_RESULT_STATE_RECONCILED", item["id"], {
                    "run_id": str(item["run_id"]), "run_status": item["run_status"], "scientific_result": "NOT_ASSESSED",
                })
                technical_blocked += 1

            cur.execute(
                "SELECT w.id FROM lab_work_items w WHERE w.project_id=%s AND w.status='READY' LIMIT 1",
                (project_id,),
            )
            ready = cur.fetchone()
            if ready:
                cur.execute(
                    "SELECT id FROM research_workers WHERE availability_state='IDLE' AND enabled=TRUE AND EXISTS ("
                    "SELECT 1 FROM research_worker_sessions s WHERE s.worker_id=research_workers.id "
                    "AND s.current_project_id=%s AND s.status='ACTIVE' AND s.current_work_item_id IS NULL) FOR UPDATE",
                    (project_id,),
                )
                for worker in cur.fetchall():
                    cur.execute(
                        "UPDATE research_workers SET availability_state='AVAILABLE',availability_updated_at=%s,"
                        "idle_reason=NULL,idle_since=NULL WHERE id=%s",
                        (now, worker["id"]),
                    )
                    self._event(cur, project_id, "FALSE_IDLE_TRANSITION", ready["id"], {
                        "worker_id": str(worker["id"]), "ready_work_item_id": str(ready["id"]),
                    })
                    false_idle += 1
        if superseded:
            self.resolve_dependencies(project_id)
        return {"obsolete_work_superseded": superseded, "technical_result_blocked": technical_blocked,
                "false_idle_recovered": false_idle}

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
            self.policy_invariants_reconcile(project_id)
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
        self.policy_invariants_reconcile(project_id)
        lease_state = "IDLE"
        lease_detail: dict[str, Any] = {}
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT worker_id,current_work_item_id,status FROM research_worker_sessions WHERE id=%s AND current_project_id=%s FOR UPDATE", (session_id, project_id))
            session = cur.fetchone()
            if not session:
                raise GPUError("LAB_SESSION_NOT_FOUND", session_id)
            now = self._now()
            self._clear_stale_session_ownership(cur, str(session["worker_id"]), now)
            cur.execute(
                "SELECT worker_id,current_work_item_id,status FROM research_worker_sessions "
                "WHERE id=%s AND current_project_id=%s FOR UPDATE",
                (session_id, project_id),
            )
            session = cur.fetchone()
            if not session:
                raise GPUError("LAB_SESSION_NOT_FOUND", session_id)
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
                "context_version": "v3.6.6",
                "generated_at": self._now(),
                "worker": {"worker_id": str(session["worker_id"]), "session_id": session_id, "status": session["status"]},
                "project": {"project_id": project_id, "project_state_version": state["research_state_version"]},
                "canonical_objective": self._record(objective),
                "work": None if not item else {
                    "work_item_id": str(item["id"]), "work_item_version": item["work_version"],
                    "authority_key": item["authority_key"], "subject_id": item["subject_id"],
                    "subject_version": item["canonical_subject_version"], "status": item["status"],
                    "role": item["scientific_role"], "mode": item["kind"],
                    "correction_layer": (item["related_refs"] or {}).get("correction_layer"),
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
                self._refresh_portfolio_branches(cur, project_id, canonical_objective_id, now)
                cur.execute("SELECT * FROM hypothesis_portfolios WHERE id=%s", (existing["id"],))
                return self._record(cur.fetchone()) or {}
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
                branch_dependencies = list(branch["branch_dependencies"] or [])
                branch_dependency_outcomes = [
                    self._branch_dependency_status(cur, project_id, dependency)
                    for dependency in branch_dependencies
                ]
                unresolved_branch_dependencies = [
                    {"dependency": dependency, "reason": detail, "invalidated": invalidated}
                    for dependency, (satisfied, invalidated, detail) in zip(
                        branch_dependencies, branch_dependency_outcomes, strict=True
                    )
                    if not satisfied
                ]
                terminal = branch["state"] in {"RESOLVED", "REFUTED", "ABANDONED", "SUPERSEDED"}
                if terminal:
                    next_actionability, saturation = "RESOLVED", "RESOLVED"
                elif unresolved_branch_dependencies:
                    next_actionability = "WAITING_BRANCH_DEPENDENCY"
                    saturation = "BLOCKED_GLOBAL" if branch["branch_blocking_scope"] == "OBJECTIVE_GLOBAL" else "BLOCKED_LOCAL"
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
                    "waiting_dependency": bool(waiting or unresolved_branch_dependencies), "unresolved_dependencies": unresolved_dependencies,
                    "unresolved_branch_dependencies": unresolved_branch_dependencies,
                    "current_gate_ids": [str(item["gate_id"]) for item in active if item["gate_id"]],
                    "unresolved_question": branch["question_id"], "scientific_niche": branch["mechanistic_niche_id"],
                    "next_actionability": next_actionability, "branch_saturation_state": saturation,
                    "last_scientific_progress_at": max((item["updated_at"] for item in work), default=branch["created_at"]),
                    "last_progress_at": max((item["updated_at"] for item in work), default=branch["created_at"]),
                })
            return result

    def agenda_coverage_get(self, project_id: str) -> list[dict]:
        """Read-only v3.6 agenda coverage, including unbranched active questions."""
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
        # ResearchAgenda is authoritative for questions.  Do not hide an OPEN
        # AgendaItem just because a HypothesisBranch has not yet been proposed.
        # Branch.question_id is the durable link when a branch already exists.
        linked_questions = {str(branch.get("unresolved_question")) for branch in coverage if branch.get("unresolved_question")}
        agenda_items = self.store.objects_list(project_id, "AgendaItem", {"OPEN", "ACTIVE"}, limit=None)
        for agenda_item in agenda_items:
            agenda_id = str(agenda_item["id"])
            if agenda_id in linked_questions:
                continue
            result.append({
                "canonical_objective_id": None,
                "agenda_item_id": agenda_id,
                "agenda_question": agenda_item.get("data", {}).get("question"),
                "branches": [],
                "actionable_uncovered_count": 1,
                "coverage_state": "UNCOVERED_ACTIONABLE",
                "planner_action": "CONSIDER_BRANCH_PROPOSAL",
            })
        return result

    def work_planner_candidates(self, project_id: str, limit: int = 50) -> dict:
        """Agenda-aware planning input. It never creates work merely to occupy idle workers."""
        coverage = self.branch_coverage_get(project_id)
        agenda_coverage = self.agenda_coverage_get(project_id)
        ready = self.work_list(project_id, ["READY"], limit)
        global_blocks = self._objective_global_blocks(project_id)
        active_branch_ids = {str(item["branch_id"]) for item in ready if item.get("branch_id")}
        undercovered = [branch for branch in coverage if branch["status"] in {"OPEN", "WAITING"} and branch["active_work_count"] == 0]
        unbranched_agenda_items = [
            item for item in agenda_coverage
            if item.get("agenda_item_id") and item.get("coverage_state") == "UNCOVERED_ACTIONABLE"
        ]
        return {
            "project_id": project_id,
            "ready_canonical_work": [item for item in ready if item.get("authority_status") == "AUTHORITATIVE"],
            "branch_coverage": coverage,
            "undercovered_branches": undercovered,
            "unbranched_agenda_items": unbranched_agenda_items,
            "objective_global_blocks": global_blocks,
            "planner_action": "IDLE_OBJECTIVE_GLOBAL_BLOCK" if global_blocks else "CLAIM_EXISTING" if ready else "CONSIDER_PROPOSAL" if (undercovered or unbranched_agenda_items) else "IDLE",
            "materialization_requires": "approved WorkProposal or authorized planner/gate transition",
        }

    def _objective_global_blocks(self, project_id: str) -> list[dict]:
        """Return durable global blockers without widening branch-local waits."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,blocked_reason,branch_id,status FROM lab_work_items WHERE project_id=%s "
                "AND dependency_scope='OBJECTIVE_GLOBAL' AND status IN ('WAITING_DEPENDENCY','BLOCKED','DORMANT','REPLAN_REQUIRED') "
                "ORDER BY created_at,id",
                (project_id,),
            )
            return [self._record(row) or {} for row in cur.fetchall()]

    def portfolio_scheduler_shadow(self, project_id: str, limit: int = 50) -> dict:
        """Read-only v3.6 scheduling recommendation using existing safe READY work."""
        coverage = {item["branch_id"]: item for item in self.branch_coverage_get(project_id)}
        ready = [item for item in self.work_list(project_id, ["READY"], limit) if self._scheduler_actionable_work(item)]
        global_blocks = self._objective_global_blocks(project_id)
        workers = self._project_scheduler_workers(project_id)
        return ResearchPortfolioV36Shadow.project(workers, ready, coverage, global_blocks)

    def portfolio_historical_replay(self, project_id: str, limit: int = 1000) -> dict:
        """Read-only replay of the operational information known at each event.

        It reports opportunities, never counterfactual scientific outcomes. Older
        records without v3.6 branch metadata remain unknown instead of being
        reconstructed from today's database state.
        """
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT event_type,subject_id,payload,created_at FROM research_events "
                "WHERE project_id=%s ORDER BY created_at,id LIMIT %s",
                (project_id, min(max(1, limit), 5_000)),
            )
            events = cur.fetchall()

        work: dict[str, dict[str, Any]] = {}
        workers: dict[str, str] = {}
        schedule_points: list[dict[str, Any]] = []
        missed_independent = duplicate_signals = unknown_branch_events = 0

        def alternatives(branch_id: str | None) -> list[str]:
            return [item_id for item_id, item in work.items()
                    if item.get("status") == "READY" and item.get("authority_status") == "AUTHORITATIVE"
                    and item.get("branch_id") and item.get("branch_id") != branch_id]

        for event in events:
            event_type = event["event_type"]
            subject_id = str(event["subject_id"]) if event["subject_id"] else None
            payload = event["payload"] or {}
            point: dict[str, Any] | None = None
            if event_type in {"WORKER_JOINED", "WORKER_SESSION_RECOVERED"}:
                worker_id = str(payload.get("worker_id") or "")
                if worker_id:
                    workers[worker_id] = "AVAILABLE"
            elif event_type == "WORK_ITEM_CREATED" and subject_id:
                branch_id = payload.get("branch_id")
                unknown_branch_events += int(not bool(branch_id))
                work[subject_id] = {"status": payload.get("status", "READY"),
                                    "branch_id": str(branch_id) if branch_id else None,
                                    "authority_status": payload.get("authority_status"),
                                    "equivalence_key": payload.get("equivalence_key")}
                if payload.get("equivalence_key") and any(
                    item.get("equivalence_key") == payload["equivalence_key"]
                    and item.get("status") in ACTIVE_WORK_STATUSES
                    for existing_id, item in work.items() if existing_id != subject_id
                ):
                    duplicate_signals += 1
            elif event_type == "WORK_ITEM_READY" and subject_id:
                work.setdefault(subject_id, {})["status"] = "READY"
                point = {"kind": "READY_WORK", "work_item_id": subject_id}
            elif event_type == "WORK_CLAIMED" and subject_id:
                item = work.setdefault(subject_id, {})
                item["status"] = "CLAIMED"
                item["worker_id"] = str(payload.get("worker_id") or "")
                if item["worker_id"]:
                    workers[item["worker_id"]] = "BUSY"
                point = {"kind": "WORK_CLAIMED", "work_item_id": subject_id, "worker_id": item.get("worker_id")}
            elif event_type == "WORK_RELEASED" and subject_id:
                item = work.setdefault(subject_id, {})
                item["status"] = payload.get("next_status", "READY")
                worker_id = str(payload.get("worker_id") or item.get("worker_id") or "")
                if worker_id:
                    workers[worker_id] = "AVAILABLE"
                independent = alternatives(item.get("branch_id"))
                if item["status"] == "WAITING_DEPENDENCY" and independent:
                    missed_independent += 1
                point = {"kind": "WORK_RELEASED", "work_item_id": subject_id, "worker_id": worker_id or None,
                         "released_to": item["status"], "independent_ready_work_item_ids": independent}
            elif event_type in {"WORK_ITEM_INVALIDATED", "WORK_ITEM_SUPERSEDED", "WORK_ITEM_EQUIVALENCE_SUPERSEDED"} and subject_id:
                work.setdefault(subject_id, {})["status"] = "INVALIDATED" if event_type == "WORK_ITEM_INVALIDATED" else "SUPERSEDED"
            elif event_type == "WORK_COMPLETED" and subject_id:
                work.setdefault(subject_id, {})["status"] = "COMPLETED"
                worker_id = str(payload.get("worker_id") or "")
                if worker_id:
                    workers[worker_id] = "AVAILABLE"
            elif event_type == "WORK_ITEM_EXPERIMENT_ATTACHED" and subject_id:
                work.setdefault(subject_id, {})["status"] = "RUNNING_DETACHED"
                worker_id = str(payload.get("worker_id") or "")
                if worker_id:
                    workers[worker_id] = "AVAILABLE"
            elif event_type == "HYPOTHESIS_BRANCH_CREATED" and subject_id:
                point = {"kind": "BRANCH_CREATED", "branch_id": subject_id, "state": payload.get("state", "OPEN")}
            elif event_type == "HYPOTHESIS_BRANCH_TRANSITIONED" and subject_id:
                point = {"kind": "BRANCH_TRANSITION", "branch_id": subject_id, "state": payload.get("state", "UNKNOWN")}

            if point is not None:
                point.update({
                    "at": event["created_at"], "event_type": event_type,
                    "known_available_worker_ids": sorted(worker_id for worker_id, status in workers.items() if status == "AVAILABLE"),
                    "known_ready_authoritative_work_item_ids": sorted(
                        item_id for item_id, item in work.items()
                        if item.get("status") == "READY" and item.get("authority_status") == "AUTHORITATIVE"
                    ),
                })
                schedule_points.append(point)

        return {
            "replay_version": "v3.6-historical-read-only", "project_id": project_id,
            "events_examined": len(events), "schedule_points": schedule_points,
            "summary": {"waiting_releases_with_independent_ready_opportunity": missed_independent,
                        "duplicate_equivalence_signals": duplicate_signals,
                        "observed_available_workers_at_end": sorted(worker_id for worker_id, status in workers.items() if status == "AVAILABLE")},
            "limitations": {"unknown_branch_event_count": unknown_branch_events,
                            "counterfactual_claim": "No suggested assignment is treated as evidence that an unexecuted branch would have succeeded.",
                            "historical_fidelity": "Only event payload present at each event is used; pre-v3.6 events without branch metadata remain unknown."},
            "mutated": False,
        }

    def portfolio_production_audit(self, project_id: str) -> dict:
        """Read-only v3.6 coordination audit; no scheduler mutation or planning."""
        coverage = self.branch_coverage_get(project_id)
        ready = [item for item in self.work_list(project_id, ["READY"], 500) if self._scheduler_actionable_work(item)]
        workers = self._project_scheduler_workers(project_id)
        with self.store._connect() as conn, conn.cursor() as cur:
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
            "objective_global_blocks": [item for item in waiting_rows if item.get("dependency_scope") == "OBJECTIVE_GLOBAL"],
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
        claim an existing safe READY WorkItem, or persist an explicit
        idle reason.  `claim_work` remains the concurrency guard, so concurrent
        scheduler processes cannot assign the same item twice.
        """
        if not self.feature_flags.get("BRANCH_AWARE_ASSIGNMENT"):
            raise GPUError("PORTFOLIO_SCHEDULER_DISABLED", "Set BRANCH_AWARE_ASSIGNMENT=true after shadow review")
        with self.store._connect() as conn, conn.cursor() as cur:
            worker = self._worker(cur, worker_id)
            self._session(cur, session_id, worker_id, project_id)
            worker = self._reconcile_worker_availability(cur, worker_id, self._now())
            if worker.get("availability_state") not in {"AVAILABLE", "IDLE"}:
                raise GPUError("LAB_WORKER_NOT_AVAILABLE", str(worker.get("availability_state")))

        global_blocks = self._objective_global_blocks(project_id)
        if global_blocks:
            now = self._now()
            idle_reason = "OBJECTIVE_GLOBAL_DEPENDENCY_BLOCK"
            with self.store._connect() as conn, conn.cursor() as cur:
                self._session(cur, session_id, worker_id, project_id)
                cur.execute(
                    "UPDATE research_workers SET availability_state='IDLE',availability_updated_at=%s,idle_reason=%s,idle_since=%s WHERE id=%s",
                    (now, idle_reason, now, worker_id),
                )
            plan = self._parallel_plan_record(
                project_id, None, [{"worker_id": worker_id, "session_id": session_id, "work_item_id": None}],
                {"reason": idle_reason, "global_block_work_item_ids": [item["id"] for item in global_blocks], "new_work_created": False},
            )
            return {"status": "IDLE", "idle_reason": idle_reason, "global_blocks": global_blocks, "plan": plan}

        coverage = {entry["branch_id"]: entry for entry in self.branch_coverage_get(project_id)}
        ready = [
            item for item in self.work_list(project_id, ["READY"], limit)
            if self._scheduler_actionable_work(item)
            and self._worker_can_claim_item(worker, item)
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
            # Affinity preserves useful context when possible, but it is never
            # an ownership lock: any eligible available worker may still claim.
            affinity = 0 if str(item.get("preferred_worker_id") or "") == str(worker_id) else 1
            actionability = 0 if branch.get("next_actionability") == "READY_EXISTING_WORK" else 1
            return (concentration, affinity, actionability, -float(item.get("priority") or 0), str(item["id"]))

        chosen: dict | None = None
        for candidate in sorted(ready, key=rank):
            try:
                chosen = self.claim_work(candidate["id"], worker_id, session_id)
                break
            except GPUError as error:
                if error.error_type not in {
                    "LAB_WORK_NOT_CLAIMABLE", "LAB_WORK_DEPENDENCY_UNSATISFIED",
                    "LAB_WORK_DEPENDENCY_INVALIDATED", "DISCOVERY_ROUND_STALE",
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

        planner_candidates = self.work_planner_candidates(project_id)
        planner_needed = (
            self.feature_flags.get("PLANNER_ON_IDLE")
            and planner_candidates["planner_action"] == "CONSIDER_PROPOSAL"
        )
        now = self._now()
        idle_reason = "PLANNER_EVALUATION_REQUIRED" if planner_needed else "NO_EXISTING_ACTIONABLE_CANONICAL_WORK"
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, session_id, worker_id, project_id)
            cur.execute(
                "UPDATE research_workers SET availability_state='IDLE',availability_updated_at=%s,idle_reason=%s,idle_since=%s WHERE id=%s",
                (now, idle_reason, now, worker_id),
            )
        plan = self._parallel_plan_record(
            project_id, None, [{"worker_id": worker_id, "session_id": session_id, "work_item_id": None}],
            {"reason": idle_reason, "new_work_created": False,
             "planner_action": "REQUEST_CENTRAL_PLANNER_EVALUATION" if planner_needed else "IDLE",
             "undercovered_branch_ids": [entry["branch_id"] for entry in planner_candidates["undercovered_branches"]] if planner_needed else [],
             "unbranched_agenda_item_ids": [entry["agenda_item_id"] for entry in planner_candidates["unbranched_agenda_items"]] if planner_needed else []},
        )
        return {"status": "IDLE", "idle_reason": idle_reason, "plan": plan,
                "planner_action": "REQUEST_CENTRAL_PLANNER_EVALUATION" if planner_needed else "IDLE",
                "planner_candidates": planner_candidates if planner_needed else None}

    def canonical_execution_projection(self, project_id: str) -> dict:
        """Separate current canonical execution from physically real historical runs."""
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM lab_work_items WHERE project_id=%s AND related_refs ? 'experiment_run_id' ORDER BY updated_at DESC", (project_id,))
            current, historical = [], []
            for item in cur.fetchall():
                entry = {"work_item_id": str(item["id"]), "run_id": item["related_refs"].get("experiment_run_id"), "subject_id": item["subject_id"], "subject_version": item["canonical_subject_version"], "physical_work_status": item["status"], "canonical": item["status"] not in {"SUPERSEDED", "INVALIDATED"} and item["authority_status"] == "AUTHORITATIVE", "superseded_by": str(item["superseded_by"]) if item["superseded_by"] else None}
                (current if entry["canonical"] else historical).append(entry)
            return {"current_canonical_runs": current, "historical_or_noncanonical_runs": historical}

    def evidence_need_lookup(self, project_id: str, evidence_need: dict[str, Any]) -> dict[str, Any]:
        """Read the canonical duplicate/terminal state for one EvidenceNeed."""
        normalized = self._evidence_need_normalize(evidence_need)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT n.*,w.status,w.id AS canonical_work_item_id FROM lab_evidence_needs n "
                "JOIN lab_work_items w ON w.id=n.work_item_id "
                "WHERE n.project_id=%s AND n.need_key=%s AND n.exception_key=%s",
                (project_id, normalized["need_key"], normalized["exception_key"]),
            )
            row = cur.fetchone()
            if not row:
                return {"result": "NOT_FOUND", "need_key": normalized["need_key"]}
            item = self._record(row) or {}
            result = "REUSED_ACTIVE" if item["status"] in ACTIVE_WORK_STATUSES else "ALREADY_SATISFIED" if item["status"] == "COMPLETED" else "RETRY_ALLOWED"
            return {"result": result, "need_key": normalized["need_key"], "evidence_need": item}

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

    def consistency_audit_v365(self, project_id: str, persist: bool = False) -> dict[str, Any]:
        """Detect semantic authority contradictions, including legacy mutations."""
        detected: list[dict[str, Any]] = []
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT w.id AS work_item_id,w.subject_id,w.canonical_subject_version,w.promotion_state,"
                "r.id AS review_result_id,r.review_work_item_id,r.verdict "
                "FROM lab_work_items w JOIN lab_review_results r ON r.project_id=w.project_id "
                "AND r.target_subject_id=w.subject_id AND r.target_subject_version=w.canonical_subject_version "
                "WHERE w.project_id=%s AND w.authority_class='AUTHORITATIVE' "
                "AND w.promotion_state='PROMOTED' AND r.verdict NOT IN ('PASS','PASS_WITH_NONMATERIAL_NOTES')",
                (project_id,),
            )
            for row in cur.fetchall():
                detected.append({
                    "conflict_type": "VERDICT_AUTHORITY_MISMATCH",
                    "work_item_id": str(row["work_item_id"]), "review_result_id": str(row["review_result_id"]),
                    "review_work_item_id": str(row["review_work_item_id"]), "verdict": row["verdict"],
                    "detail": "Promoted authority is incompatible with a non-PASS review verdict.",
                })
            cur.execute(
                "SELECT downstream.id AS work_item_id,r.id AS review_result_id,r.review_work_item_id,r.verdict "
                "FROM lab_work_dependencies d JOIN lab_work_items downstream ON downstream.id=d.work_item_id "
                "JOIN lab_work_items review_work ON review_work.id=d.target_id::uuid "
                "JOIN lab_review_results r ON r.review_work_item_id=review_work.id AND r.project_id=downstream.project_id "
                "WHERE downstream.project_id=%s AND d.target_type='WORK_ITEM' "
                "AND review_work.status='COMPLETED' AND r.verdict NOT IN ('PASS','PASS_WITH_NONMATERIAL_NOTES') "
                "AND downstream.status IN ('READY','CLAIMED','RUNNING','RUNNING_DETACHED')",
                (project_id,),
            )
            for row in cur.fetchall():
                detected.append({
                    "conflict_type": "COMPLETED_REVIEW_NONPASS_RELEASE",
                    "work_item_id": str(row["work_item_id"]), "review_result_id": str(row["review_result_id"]),
                    "review_work_item_id": str(row["review_work_item_id"]), "verdict": row["verdict"],
                    "detail": "Downstream work is active because review completion was used instead of its verdict.",
                })
            cur.execute(
                "SELECT scientific_object_id,array_agg(id::text ORDER BY created_at) AS gate_ids,"
                "array_agg(canonical_subject_version ORDER BY created_at) AS versions "
                "FROM scientific_gates WHERE project_id=%s AND status='PASS' AND superseded_by IS NULL "
                "AND invalidation_reason IS NULL AND authoritative_work_item_id IS NOT NULL "
                "GROUP BY scientific_object_id HAVING count(*)>1",
                (project_id,),
            )
            for row in cur.fetchall():
                detected.append({
                    "conflict_type": "DUPLICATE_EXECUTION_AUTHORITY",
                    "work_item_id": str(row["scientific_object_id"]),
                    "gate_ids": list(row["gate_ids"]), "versions": list(row["versions"]),
                    "detail": "More than one unsuperseded PASS gate has execution authority for this scientific object.",
                })
            cur.execute(
                "SELECT id FROM lab_work_items WHERE project_id=%s AND authority_class IN ('SPECULATIVE','PREPARATORY') "
                "AND promotion_state='PROMOTED'",
                (project_id,),
            )
            for row in cur.fetchall():
                detected.append({
                    "conflict_type": "SPECULATIVE_OUTPUT_PROMOTED_EARLY",
                    "work_item_id": str(row["id"]),
                    "detail": "Speculative/preparatory work may not become promoted authority directly.",
                })
            if persist:
                for conflict in detected:
                    identity = str(conflict.get("review_result_id") or ":".join(conflict.get("gate_ids") or []))
                    key = f"v365:{conflict['conflict_type']}:{conflict['work_item_id']}:{identity}"
                    cur.execute(
                        "INSERT INTO lab_consistency_conflicts(id,project_id,conflict_key,conflict_type,status,object_ids,details,detected_at) "
                        "VALUES(%s,%s,%s,%s,'OPEN',%s,%s,%s) "
                        "ON CONFLICT(project_id,conflict_key) DO UPDATE SET status='OPEN',details=EXCLUDED.details,detected_at=EXCLUDED.detected_at,resolved_at=NULL",
                        (str(uuid.uuid4()), project_id, key, conflict["conflict_type"],
                         json.dumps([conflict["work_item_id"], *conflict.get("gate_ids", []), *([conflict["review_result_id"]] if conflict.get("review_result_id") else [])]), json.dumps(conflict), now),
                    )
                    try:
                        uuid.UUID(str(conflict["work_item_id"]))
                        event_subject_id: str | None = str(conflict["work_item_id"])
                    except (ValueError, TypeError):
                        # An experiment/subject identity can be an external
                        # immutable ID rather than a ResearchObject UUID.
                        event_subject_id = None
                    self._event(cur, project_id, "V365_CONSISTENCY_CONFLICT_DETECTED", event_subject_id, conflict)
        return {"audit_version": "v3.6.5", "persisted": persist, "conflicts": detected, "conflict_count": len(detected)}

    def _lab_budget_get(self, project_id: str) -> dict[str, Any]:
        with self.store._connect() as conn, conn.cursor() as cur:
            return self._budget(cur, project_id)
