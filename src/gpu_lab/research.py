import hashlib
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .errors import GPUError

logger = logging.getLogger(__name__)

RESEARCH_OBJECT_KINDS = (
    "Paper",
    "EvidenceUnit",
    "EvidenceFamily",
    "ExecutablePaper",
    "Claim",
    "Mechanism",
    "Anomaly",
    "Contradiction",
    "Hypothesis",
    "Prediction",
    "HypothesisLineage",
    "Experiment",
    "ExperimentRun",
    "Artifact",
    "Reproduction",
    "NegativeResult",
    "Lesson",
    "ResearchState",
    "ScientificVariable",
    "Phenomenon",
    "MechanismState",
    "CausalEdge",
    "InformationPath",
    "Assumption",
    "Scope",
    "Intervention",
    "ExpectedEffect",
    "WorldModel",
    "WorldModelVersion",
    "ResearchAgenda",
    "AgendaItem",
    "HypothesisPortfolio",
    "CandidatePortfolio",
    "BreakthroughSignal",
    "HypothesisNiche",
    "ExperimentBranch",
    "ExperimentNode",
    "BranchRelation",
    "ResearchDecision",
    "ResearchActionCandidate",
    "ActionApproval",
    "ComparativeLesson",
    "MetaLesson",
    "NullModel",
    "ResearchSituation",
    "ResearchDecisionOutcome",
    "ResearchStrategyPattern",
    "StrategyOutcome",
    "EngineeringTask",
    "EngineeringResult",
    "ResearchPolicy",
    "PolicyHypothesis",
    "ResearchPolicyPatch",
    "PolicyExperiment",
    "PolicyNegativeResult",
    "ResearchPolicyWeakness",
    "ImprovementRun",
    "ResearchPolicyArtifact",
    "MetaWorldModel",
    "MetaResearchAgenda",
    "ImprovementOpportunity",
    "PolicyAutonomyConfig",
    "PolicyRegression",
    "BenchmarkGap",
    "BenchmarkEpisodeProposal",
    "MetaStrategyPattern",
    "PolicyCanary",
    "PolicyShadowEvaluation",
    "PolicyHindsight",
    "PolicyUserFeedback",
    "PolicyEvaluationAudit",
    "PolicyRankerReadiness",
    "MetaResearchCampaign",
    "LiteratureScoutRequest",
    "LiteraturePolicyTransfer",
    "ProviderAdapterCandidate",
    "PolicyTournament",
)
RESEARCH_OBJECT_STATUSES = {
    "ACTIVE",
    "APPROVED",
    "BLOCKED",
    "CANDIDATE",
    "COMPLETED",
    "DEFERRED",
    "FAILED",
    "IMPLEMENTED_UNVERIFIED",
    "INCONCLUSIVE",
    "OPEN",
    "PARTIAL",
    "PENDING_EVIDENCE_REVIEW",
    "PREPARED",
    "PROPOSED",
    "REFUTED",
    "REPRODUCED",
    "RESERVED",
    "RESOLVED",
    "RESULT_INSPECTED",
    "RESULT_NOT_INSPECTED",
    "RUNNING",
    "SELECTED",
    "SUPPORTED",
    "SURVIVES_INITIAL_TEST",
    "VERIFIED_INTEGRATION",
    "VERIFIED_REAL",
    "VERIFIED_UNIT",
    "WEAKENED",
    "DRAFT",
    "UNDER_EVALUATION",
    "SUPPORTED_ON_BENCHMARK",
    "CROSS_PROJECT_SUPPORTED",
    "CROSS_MODEL_SUPPORTED",
    "REJECTED",
    "SUPERSEDED",
    "ROLLED_BACK",
    "DEPRECATED",
    "MODEL_INCOMPATIBLE",
    "DOMAIN_RESTRICTED",
    "PRODUCTION",
    "OVERFIT",
    "INVALID_EVALUATION",
    "PROJECT_SPECIFIC",
    "MODEL_SENSITIVE",
    "RECOMMENDED_FOR_PROMOTION",
    "cancelled",
    "completed",
    "failed",
    "running",
    "unknown",
    "TECHNICAL_CANCELLED",
    "TECHNICAL_ORPHANED",
}

# Orthogonal epistemic classifications. These are metadata about the role and
# quality of a record; they never rewrite the underlying observation.
DECISION_ROLES = {
    "SCIENTIFIC_ACTION",
    "RESULT_INSPECTION",
    "REPRODUCTION_ACTION",
    "ADMINISTRATIVE_RECOVERY",
    "LEGACY_BACKFILL",
    "SYSTEM_VERIFICATION",
    "PROVIDER_CONTRACT_TEST",
    "BENCHMARK_EVALUATION",
}
EXECUTION_VERIFICATIONS = {"NOT_EXECUTED", "MOCK", "UNIT", "INTEGRATION", "REAL_CPU", "REAL_GPU"}
SCIENTIFIC_ROLES = {
    "NOT_SCIENTIFIC",
    "SYSTEM_SMOKE",
    "CONTRACT_TEST",
    "REPRODUCTION",
    "DIAGNOSTIC",
    "NULL_TEST",
    "CAUSAL_TEST",
    "REPLICATION",
    "GENERALIZATION",
    "TRAINING_EVALUATION",
}
SCIENTIFIC_VERIFICATIONS = {
    "NOT_SCIENTIFIC",
    "PREREQUISITE_ONLY",
    "RESULT_UNINSPECTED",
    "RESULT_INSPECTED",
    "SUPPORTED_WITHIN_SCOPE",
    "REPLICATED_WITHIN_SCOPE",
    "GENERALIZED",
}
CYCLE_STATUSES = {"OPEN", "SELECTED", "EXECUTED", "INSPECTED", "ASSESSED", "CLOSED", "BLOCKED", "ABANDONED"}
LEARNING_NAMESPACES = {"PRODUCTION_SCIENCE", "BENCHMARK", "SYSTEM_TEST"}
STRATEGY_LEARNING_ROLES = {
    "EXCLUDE",
    "SCIENTIFIC_ACTION_OUTCOME",
    "EXPERIMENTAL_DESIGN_FAILURE",
    "RESOURCE_FAILURE",
    "REPRODUCTION_LESSON",
    "META_RESEARCH_ONLY",
}


def classify_decision_data(data: dict[str, Any]) -> dict[str, str]:
    """Deterministically classify a decision while preserving legacy fields."""
    selected = data.get("selected_action", {}) if isinstance(data.get("selected_action"), dict) else {}
    action = str(selected.get("action_type", data.get("action_type", "UNKNOWN"))).upper()
    process_only = isinstance(selected.get("payload"), dict) and (
        selected["payload"].get("non_executing_discovery_candidate") is True
        or selected["payload"].get("non_scientific_process_action") is True
        or selected["payload"].get("does_not_authorize_execution") is True
    )
    legacy = data.get("legacy_provenance")
    role = str(data.get("decision_role") or "").upper()
    if process_only or action == "PORTFOLIO_REFINEMENT":
        role = "ADMINISTRATIVE_RECOVERY"
    elif role not in DECISION_ROLES:
        if isinstance(legacy, dict) and legacy.get("reconstructed"):
            role = "LEGACY_BACKFILL"
        elif isinstance(selected.get("payload"), dict) and selected["payload"].get("mode") in {"INSPECT_RESULT", "INSPECT_FAILURE", "RECOVER_UNFINISHED"}:
            role = "RESULT_INSPECTION"
        elif action in {"REPRODUCTION"}:
            role = "REPRODUCTION_ACTION"
        elif action in {"SYSTEM_SMOKE", "MCP_CONTRACT_TEST", "CONTRACT_TEST"}:
            role = "SYSTEM_VERIFICATION"
        else:
            # Pre-v2.1 unclassified history has no proof of prospective
            # selection. Keep it observable, but fail closed for strategy
            # learning until deterministic backfill can identify its role.
            role = "LEGACY_BACKFILL" if not data.get("brain_policy_version") else "SCIENTIFIC_ACTION"
    scientific_role = str(data.get("scientific_role") or "").upper()
    if process_only or action == "PORTFOLIO_REFINEMENT":
        scientific_role = "NOT_SCIENTIFIC"
    elif scientific_role not in SCIENTIFIC_ROLES:
        if role in {"LEGACY_BACKFILL", "ADMINISTRATIVE_RECOVERY", "SYSTEM_VERIFICATION", "PROVIDER_CONTRACT_TEST", "BENCHMARK_EVALUATION"}:
            scientific_role = "NOT_SCIENTIFIC"
        elif action == "REPRODUCTION":
            scientific_role = "REPRODUCTION"
        elif action in {"CAUSAL_INTERVENTION", "ABLATION", "STATE_SUBSTITUTION"}:
            scientific_role = "CAUSAL_TEST"
        elif action == "NULL_MODEL_TEST":
            scientific_role = "NULL_TEST"
        elif action in {"REPLICATION", "GENERALIZATION", "TRAINING_RUN"}:
            scientific_role = {"REPLICATION": "REPLICATION", "GENERALIZATION": "GENERALIZATION", "TRAINING_RUN": "TRAINING_EVALUATION"}[action]
        else:
            scientific_role = "DIAGNOSTIC" if role == "SCIENTIFIC_ACTION" else "NOT_SCIENTIFIC"
    namespace = str(data.get("learning_namespace") or "").upper()
    if namespace not in LEARNING_NAMESPACES:
        namespace = "PRODUCTION_SCIENCE" if role == "SCIENTIFIC_ACTION" else "SYSTEM_TEST"
    execution = str(data.get("execution_verification") or "").upper()
    if execution not in EXECUTION_VERIFICATIONS:
        execution = "NOT_EXECUTED"
    scientific_verification = str(data.get("scientific_verification") or "").upper()
    if scientific_verification not in SCIENTIFIC_VERIFICATIONS:
        scientific_verification = (
            "NOT_SCIENTIFIC"
            if scientific_role in {"NOT_SCIENTIFIC", "SYSTEM_SMOKE", "CONTRACT_TEST"}
            else "RESULT_UNINSPECTED"
        )
    cycle = str(data.get("cycle_status") or "").upper()
    if cycle not in CYCLE_STATUSES:
        cycle = "CLOSED" if data.get("decision_outcome_id") else "SELECTED"
    return {
        "decision_role": role,
        "scientific_role": scientific_role,
        "execution_verification": execution,
        "scientific_verification": scientific_verification,
        "cycle_status": cycle,
        "learning_namespace": namespace,
    }


def strategy_learning_eligibility(decision: dict[str, Any], outcome: dict[str, Any] | None) -> dict[str, Any]:
    data = decision.get("data", {})
    if not isinstance(data, dict) or not data:
        return {
            "eligible": False,
            "reasons": [],
            "exclusions": ["DECISION_CONTEXT_MISSING"],
            "decision_role": "LEGACY_BACKFILL",
            "scientific_role": "NOT_SCIENTIFIC",
            "execution_verification": "NOT_EXECUTED",
            "scientific_verification": "NOT_SCIENTIFIC",
            "cycle_status": "OPEN",
            "learning_namespace": "SYSTEM_TEST",
        }
    classification = classify_decision_data(data)
    reasons: list[str] = []
    if classification["decision_role"] != "SCIENTIFIC_ACTION":
        reasons.append("DECISION_ROLE_NOT_SCIENTIFIC_ACTION")
    if classification["learning_namespace"] != "PRODUCTION_SCIENCE":
        reasons.append("LEARNING_NAMESPACE_NOT_PRODUCTION_SCIENCE")
    if outcome is None:
        reasons.append("DECISION_OUTCOME_MISSING")
    else:
        outcome_data = outcome.get("data", {})
        if classification["cycle_status"] != "CLOSED":
            reasons.append("DECISION_CYCLE_NOT_CLOSED")
        if str(outcome.get("status")) not in {"RESULT_INSPECTED", "ASSESSED", "CLOSED"}:
            reasons.append("OUTCOME_NOT_INSPECTED")
        if outcome_data.get("realized_information_gain") in {None, "UNKNOWN"}:
            reasons.append("REALIZED_INFORMATION_GAIN_UNKNOWN")
        if not outcome_data.get("hindsight_assessment"):
            reasons.append("HINDSIGHT_MISSING")
        basis = outcome_data.get("information_gain_basis", [])
        if not basis:
            if outcome_data.get("uncertainties_resolved"):
                basis = ["UNCERTAINTY_RESOLVED"]
            elif outcome_data.get("hypotheses_eliminated"):
                basis = ["HYPOTHESIS_ELIMINATED"]
            elif outcome_data.get("evidence_family_ids"):
                basis = ["EVIDENCE_FAMILY_ADDED"]
        if not isinstance(basis, list) or not basis:
            reasons.append("INFORMATION_GAIN_BASIS_MISSING")
    if classification["scientific_role"] in {"NOT_SCIENTIFIC", "SYSTEM_SMOKE", "CONTRACT_TEST"}:
        reasons.append("SCIENTIFIC_ROLE_EXCLUDED")
    eligible = not reasons
    return {"eligible": eligible, "reasons": reasons if eligible else [], "exclusions": reasons if not eligible else [], **classification}


def _json_document(value: Any) -> Any:
    """Convert PostgreSQL-native values to a JSON document before persistence."""
    return json.loads(json.dumps(value, default=str))


class _TrackedResearchConnection:
    """Finalize a durable snapshot boundary only after the primary transaction commits."""

    def __init__(self, store: "ResearchStore"):
        self.store = store
        self.connection = psycopg.connect(store.url, row_factory=dict_row)

    def __enter__(self):
        self.connection.__enter__()
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        transaction_id = None
        if exc_type is None:
            with self.connection.cursor() as cur:
                cur.execute("SELECT txid_current_if_assigned() AS transaction_id")
                transaction_id = cur.fetchone()["transaction_id"]
        result = self.connection.__exit__(exc_type, exc_value, traceback)
        if exc_type is None and transaction_id is not None:
            try:
                self.store._finalize_transaction(int(transaction_id))
            except Exception:
                # The domain transaction is already committed. Never report it as failed and
                # invite a duplicate retry; pending temporal bookkeeping is recoverable.
                logger.exception(
                    "Temporal finalization deferred for committed transaction %s",
                    transaction_id,
                )
        return result


class ResearchStore:
    """PostgreSQL canonical scientific state with immutable event history."""

    def __init__(self, url: str):
        self.url = url
        self.vector_available = False
        self._migrate()
        self._finalize_orphaned_transactions()

    def _connect(self):
        return _TrackedResearchConnection(self)

    def _finalize_transaction(self, transaction_id: int) -> None:
        with psycopg.connect(self.url) as conn, conn.cursor() as cur:
            for table in (
                "research_object_versions",
                "research_project_versions",
                "research_events",
            ):
                cur.execute(
                    f"UPDATE {table} SET committed_at=clock_timestamp() "
                    "WHERE commit_token=%s AND committed_at IS NULL",
                    (transaction_id,),
                )

    def _finalize_orphaned_transactions(self) -> None:
        """Recover transactions committed before their post-commit finalizer completed."""
        with psycopg.connect(self.url) as conn, conn.cursor() as cur:
            for table in (
                "research_object_versions",
                "research_project_versions",
                "research_events",
            ):
                cur.execute(
                    f"UPDATE {table} SET committed_at=clock_timestamp(),legacy_backfill=TRUE "
                    "WHERE committed_at IS NULL"
                )

    def _migrate(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_research_migration'))")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.vector_available = True
            except (psycopg.errors.UndefinedFile, psycopg.errors.InsufficientPrivilege):
                conn.rollback()
                cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_research_migration'))")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_projects (
                    id UUID PRIMARY KEY, name TEXT UNIQUE NOT NULL, question TEXT NOT NULL,
                    state JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_objects (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE', data JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_edges (
                    source_id UUID NOT NULL REFERENCES research_objects(id), target_id UUID NOT NULL REFERENCES research_objects(id), relation TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL, PRIMARY KEY(source_id,target_id,relation)
                );
                CREATE TABLE IF NOT EXISTS research_events (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    event_type TEXT NOT NULL, subject_id UUID, payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    committed_at TIMESTAMPTZ,
                    commit_token BIGINT NOT NULL DEFAULT txid_current(),
                    legacy_backfill BOOLEAN NOT NULL DEFAULT FALSE
                );
                CREATE TABLE IF NOT EXISTS research_execution_attempts (
                    experiment_id UUID NOT NULL REFERENCES research_objects(id),
                    idempotency_key TEXT NOT NULL,
                    run_id UUID UNIQUE NOT NULL REFERENCES research_objects(id),
                    job_id TEXT UNIQUE NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY(experiment_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS research_events_project_created_idx
                    ON research_events(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS research_objects_project_kind_created_idx
                    ON research_objects(project_id, kind, created_at DESC);
                CREATE INDEX IF NOT EXISTS research_objects_data_gin_idx
                    ON research_objects USING GIN (data jsonb_path_ops);
                CREATE INDEX IF NOT EXISTS research_execution_attempts_job_idx
                    ON research_execution_attempts(job_id);
                ALTER TABLE research_objects
                    ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;
                ALTER TABLE research_objects
                    ADD COLUMN IF NOT EXISTS embedding_metadata JSONB;
                CREATE TABLE IF NOT EXISTS research_object_versions (
                    revision_id BIGSERIAL PRIMARY KEY,
                    object_id UUID NOT NULL REFERENCES research_objects(id),
                    project_id UUID NOT NULL REFERENCES research_projects(id),
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    data JSONB NOT NULL,
                    object_created_at TIMESTAMPTZ NOT NULL,
                    valid_from TIMESTAMPTZ NOT NULL,
                    embedding_text TEXT,
                    embedding_updated_at TIMESTAMPTZ,
                    embedding_metadata JSONB,
                    legacy_backfill BOOLEAN NOT NULL DEFAULT FALSE,
                    committed_at TIMESTAMPTZ,
                    commit_token BIGINT NOT NULL DEFAULT txid_current()
                );
                CREATE INDEX IF NOT EXISTS research_object_versions_temporal_idx
                    ON research_object_versions(project_id, valid_from, object_id, revision_id DESC);
                CREATE TABLE IF NOT EXISTS research_project_versions (
                    revision_id BIGSERIAL PRIMARY KEY,
                    project_id UUID NOT NULL REFERENCES research_projects(id),
                    name TEXT NOT NULL,
                    question TEXT NOT NULL,
                    state JSONB NOT NULL,
                    project_created_at TIMESTAMPTZ NOT NULL,
                    valid_from TIMESTAMPTZ NOT NULL,
                    legacy_backfill BOOLEAN NOT NULL DEFAULT FALSE,
                    committed_at TIMESTAMPTZ,
                    commit_token BIGINT NOT NULL DEFAULT txid_current()
                );
                CREATE INDEX IF NOT EXISTS research_project_versions_temporal_idx
                    ON research_project_versions(project_id, valid_from, revision_id DESC);
                ALTER TABLE research_object_versions ADD COLUMN IF NOT EXISTS committed_at TIMESTAMPTZ;
                ALTER TABLE research_object_versions ADD COLUMN IF NOT EXISTS commit_token BIGINT;
                ALTER TABLE research_project_versions ADD COLUMN IF NOT EXISTS committed_at TIMESTAMPTZ;
                ALTER TABLE research_project_versions ADD COLUMN IF NOT EXISTS commit_token BIGINT;
                ALTER TABLE research_events ADD COLUMN IF NOT EXISTS committed_at TIMESTAMPTZ;
                ALTER TABLE research_events ADD COLUMN IF NOT EXISTS commit_token BIGINT;
                ALTER TABLE research_events ADD COLUMN IF NOT EXISTS legacy_backfill BOOLEAN NOT NULL DEFAULT FALSE;
                CREATE INDEX IF NOT EXISTS research_object_versions_pending_commit_idx
                    ON research_object_versions(commit_token) WHERE committed_at IS NULL;
                CREATE INDEX IF NOT EXISTS research_project_versions_pending_commit_idx
                    ON research_project_versions(commit_token) WHERE committed_at IS NULL;
                CREATE INDEX IF NOT EXISTS research_events_pending_commit_idx
                    ON research_events(commit_token) WHERE committed_at IS NULL;
            """)
            migration_boundary = datetime.now(UTC)
            for table in (
                "research_object_versions",
                "research_project_versions",
                "research_events",
            ):
                cur.execute(
                    f"UPDATE {table} SET committed_at=%s,legacy_backfill=TRUE "
                    "WHERE committed_at IS NULL AND commit_token IS NULL",
                    (migration_boundary,),
                )
                cur.execute(
                    f"ALTER TABLE {table} ALTER COLUMN commit_token "
                    "SET DEFAULT txid_current()"
                )
            allowed_kinds = ",".join("'" + kind + "'" for kind in RESEARCH_OBJECT_KINDS)
            cur.execute(
                "SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint "
                "WHERE conrelid='research_objects'::regclass "
                "AND conname='research_objects_kind_check'"
            )
            constraint = cur.fetchone()
            definition = constraint["definition"] if constraint else ""
            if not all(f"'{kind}'" in definition for kind in RESEARCH_OBJECT_KINDS):
                cur.execute(
                    "ALTER TABLE research_objects DROP CONSTRAINT IF EXISTS "
                    "research_objects_kind_check"
                )
                cur.execute(
                    "ALTER TABLE research_objects ADD CONSTRAINT research_objects_kind_check "
                    f"CHECK(kind IN ({allowed_kinds}))"
                )
            cur.execute(
                "SELECT project_id,data->>'name' AS name,"
                "array_agg(id ORDER BY created_at,id) AS ids FROM research_objects "
                "WHERE kind='HypothesisNiche' AND data ? 'name' "
                "GROUP BY project_id,data->>'name' HAVING count(*)>1"
            )
            for duplicate_group in cur.fetchall():
                canonical_id = str(duplicate_group["ids"][0])
                for duplicate_id in duplicate_group["ids"][1:]:
                    cur.execute("SELECT data FROM research_objects WHERE id=%s", (duplicate_id,))
                    duplicate = cur.fetchone()
                    renamed_data = {
                        **duplicate["data"],
                        "name": f"{duplicate_group['name']} [duplicate {duplicate_id}]",
                        "duplicate_of": canonical_id,
                    }
                    cur.execute(
                        "UPDATE research_objects SET data=%s WHERE id=%s",
                        (json.dumps(renamed_data), duplicate_id),
                    )
                    self._event(
                        cur,
                        duplicate_group["project_id"],
                        "HYPOTHESIS_NICHE_DUPLICATE_RENAMED",
                        duplicate_id,
                        {"canonical_id": canonical_id, "name": renamed_data["name"]},
                    )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS research_objects_hypothesis_niche_name_unique "
                "ON research_objects(project_id, (data->>'name')) "
                "WHERE kind='HypothesisNiche'"
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS research_objects_evidence_family_key_unique "
                "ON research_objects(project_id, (data->>'independence_key')) "
                "WHERE kind='EvidenceFamily'"
            )
            cur.execute(
                "SELECT project_id,data->>'decision_id' AS decision_id,"
                "array_agg(id ORDER BY created_at,id) AS ids FROM research_objects "
                "WHERE kind='ResearchDecisionOutcome' AND data ? 'decision_id' "
                "GROUP BY project_id,data->>'decision_id' HAVING count(*)>1"
            )
            for duplicate_group in cur.fetchall():
                canonical_id = str(duplicate_group["ids"][0])
                for duplicate_id in duplicate_group["ids"][1:]:
                    cur.execute(
                        "SELECT data FROM research_objects WHERE id=%s FOR UPDATE",
                        (duplicate_id,),
                    )
                    duplicate = cur.fetchone()
                    if not duplicate:
                        continue
                    repaired_data = {
                        **duplicate["data"],
                        "duplicate_of": canonical_id,
                        "decision_id": None,
                        "legacy_duplicate_reconciled": True,
                    }
                    cur.execute(
                        "UPDATE research_objects SET data=%s,status='BLOCKED' WHERE id=%s",
                        (json.dumps(repaired_data), duplicate_id),
                    )
                    self._event(
                        cur,
                        duplicate_group["project_id"],
                        "DECISION_OUTCOME_DUPLICATE_RECONCILED",
                        duplicate_id,
                        {"canonical_id": canonical_id},
                    )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS research_objects_decision_outcome_unique "
                "ON research_objects((data->>'decision_id')) "
                "WHERE kind='ResearchDecisionOutcome'"
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS research_objects_strategy_key_unique "
                "ON research_objects((data->>'strategy_key')) "
                "WHERE kind='ResearchStrategyPattern'"
            )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS research_objects_single_production_policy_unique "
                "ON research_objects(project_id) WHERE kind='ResearchPolicy' AND status='PRODUCTION'"
            )
            cur.execute(
                "SELECT project_id,data->>'decision_outcome_id' AS outcome_id,"
                "data->>'scope_level' AS scope_level,array_agg(id ORDER BY created_at,id) AS ids "
                "FROM research_objects WHERE kind='StrategyOutcome' "
                "AND data ? 'decision_outcome_id' AND data ? 'scope_level' "
                "GROUP BY project_id,data->>'decision_outcome_id',data->>'scope_level' "
                "HAVING count(*)>1"
            )
            for duplicate_group in cur.fetchall():
                canonical_id = str(duplicate_group["ids"][0])
                for duplicate_id in duplicate_group["ids"][1:]:
                    cur.execute(
                        "SELECT data FROM research_objects WHERE id=%s FOR UPDATE",
                        (duplicate_id,),
                    )
                    duplicate = cur.fetchone()
                    if not duplicate:
                        continue
                    repaired_data = {
                        **duplicate["data"],
                        "duplicate_of": canonical_id,
                        "decision_outcome_id": None,
                        "legacy_duplicate_reconciled": True,
                    }
                    cur.execute(
                        "UPDATE research_objects SET data=%s,status='BLOCKED' WHERE id=%s",
                        (json.dumps(repaired_data), duplicate_id),
                    )
                    self._event(
                        cur,
                        duplicate_group["project_id"],
                        "STRATEGY_OUTCOME_DUPLICATE_RECONCILED",
                        duplicate_id,
                        {"canonical_id": canonical_id, "scope_level": duplicate_group["scope_level"]},
                    )
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS research_objects_strategy_outcome_unique "
                "ON research_objects((data->>'decision_outcome_id'), (data->>'scope_level')) "
                "WHERE kind='StrategyOutcome'"
            )
            if self.vector_available:
                cur.execute("ALTER TABLE research_objects ADD COLUMN IF NOT EXISTS embedding vector")
            cur.execute("""
                CREATE OR REPLACE FUNCTION gpu_lab_capture_research_object_version()
                RETURNS trigger AS $$
                BEGIN
                    INSERT INTO research_object_versions(
                        object_id,project_id,kind,status,data,object_created_at,valid_from,
                        embedding_text,embedding_updated_at,embedding_metadata,legacy_backfill,
                        committed_at,commit_token
                    ) VALUES(
                        NEW.id,NEW.project_id,NEW.kind,NEW.status,NEW.data,NEW.created_at,
                        clock_timestamp(),to_jsonb(NEW)->>'embedding',NEW.embedding_updated_at,
                        NEW.embedding_metadata,FALSE,NULL,txid_current()
                    );
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                DROP TRIGGER IF EXISTS research_objects_capture_version ON research_objects;
                CREATE TRIGGER research_objects_capture_version
                    AFTER INSERT OR UPDATE ON research_objects
                    FOR EACH ROW EXECUTE FUNCTION gpu_lab_capture_research_object_version();

                CREATE OR REPLACE FUNCTION gpu_lab_capture_research_project_version()
                RETURNS trigger AS $$
                BEGIN
                    INSERT INTO research_project_versions(
                        project_id,name,question,state,project_created_at,valid_from,legacy_backfill,
                        committed_at,commit_token
                    ) VALUES(
                        NEW.id,NEW.name,NEW.question,NEW.state,NEW.created_at,
                        clock_timestamp(),FALSE,NULL,txid_current()
                    );
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                DROP TRIGGER IF EXISTS research_projects_capture_version ON research_projects;
                CREATE TRIGGER research_projects_capture_version
                    AFTER INSERT OR UPDATE ON research_projects
                    FOR EACH ROW EXECUTE FUNCTION gpu_lab_capture_research_project_version();
            """)
            cur.execute("""
                INSERT INTO research_object_versions(
                    object_id,project_id,kind,status,data,object_created_at,valid_from,
                    embedding_text,embedding_updated_at,embedding_metadata,legacy_backfill,
                    committed_at,commit_token
                )
                SELECT o.id,o.project_id,o.kind,o.status,o.data,o.created_at,clock_timestamp(),
                       to_jsonb(o)->>'embedding',
                       CASE WHEN to_jsonb(o)->>'embedding' IS NOT NULL
                            THEN clock_timestamp() ELSE o.embedding_updated_at END,
                       o.embedding_metadata,TRUE,NULL,txid_current()
                FROM research_objects o
                WHERE NOT EXISTS(
                    SELECT 1 FROM research_object_versions v WHERE v.object_id=o.id
                );
                INSERT INTO research_project_versions(
                    project_id,name,question,state,project_created_at,valid_from,legacy_backfill,
                    committed_at,commit_token
                )
                SELECT p.id,p.name,p.question,p.state,p.created_at,clock_timestamp(),TRUE,
                       NULL,txid_current()
                FROM research_projects p
                WHERE NOT EXISTS(
                    SELECT 1 FROM research_project_versions v WHERE v.project_id=p.id
                );
            """)

    def project_create(self, name: str, question: str) -> dict:
        ident, now = uuid.uuid4(), datetime.now(UTC)
        state = {"research_question": question, "established_facts": [], "active_hypotheses": [], "next_discriminating_experiments": []}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO research_projects VALUES(%s,%s,%s,%s,%s)", (ident, name, question, json.dumps(state), now))
            self._event(cur, ident, "RESEARCH_PROJECT_CREATED", None, {"name": name, "question": question})
        return {"project_id": str(ident), "name": name, "state": state}

    @staticmethod
    def _normalize_as_of(as_of: datetime | str | None) -> datetime | None:
        if as_of is None:
            return None
        try:
            value = datetime.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        except (TypeError, ValueError) as exc:
            raise GPUError("INVALID_TEMPORAL_CUTOFF", str(as_of)) from exc
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise GPUError(
                "INVALID_TEMPORAL_CUTOFF", "Temporal cutoff must be an ISO timestamp with UTC offset"
            )
        return value.astimezone(UTC)

    def _prepare_as_of(self, as_of: datetime | str | None) -> datetime | None:
        return self._normalize_as_of(as_of)

    def temporal_finalize_pending(self) -> dict[str, Any]:
        """Explicit maintenance recovery; historical read tools remain side-effect free."""
        self._finalize_orphaned_transactions()
        with psycopg.connect(self.url, row_factory=dict_row) as conn, conn.cursor() as cur:
            pending = {}
            for table in (
                "research_object_versions",
                "research_project_versions",
                "research_events",
            ):
                cur.execute(f"SELECT count(*) AS count FROM {table} WHERE committed_at IS NULL")
                pending[table] = int(cur.fetchone()["count"])
        return {"status": "recovered", "pending": pending}

    def project_get(self, project_id: str, as_of: datetime | str | None = None) -> dict:
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                cur.execute(
                    "SELECT id,name,question,state,created_at FROM research_projects WHERE id=%s",
                    (project_id,),
                )
            else:
                cur.execute(
                    "SELECT project_id AS id,name,question,state,project_created_at AS created_at,"
                    "valid_from,committed_at,legacy_backfill "
                    "FROM research_project_versions WHERE project_id=%s "
                    "AND committed_at<=%s ORDER BY committed_at DESC,revision_id DESC LIMIT 1",
                    (project_id, cutoff),
                )
            project = cur.fetchone()
        if not project:
            if cutoff is not None:
                raise GPUError("TEMPORAL_SNAPSHOT_UNAVAILABLE", f"{project_id} at {cutoff.isoformat()}")
            raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
        return project

    def object_create(
        self,
        project_id: str,
        kind: str,
        data: dict[str, Any],
        event_type: str,
        status: str = "ACTIVE",
    ) -> dict:
        if kind not in RESEARCH_OBJECT_KINDS:
            raise GPUError("INVALID_RESEARCH_OBJECT_KIND", kind)
        self._validate_status(status)
        ident, now = uuid.uuid4(), datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s)",
                (ident, project_id, kind, status, json.dumps(data), now),
            )
            self._event(cur, project_id, event_type, ident, data)
        return {
            "id": str(ident),
            "project_id": project_id,
            "kind": kind,
            "status": status,
            "data": data,
        }

    def production_policy_ensure(self, project_id: str, data: dict[str, Any]) -> dict:
        """Return the sole production policy, creating it under a project lock."""
        ident, now = uuid.uuid4(), datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"production-policy:{project_id}",))
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects "
                "WHERE project_id=%s AND kind='ResearchPolicy' AND status='PRODUCTION' FOR UPDATE",
                (project_id,),
            )
            existing = cur.fetchone()
            if existing:
                return {**existing, "id": str(existing["id"])}
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ResearchPolicy','PRODUCTION',%s,%s)",
                (ident, project_id, json.dumps(data), now),
            )
            self._event(cur, project_id, "RESEARCH_POLICY_CREATED", ident, data)
        return {"id": str(ident), "project_id": project_id, "kind": "ResearchPolicy", "status": "PRODUCTION", "data": data}

    def production_policy_promote(self, project_id: str, current_policy_id: str, data: dict[str, Any]) -> dict:
        """Atomically supersede the current production policy and insert its successor."""
        ident, now = uuid.uuid4(), datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"production-policy:{project_id}",))
            cur.execute(
                "SELECT id FROM research_objects WHERE id=%s AND project_id=%s "
                "AND kind='ResearchPolicy' AND status='PRODUCTION' FOR UPDATE",
                (current_policy_id, project_id),
            )
            if not cur.fetchone():
                raise GPUError("POLICY_PRODUCTION_CONFLICT", current_policy_id)
            cur.execute("UPDATE research_objects SET status='SUPERSEDED' WHERE id=%s", (current_policy_id,))
            self._event(cur, project_id, "RESEARCH_POLICY_SUPERSEDED", current_policy_id, {})
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ResearchPolicy','PRODUCTION',%s,%s)",
                (ident, project_id, json.dumps(data), now),
            )
            self._event(cur, project_id, "RESEARCH_POLICY_PROMOTED", ident, data)
        return {"id": str(ident), "project_id": project_id, "kind": "ResearchPolicy", "status": "PRODUCTION", "data": data}

    def production_policy_rollback(self, project_id: str, current_policy_id: str, target_policy_id: str) -> dict:
        """Atomically restore a prior policy as the sole production policy."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"production-policy:{project_id}",))
            cur.execute("SELECT id FROM research_objects WHERE id=%s AND project_id=%s AND kind='ResearchPolicy' AND status='PRODUCTION' FOR UPDATE", (current_policy_id, project_id))
            if not cur.fetchone():
                raise GPUError("POLICY_PRODUCTION_CONFLICT", current_policy_id)
            cur.execute("SELECT id,data FROM research_objects WHERE id=%s AND project_id=%s AND kind='ResearchPolicy' FOR UPDATE", (target_policy_id, project_id))
            target = cur.fetchone()
            if not target:
                raise GPUError("INVALID_POLICY_ROLLBACK_TARGET", target_policy_id)
            cur.execute("UPDATE research_objects SET status='ROLLED_BACK' WHERE id=%s", (current_policy_id,))
            data = {**target["data"], "rollback_from_policy_id": str(current_policy_id)}
            cur.execute("UPDATE research_objects SET status='PRODUCTION',data=%s WHERE id=%s", (json.dumps(data), target_policy_id))
            self._event(cur, project_id, "RESEARCH_POLICY_ROLLED_BACK", current_policy_id, {"restored_policy_id": str(target_policy_id)})
            self._event(cur, project_id, "RESEARCH_POLICY_ROLLED_BACK", target_policy_id, {"rollback_from_policy_id": str(current_policy_id)})
        return {"id": str(target_policy_id), "project_id": project_id, "kind": "ResearchPolicy", "status": "PRODUCTION", "data": data}

    def evidence_family_create_atomic(
        self,
        project_id: str,
        data: dict[str, Any],
    ) -> dict:
        """Create one empirical-origin family, idempotently keyed within a project."""
        project_id = self._canonical_uuid(project_id)
        independence_key = str(data["independence_key"])
        ident, now = uuid.uuid4(), datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"evidence-family:{project_id}:{independence_key}",),
            )
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='EvidenceFamily' "
                "AND data->>'independence_key'=%s LIMIT 1 FOR UPDATE",
                (project_id, independence_key),
            )
            existing = cur.fetchone()
            if existing:
                comparable = {
                    key: existing["data"].get(key)
                    for key in (
                        "origin_id",
                        "independence_key",
                        "derived_from_evidence_family_id",
                    )
                }
                requested_comparable = {key: data.get(key) for key in comparable}
                if comparable != requested_comparable:
                    raise GPUError(
                        "EVIDENCE_FAMILY_KEY_CONFLICT",
                        "The independence key is already bound to a different empirical origin",
                    )
                roles = list(
                    dict.fromkeys(
                        [
                            *existing["data"].get("origin_roles", []),
                            *data.get("origin_roles", []),
                        ]
                    )
                )
                if roles != existing["data"].get("origin_roles", []):
                    existing_data = {**existing["data"], "origin_roles": roles}
                    cur.execute(
                        "UPDATE research_objects SET data=%s WHERE id=%s",
                        (json.dumps(existing_data), existing["id"]),
                    )
                    self._event(
                        cur,
                        project_id,
                        "EVIDENCE_FAMILY_ROLE_ADDED",
                        existing["id"],
                        {"origin_roles": roles},
                    )
                    existing = {**existing, "data": existing_data}
                return {**existing, "idempotent_replay": True}
            stored = {
                **data,
                "supporting_entity_ids": [],
                "contradicting_entity_ids": [],
                "derived_record_ids": [],
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'EvidenceFamily','ACTIVE',%s,%s)",
                (ident, project_id, json.dumps(stored), now),
            )
            self._event(cur, project_id, "EVIDENCE_FAMILY_CREATED", ident, data)
        return {
            "id": str(ident),
            "project_id": project_id,
            "kind": "EvidenceFamily",
            "status": "ACTIVE",
            "data": stored,
        }

    def evidence_family_link_atomic(
        self, family_id: str, entity_id: str, relationship: str
    ) -> dict:
        """Attach one derived record without manufacturing a second empirical origin."""
        family_id = self._canonical_uuid(family_id)
        entity_id = self._canonical_uuid(entity_id)
        fields = {
            "SUPPORTS": ("supporting_entity_ids", "supporting_evidence_family_ids"),
            "CONTRADICTS": (
                "contradicting_entity_ids",
                "contradicting_evidence_family_ids",
            ),
            "DERIVED": ("derived_record_ids", "evidence_family_ids"),
        }
        if relationship not in fields:
            raise GPUError("INVALID_EVIDENCE_FAMILY_RELATIONSHIP", relationship)
        family_field, entity_field = fields[relationship]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE id=ANY(%s::uuid[]) FOR UPDATE",
                ([family_id, entity_id],),
            )
            records = {str(item["id"]): item for item in cur.fetchall()}
            if len(records) != 2:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", "EvidenceFamily or linked entity")
            family, entity = records[family_id], records[entity_id]
            if family["kind"] != "EvidenceFamily":
                raise GPUError("NOT_AN_EVIDENCEFAMILY", family_id)
            if family["project_id"] != entity["project_id"]:
                raise GPUError("RESEARCH_PROJECT_MISMATCH", entity_id)
            if entity["kind"] == "CausalEdge" and relationship in {"SUPPORTS", "CONTRADICTS"}:
                raise GPUError(
                    "CAUSAL_EVIDENCE_FAMILY_REQUIRES_RESULT_ASSESSMENT",
                    "Causal support families are attached only by atomic result assessment",
                )
            family_data = {**family["data"]}
            entity_data = {**entity["data"]}
            already_linked = (
                entity_id in family_data.get(family_field, [])
                and family_id in entity_data.get(entity_field, [])
                and entity_id in family_data.get("derived_record_ids", [])
                and family_id in entity_data.get("evidence_family_ids", [])
            )
            if already_linked:
                return {
                    "family": family,
                    "entity": entity,
                    "relationship": relationship,
                    "idempotent_replay": True,
                }
            family_data[family_field] = list(
                dict.fromkeys([*family_data.get(family_field, []), entity_id])
            )
            family_data["derived_record_ids"] = list(
                dict.fromkeys([*family_data.get("derived_record_ids", []), entity_id])
            )
            entity_data[entity_field] = list(
                dict.fromkeys([*entity_data.get(entity_field, []), family_id])
            )
            entity_data["evidence_family_ids"] = list(
                dict.fromkeys([*entity_data.get("evidence_family_ids", []), family_id])
            )
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(family_data), family_id),
            )
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(entity_data), entity_id),
            )
            cur.execute(
                "INSERT INTO research_edges VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (family_id, entity_id, relationship, datetime.now(UTC)),
            )
            self._event(
                cur,
                family["project_id"],
                "EVIDENCE_FAMILY_RECORD_LINKED",
                family_id,
                {"entity_id": entity_id, "relationship": relationship},
            )
        return {
            "family": {**family, "data": family_data},
            "entity": {**entity, "data": entity_data},
            "relationship": relationship,
            "idempotent_replay": False,
        }

    def hypothesis_create_with_edges(
        self,
        project_id: str,
        data: dict[str, Any],
        edges: list[tuple[str, str]],
    ) -> dict:
        """Atomically persist a QD hypothesis, its lineage, niche, and proximity edges."""
        ident, now = uuid.uuid4(), datetime.now(UTC)
        source_ids = sorted({source_id for source_id, _relation in edges})
        with self._connect() as conn, conn.cursor() as cur:
            if source_ids:
                cur.execute(
                    "SELECT id,project_id FROM research_objects WHERE id=ANY(%s::uuid[]) FOR SHARE",
                    (source_ids,),
                )
                sources = cur.fetchall()
                if len(sources) != len(source_ids):
                    raise GPUError("RESEARCH_OBJECT_NOT_FOUND", "One or more QD edge sources")
                if any(str(item["project_id"]) != str(project_id) for item in sources):
                    raise GPUError("RESEARCH_PROJECT_MISMATCH", "QD edges must remain in one project")
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'Hypothesis','ACTIVE',%s,%s)",
                (ident, project_id, json.dumps(data), now),
            )
            for source_id, relation in edges:
                cur.execute(
                    "INSERT INTO research_edges VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (source_id, ident, relation, now),
                )
            self._event(cur, project_id, "QD_HYPOTHESIS_CREATED", ident, data)
        return {
            "id": str(ident),
            "project_id": project_id,
            "kind": "Hypothesis",
            "status": "ACTIVE",
            "data": data,
        }

    def hypothesis_niche_create(
        self,
        project_id: str,
        name: str,
        description: str,
        diversity_signature: dict[str, Any],
    ) -> dict:
        """Serialize niche identity creation across gateway processes with an advisory lock."""
        project_id = self._canonical_uuid(project_id)
        ident, now = uuid.uuid4(), datetime.now(UTC)
        data = {
            "name": name,
            "description": description,
            "active_best_hypothesis_id": None,
            "diversity_signature": diversity_signature,
        }
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"hypothesis-niche:{project_id}:{name}",),
            )
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='HypothesisNiche' AND data->>'name'=%s "
                "ORDER BY created_at LIMIT 1",
                (project_id, name),
            )
            existing = cur.fetchone()
            if existing:
                if (
                    existing["data"].get("description") != description
                    or existing["data"].get("diversity_signature") != diversity_signature
                ):
                    raise GPUError(
                        "HYPOTHESIS_NICHE_CONFLICT",
                        "A niche with this name already has a different definition",
                    )
                return {**existing, "idempotent_replay": True}
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'HypothesisNiche','ACTIVE',%s,%s)",
                (ident, project_id, json.dumps(data), now),
            )
            self._event(cur, project_id, "HYPOTHESIS_NICHE_CREATED", ident, data)
        return {
            "id": str(ident),
            "project_id": project_id,
            "kind": "HypothesisNiche",
            "status": "ACTIVE",
            "data": data,
        }

    def hypothesis_niche_set_best(
        self, niche_id: str, hypothesis_id: str, rationale: str
    ) -> dict:
        """Lock and revalidate niche membership before selecting its representative."""
        niche_id = self._canonical_uuid(niche_id)
        hypothesis_id = self._canonical_uuid(hypothesis_id)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE id=ANY(%s::uuid[]) FOR UPDATE",
                ([niche_id, hypothesis_id],),
            )
            items = {str(item["id"]): item for item in cur.fetchall()}
            if len(items) != 2:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", "Niche or hypothesis")
            niche, hypothesis = items[niche_id], items[hypothesis_id]
            if niche["kind"] != "HypothesisNiche" or hypothesis["kind"] != "Hypothesis":
                raise GPUError("INVALID_HYPOTHESIS_NICHE_MEMBER", hypothesis_id)
            if str(niche["project_id"]) != str(hypothesis["project_id"]):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", hypothesis_id)
            if str(hypothesis["data"].get("niche_id")) != niche_id:
                raise GPUError("HYPOTHESIS_NOT_IN_NICHE", hypothesis_id)
            if hypothesis["status"] not in {"ACTIVE", "SURVIVES_INITIAL_TEST", "SUPPORTED"}:
                raise GPUError("HYPOTHESIS_NOT_ACTIVE", hypothesis["status"])
            data = {
                **niche["data"],
                "active_best_hypothesis_id": hypothesis_id,
                "selection_rationale": rationale,
            }
            cur.execute("UPDATE research_objects SET data=%s WHERE id=%s", (json.dumps(data), niche_id))
            self._event(
                cur,
                niche["project_id"],
                "HYPOTHESIS_NICHE_BEST_CHANGED",
                niche_id,
                {"active_best_hypothesis_id": hypothesis_id, "selection_rationale": rationale},
            )
        return {**niche, "data": data}

    def experiment_branch_node_create(
        self,
        project_id: str,
        branch_id: str,
        data: dict[str, Any],
        parent_node_id: str | None,
        experiment_id: str | None,
    ) -> dict:
        """Atomically persist one branch node, its typed relation, edges, and events."""
        project_id = self._canonical_uuid(project_id)
        branch_id = self._canonical_uuid(branch_id)
        parent_node_id = self._canonical_uuid(parent_node_id) if parent_node_id else None
        experiment_id = self._canonical_uuid(experiment_id) if experiment_id else None
        data = {
            **data,
            "branch_id": branch_id,
            "parent_node_id": parent_node_id,
            "experiment_id": experiment_id,
        }
        node_id, relation_id, now = uuid.uuid4(), uuid.uuid4(), datetime.now(UTC)
        reference_ids = [branch_id, *([parent_node_id] if parent_node_id else []), *([experiment_id] if experiment_id else [])]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,data FROM research_objects "
                "WHERE id=ANY(%s::uuid[]) FOR SHARE",
                (reference_ids,),
            )
            references = {str(item["id"]): item for item in cur.fetchall()}
            if len(references) != len(set(reference_ids)):
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", "One or more branch references")
            if any(str(item["project_id"]) != str(project_id) for item in references.values()):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", "Branch references must share a project")
            if references[branch_id]["kind"] != "ExperimentBranch":
                raise GPUError("NOT_AN_EXPERIMENT_BRANCH", branch_id)
            if parent_node_id:
                parent = references[parent_node_id]
                if parent["kind"] != "ExperimentNode" or str(parent["data"].get("branch_id")) != branch_id:
                    raise GPUError("INVALID_BRANCH_PARENT", parent_node_id)
            if experiment_id:
                experiment = references[experiment_id]
                if experiment["kind"] != "Experiment" or not experiment["data"].get("frozen"):
                    raise GPUError("EXPERIMENT_NOT_PREREGISTERED", experiment_id)
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ExperimentNode','PLANNED',%s,%s)",
                (node_id, project_id, json.dumps(data), now),
            )
            cur.execute(
                "INSERT INTO research_edges VALUES(%s,%s,'CONTAINS_NODE',%s)",
                (branch_id, node_id, now),
            )
            if experiment_id:
                cur.execute(
                    "INSERT INTO research_edges VALUES(%s,%s,'PLANNED_BY_NODE',%s)",
                    (experiment_id, node_id, now),
                )
            relation = None
            if parent_node_id:
                relation = {
                    "branch_id": branch_id,
                    "source_node_id": parent_node_id,
                    "target_node_id": str(node_id),
                    "relation": "BRANCHES_TO",
                }
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'BranchRelation','ACTIVE',%s,%s)",
                    (relation_id, project_id, json.dumps(relation), now),
                )
                cur.execute(
                    "INSERT INTO research_edges VALUES(%s,%s,'BRANCHES_TO',%s)",
                    (parent_node_id, node_id, now),
                )
                self._event(cur, project_id, "BRANCH_RELATION_CREATED", relation_id, relation)
            self._event(cur, project_id, "EXPERIMENT_NODE_CREATED", node_id, data)
        return {
            "id": str(node_id),
            "project_id": project_id,
            "kind": "ExperimentNode",
            "status": "PLANNED",
            "data": data,
            "relation_id": str(relation_id) if relation else None,
        }

    def comparative_lesson_create(
        self,
        project_id: str,
        branch_id: str,
        node_a_id: str,
        node_b_id: str,
        data: dict[str, Any],
    ) -> dict:
        """Atomically persist a comparative lesson, relation, graph edges, and events."""
        project_id = self._canonical_uuid(project_id)
        branch_id = self._canonical_uuid(branch_id)
        node_a_id = self._canonical_uuid(node_a_id)
        node_b_id = self._canonical_uuid(node_b_id)
        data = {
            **data,
            "branch_id": branch_id,
            "node_a_id": node_a_id,
            "node_b_id": node_b_id,
        }
        lesson_id, relation_id, now = uuid.uuid4(), uuid.uuid4(), datetime.now(UTC)
        reference_ids = [branch_id, node_a_id, node_b_id]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"comparative-lesson:{branch_id}:{node_a_id}:{node_b_id}",),
            )
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects "
                "WHERE id=ANY(%s::uuid[]) FOR SHARE",
                (reference_ids,),
            )
            references = {str(item["id"]): item for item in cur.fetchall()}
            if len(references) != 3:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", "One or more comparison references")
            if any(str(item["project_id"]) != str(project_id) for item in references.values()):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", "Comparison references must share a project")
            if references[branch_id]["kind"] != "ExperimentBranch":
                raise GPUError("NOT_AN_EXPERIMENT_BRANCH", branch_id)
            for node_id in (node_a_id, node_b_id):
                node = references[node_id]
                if (
                    node["kind"] != "ExperimentNode"
                    or node["status"] != "RESULT_INSPECTED"
                    or str(node["data"].get("branch_id")) != branch_id
                ):
                    raise GPUError("INVALID_BRANCH_COMPARISON", node_id)
            pair_filter = {
                "branch_id": branch_id,
                "node_a_id": node_a_id,
                "node_b_id": node_b_id,
            }
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='ComparativeLesson' AND data @> %s::jsonb "
                "ORDER BY created_at LIMIT 1",
                (project_id, json.dumps(pair_filter)),
            )
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    "SELECT id FROM research_objects WHERE project_id=%s "
                    "AND kind='BranchRelation' AND data->>'comparative_lesson_id'=%s "
                    "ORDER BY created_at LIMIT 1",
                    (project_id, str(existing["id"])),
                )
                existing_relation = cur.fetchone()
                return {
                    **existing,
                    "relation_id": (
                        str(existing_relation["id"]) if existing_relation else None
                    ),
                    "idempotent_replay": True,
                }
            relation = {
                "branch_id": branch_id,
                "source_node_id": node_a_id,
                "target_node_id": node_b_id,
                "relation": "COMPARED_WITH",
                "comparative_lesson_id": str(lesson_id),
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ComparativeLesson','RESULT_INSPECTED',%s,%s)",
                (lesson_id, project_id, json.dumps(data), now),
            )
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'BranchRelation','ACTIVE',%s,%s)",
                (relation_id, project_id, json.dumps(relation), now),
            )
            for node_id in (node_a_id, node_b_id):
                cur.execute(
                    "INSERT INTO research_edges VALUES(%s,%s,'INFORMS_COMPARISON',%s)",
                    (node_id, lesson_id, now),
                )
            self._event(cur, project_id, "COMPARATIVE_LESSON_CREATED", lesson_id, data)
            self._event(cur, project_id, "BRANCH_RELATION_CREATED", relation_id, relation)
        return {
            "id": str(lesson_id),
            "project_id": project_id,
            "kind": "ComparativeLesson",
            "status": "RESULT_INSPECTED",
            "data": data,
            "relation_id": str(relation_id),
        }

    def meta_lesson_create(self, project_id: str, data: dict[str, Any]) -> dict:
        """Idempotently persist one process lesson for an exact scientific-state basis."""
        if not isinstance(data.get("basis"), dict):
            raise GPUError("INVALID_META_LESSON", "A structured basis is required")
        fingerprint = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload = {**data, "basis_fingerprint": fingerprint}
        ident, now = uuid.uuid4(), datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"meta-lesson:{project_id}:{fingerprint}",),
            )
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='MetaLesson' "
                "AND data->>'basis_fingerprint'=%s ORDER BY created_at LIMIT 1",
                (project_id, fingerprint),
            )
            existing = cur.fetchone()
            if existing:
                return {**existing, "idempotent_replay": True}
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'MetaLesson','RESULT_INSPECTED',%s,%s)",
                (ident, project_id, json.dumps(payload), now),
            )
            self._event(cur, project_id, "META_REVIEW_COMPLETED", ident, payload)
        return {
            "id": str(ident),
            "project_id": project_id,
            "kind": "MetaLesson",
            "status": "RESULT_INSPECTED",
            "data": payload,
        }

    def world_model_child_create(
        self,
        world_model_id: str,
        child_kind: str,
        child_data: dict[str, Any],
        child_event: str,
        list_field: str,
        changes: dict[str, Any],
        evidence_ids: list[str] | None = None,
        decision_id: str | None = None,
        child_status: str = "IMPLEMENTED_UNVERIFIED",
    ) -> dict:
        """Atomically create a model child, append it, and advance the model version."""
        if child_kind not in RESEARCH_OBJECT_KINDS or list_field not in {"node_ids", "edge_ids"}:
            raise GPUError("INVALID_WORLD_MODEL_CHILD", child_kind)
        self._validate_status(child_status)
        now, child_id, version_id = datetime.now(UTC), uuid.uuid4(), uuid.uuid4()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT project_id,status,data FROM research_objects "
                "WHERE id=%s AND kind='WorldModel' FOR UPDATE",
                (world_model_id,),
            )
            model = cur.fetchone()
            if not model:
                raise GPUError("NOT_A_WORLDMODEL", world_model_id)
            if child_kind == "CausalEdge":
                edge_status = str(child_data.get("edge_status"))
                if edge_status == "INTERVENTION_SUPPORTED" or any(
                    child_data.get(field)
                    for field in (
                        "supporting_evidence_family_ids",
                        "contradicting_evidence_family_ids",
                    )
                ):
                    raise GPUError(
                        "CAUSAL_PROMOTION_REQUIRES_RESULT_ASSESSMENT",
                        "Causal intervention support must come from an inspected result",
                    )
                if edge_status in {"OBSERVED_ASSOCIATION", "HYPOTHESIZED_CAUSAL"}:
                    cur.execute(
                        "SELECT id FROM research_objects WHERE project_id=%s "
                        "AND kind='CausalEdge' "
                        "AND data->>'source_id'=%s AND data->>'target_id'=%s "
                        "AND data->>'relation'=%s "
                        "AND coalesce(data->'scope','{}'::jsonb)=%s::jsonb "
                        "AND data->>'edge_status'='REFUTED' LIMIT 1",
                        (
                            model["project_id"],
                            child_data.get("source_id"),
                            child_data.get("target_id"),
                            child_data.get("relation"),
                            json.dumps(child_data.get("scope", {})),
                        ),
                    )
                    if cur.fetchone():
                        raise GPUError(
                            "WORLD_MODEL_CONSISTENCY_ERROR",
                            "An identical-scope refuted edge blocks positive edge creation",
                        )
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s)",
                (
                    child_id,
                    model["project_id"],
                    child_kind,
                    child_status,
                    json.dumps(child_data),
                    now,
                ),
            )
            self._event(cur, model["project_id"], child_event, child_id, child_data)
            current_version = int(model["data"].get("version", 0))
            change_key = "nodes_added" if list_field == "node_ids" else "edges_added"
            version_data = {
                "world_model_id": world_model_id,
                "version": current_version + 1,
                "parent_version_id": model["data"].get("current_version_id"),
                "changes": {
                    **changes,
                    change_key: [*changes.get(change_key, []), str(child_id)],
                },
                "evidence_ids": evidence_ids or [],
                "decision_id": decision_id,
                "timestamp": now.isoformat(),
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'WorldModelVersion','IMPLEMENTED_UNVERIFIED',%s,%s)",
                (version_id, model["project_id"], json.dumps(version_data), now),
            )
            self._event(
                cur,
                model["project_id"],
                "WORLD_MODEL_VERSION_CREATED",
                version_id,
                version_data,
            )
            model_data = {
                **model["data"],
                list_field: [*model["data"].get(list_field, []), str(child_id)],
                "current_version_id": str(version_id),
                "version": current_version + 1,
            }
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(model_data), world_model_id),
            )
            self._event(
                cur,
                model["project_id"],
                "WORLD_MODEL_CHILD_ADDED",
                world_model_id,
                {"field": list_field, "child_id": str(child_id)},
            )
        return {
            "child": {
                "id": str(child_id),
                "project_id": str(model["project_id"]),
                "kind": child_kind,
                "status": child_status,
                "data": child_data,
            },
            "version": {
                "id": str(version_id),
                "project_id": str(model["project_id"]),
                "kind": "WorldModelVersion",
                "status": "IMPLEMENTED_UNVERIFIED",
                "data": version_data,
            },
        }

    def world_model_create_atomic(self, project_id: str, name: str, scope: str) -> dict:
        """Create a WorldModel and its required initial version in one transaction."""
        now, model_id, version_id = datetime.now(UTC), uuid.uuid4(), uuid.uuid4()
        version_data = {
            "world_model_id": str(model_id),
            "version": 1,
            "parent_version_id": None,
            "changes": {"world_model_created": True},
            "evidence_ids": [],
            "decision_id": None,
            "timestamp": now.isoformat(),
        }
        model_data = {
            "name": name,
            "scope": scope,
            "node_ids": [],
            "edge_ids": [],
            "current_version_id": str(version_id),
            "version": 1,
        }
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'WorldModel','IMPLEMENTED_UNVERIFIED',%s,%s)",
                (model_id, project_id, json.dumps(model_data), now),
            )
            self._event(cur, project_id, "WORLD_MODEL_CREATED", model_id, model_data)
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'WorldModelVersion','IMPLEMENTED_UNVERIFIED',%s,%s)",
                (version_id, project_id, json.dumps(version_data), now),
            )
            self._event(
                cur, project_id, "WORLD_MODEL_VERSION_CREATED", version_id, version_data
            )
            self._event(
                cur,
                project_id,
                "WORLD_MODEL_VERSION_ADVANCED",
                model_id,
                {"version": 1},
            )
        return {
            "world_model": {
                "id": str(model_id),
                "project_id": project_id,
                "kind": "WorldModel",
                "status": "IMPLEMENTED_UNVERIFIED",
                "data": model_data,
            },
            "version": {
                "id": str(version_id),
                "project_id": project_id,
                "kind": "WorldModelVersion",
                "status": "IMPLEMENTED_UNVERIFIED",
                "data": version_data,
            },
        }

    def agenda_create_once(self, project_id: str, name: str) -> dict:
        """Serialize creation so a project has at most one active research agenda."""
        project_id = self._canonical_uuid(project_id)

        def response(row: dict, *, idempotent_replay: bool) -> dict:
            return {
                "id": str(row["id"]),
                "project_id": str(row["project_id"]),
                "kind": row["kind"],
                "status": row["status"],
                "data": row["data"],
                "created_at": row["created_at"],
                "idempotent_replay": idempotent_replay,
            }

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"agenda:{project_id}",))
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='ResearchAgenda' AND status='ACTIVE' "
                "ORDER BY created_at DESC LIMIT 1",
                (project_id,),
            )
            existing = cur.fetchone()
            if existing:
                return response(existing, idempotent_replay=True)
            ident, now = uuid.uuid4(), datetime.now(UTC)
            data = {"name": name, "item_ids": []}
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ResearchAgenda','ACTIVE',%s,%s)",
                (ident, project_id, json.dumps(data), now),
            )
            self._event(cur, project_id, "RESEARCH_AGENDA_CREATED", ident, data)
            return response(
                {
                    "id": ident,
                    "project_id": project_id,
                    "kind": "ResearchAgenda",
                    "status": "ACTIVE",
                    "data": data,
                    "created_at": now,
                },
                idempotent_replay=False,
            )

    def agenda_item_create_atomic(self, agenda_id: str, data: dict[str, Any]) -> dict:
        """Create an agenda item and append its ID without losing concurrent inserts."""
        ident, now = uuid.uuid4(), datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT project_id,status,data FROM research_objects "
                "WHERE id=%s AND kind='ResearchAgenda' FOR UPDATE",
                (agenda_id,),
            )
            agenda = cur.fetchone()
            if not agenda:
                raise GPUError("NOT_A_RESEARCHAGENDA", agenda_id)
            if agenda["status"] != "ACTIVE":
                raise GPUError("RESEARCH_AGENDA_NOT_ACTIVE", agenda["status"])
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'AgendaItem','OPEN',%s,%s)",
                (ident, agenda["project_id"], json.dumps(data), now),
            )
            self._event(cur, agenda["project_id"], "AGENDA_ITEM_CREATED", ident, data)
            agenda_data = {
                **agenda["data"],
                "item_ids": [*agenda["data"].get("item_ids", []), str(ident)],
            }
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(agenda_data), agenda_id),
            )
            self._event(
                cur,
                agenda["project_id"],
                "RESEARCH_AGENDA_ITEM_ADDED",
                agenda_id,
                {"item_id": str(ident)},
            )
        return {
            "id": str(ident),
            "project_id": str(agenda["project_id"]),
            "kind": "AgendaItem",
            "status": "OPEN",
            "data": data,
        }

    def world_model_update_and_version(
        self,
        world_model_id: str,
        model_update: dict[str, Any],
        changes: dict[str, Any],
        evidence_ids: list[str],
        decision_id: str | None,
        event_type: str,
    ) -> dict:
        """Lock a world model while applying a change and allocating its next version."""
        now, version_id = datetime.now(UTC), uuid.uuid4()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT project_id,status,data FROM research_objects "
                "WHERE id=%s AND kind='WorldModel' FOR UPDATE",
                (world_model_id,),
            )
            model = cur.fetchone()
            if not model:
                raise GPUError("NOT_A_WORLDMODEL", world_model_id)
            current_version = int(model["data"].get("version", 0))
            version_data = {
                "world_model_id": world_model_id,
                "version": current_version + 1,
                "parent_version_id": model["data"].get("current_version_id"),
                "changes": changes,
                "evidence_ids": evidence_ids,
                "decision_id": decision_id,
                "timestamp": now.isoformat(),
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'WorldModelVersion','IMPLEMENTED_UNVERIFIED',%s,%s)",
                (version_id, model["project_id"], json.dumps(version_data), now),
            )
            self._event(
                cur,
                model["project_id"],
                "WORLD_MODEL_VERSION_CREATED",
                version_id,
                version_data,
            )
            data = {
                **model["data"],
                **model_update,
                "current_version_id": str(version_id),
                "version": current_version + 1,
            }
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(data), world_model_id),
            )
            self._event(
                cur,
                model["project_id"],
                event_type,
                world_model_id,
                {"status": model["status"], **model_update},
            )
        return {
            "model": {"id": world_model_id, "status": model["status"], "data": data},
            "version": {
                "id": str(version_id),
                "project_id": str(model["project_id"]),
                "kind": "WorldModelVersion",
                "status": "IMPLEMENTED_UNVERIFIED",
                "data": version_data,
            },
        }

    def causal_edge_update_atomic(
        self,
        edge_id: str,
        data_update: dict[str, Any],
        object_status: str,
        changes: dict[str, Any],
        evidence_ids: list[str],
        decision_id: str | None,
        expected_edge_status: str | None = None,
    ) -> dict:
        """Update a causal edge and advance its world model in one transaction."""
        self._validate_status(object_status)
        if "world_model_id" in data_update:
            raise GPUError("CAUSAL_EDGE_WORLD_MODEL_IMMUTABLE", edge_id)
        if any(
            data_update.get(field)
            for field in (
                "supporting_evidence_family_ids",
                "contradicting_evidence_family_ids",
                "evidence_family_ids",
            )
        ):
            raise GPUError(
                "CAUSAL_EVIDENCE_FAMILY_REQUIRES_RESULT_ASSESSMENT",
                "Direct edge updates cannot attach empirical EvidenceFamilies",
            )
        if data_update.get("edge_status") == "INTERVENTION_SUPPORTED":
            raise GPUError(
                "CAUSAL_PROMOTION_REQUIRES_RESULT_ASSESSMENT",
                "Intervention support must be derived by result_assessment_apply",
            )
        now, version_id = datetime.now(UTC), uuid.uuid4()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT project_id,status,data FROM research_objects "
                "WHERE id=%s AND kind='CausalEdge' FOR UPDATE",
                (edge_id,),
            )
            edge = cur.fetchone()
            if not edge:
                raise GPUError("NOT_A_CAUSALEDGE", edge_id)
            current_edge_status = str(edge["data"].get("edge_status"))
            if expected_edge_status is not None and current_edge_status != expected_edge_status:
                raise GPUError(
                    "CAUSAL_EDGE_CONCURRENT_UPDATE",
                    f"Expected {expected_edge_status}, found {current_edge_status}; reload and retry",
                    retryable=True,
                )
            model_id = edge["data"].get("world_model_id")
            cur.execute(
                "SELECT project_id,status,data FROM research_objects "
                "WHERE id=%s AND kind='WorldModel' FOR UPDATE",
                (model_id,),
            )
            model = cur.fetchone()
            if not model:
                raise GPUError("NOT_A_WORLDMODEL", str(model_id))
            if str(model["project_id"]) != str(edge["project_id"]):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", edge_id)
            edge_data = {**edge["data"], **data_update}
            for field in ("supporting_ids", "against_ids"):
                edge_data[field] = list(
                    dict.fromkeys(
                        [*edge["data"].get(field, []), *data_update.get(field, [])]
                    )
                )
            new_edge_status = str(edge_data.get("edge_status"))
            combined_evidence_ids = list(
                dict.fromkeys(
                    [*edge_data.get("supporting_ids", []), *edge_data.get("against_ids", [])]
                )
            )
            if new_edge_status in {"OBSERVED_ASSOCIATION", "WEAKENED", "REFUTED"} and not combined_evidence_ids:
                raise GPUError("CAUSAL_EDGE_EVIDENCE_REQUIRED", new_edge_status)
            scope = edge_data.get("scope", {})
            if new_edge_status in {"OBSERVED_ASSOCIATION", "HYPOTHESIZED_CAUSAL"}:
                cur.execute(
                    "SELECT id FROM research_objects WHERE project_id=%s "
                    "AND kind='CausalEdge' AND id<>%s "
                    "AND data->>'source_id'=%s AND data->>'target_id'=%s "
                    "AND data->>'relation'=%s AND coalesce(data->'scope','{}'::jsonb)=%s::jsonb "
                    "AND data->>'edge_status'='REFUTED' LIMIT 1",
                    (
                        edge["project_id"],
                        edge_id,
                        edge_data.get("source_id"),
                        edge_data.get("target_id"),
                        edge_data.get("relation"),
                        json.dumps(scope),
                    ),
                )
                if cur.fetchone():
                    raise GPUError(
                        "WORLD_MODEL_CONSISTENCY_ERROR",
                        "An identical-scope refuted edge blocks a positive causal transition",
                    )
            if new_edge_status == "REFUTED":
                cur.execute(
                    "SELECT id FROM research_objects WHERE project_id=%s "
                    "AND kind='Hypothesis' AND status=ANY(%s) "
                    "AND ((coalesce(data->'required_edge_ids','[]'::jsonb) ? %s) "
                    "OR (coalesce(data->'causal_edge_ids','[]'::jsonb) ? %s)) LIMIT 1",
                    (
                        edge["project_id"],
                        ["ACTIVE", "SURVIVES_INITIAL_TEST", "SUPPORTED"],
                        edge_id,
                        edge_id,
                    ),
                )
                if cur.fetchone():
                    raise GPUError(
                        "WORLD_MODEL_CONSISTENCY_ERROR",
                        "An active hypothesis still requires the edge being refuted",
                    )
            cur.execute(
                "UPDATE research_objects SET status=%s,data=%s WHERE id=%s",
                (object_status, json.dumps(edge_data), edge_id),
            )
            self._event(
                cur,
                edge["project_id"],
                "CAUSAL_EDGE_STATUS_CHANGED",
                edge_id,
                {"status": object_status, **data_update, "evidence_ids": combined_evidence_ids},
            )
            version_number = int(model["data"].get("version", 0)) + 1
            version_data = {
                "world_model_id": str(model_id),
                "version": version_number,
                "parent_version_id": model["data"].get("current_version_id"),
                "changes": {
                    **changes,
                    "edges_status_changed": [
                        {
                            "edge_id": edge_id,
                            "from": current_edge_status,
                            "to": new_edge_status,
                        }
                    ],
                },
                "evidence_ids": combined_evidence_ids,
                "decision_id": decision_id,
                "timestamp": now.isoformat(),
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'WorldModelVersion','IMPLEMENTED_UNVERIFIED',%s,%s)",
                (version_id, edge["project_id"], json.dumps(version_data), now),
            )
            self._event(
                cur,
                edge["project_id"],
                "WORLD_MODEL_VERSION_CREATED",
                version_id,
                version_data,
            )
            model_data = {
                **model["data"],
                "current_version_id": str(version_id),
                "version": version_number,
            }
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(model_data), model_id),
            )
            self._event(
                cur,
                edge["project_id"],
                "WORLD_MODEL_VERSION_ADVANCED",
                model_id,
                {"version": version_number},
            )
        return {
            "edge": {"id": edge_id, "status": object_status, "data": edge_data},
            "version": {
                "id": str(version_id),
                "project_id": str(edge["project_id"]),
                "kind": "WorldModelVersion",
                "status": "IMPLEMENTED_UNVERIFIED",
                "data": version_data,
            },
        }

    def brain_decision_create(
        self,
        project_id: str,
        candidates: list[dict[str, Any]],
        selected_index: int,
        decision_data: dict[str, Any],
        situation_data: dict[str, Any] | None = None,
        portfolio_data: dict[str, Any] | None = None,
    ) -> dict:
        """Persist one Brain step's situation, candidates, and decision atomically."""
        if not candidates or not 0 <= selected_index < len(candidates):
            raise GPUError("INVALID_BRAIN_DECISION", "A selected candidate is required")
        now = datetime.now(UTC)
        candidate_ids = [uuid.uuid4() for _ in candidates]
        decision_id = uuid.uuid4()
        situation_id = uuid.uuid4() if situation_data is not None else None
        portfolio_id = uuid.uuid4() if portfolio_data is not None else None
        persisted = _json_document(
            [
                {**candidate, "id": str(candidate_id)}
                for candidate, candidate_id in zip(candidates, candidate_ids, strict=True)
            ]
        )
        materialized = _json_document(
            {
                **decision_data,
                **classify_decision_data(decision_data),
                "cycle_status": classify_decision_data(decision_data)["cycle_status"],
                "research_situation_id": str(situation_id) if situation_id else None,
                "candidate_portfolio_id": str(portfolio_id) if portfolio_id else None,
                "candidate_action_ids": [str(item) for item in candidate_ids],
                "candidate_actions": persisted,
                "selected_action": persisted[selected_index],
            }
        )
        with self._connect() as conn, conn.cursor() as cur:
            if situation_id is not None:
                persisted_situation = _json_document(
                    {**situation_data, "created_from_decision_id": str(decision_id)}
                )
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'ResearchSituation','ACTIVE',%s,%s)",
                    (situation_id, project_id, json.dumps(persisted_situation), now),
                )
                self._event(
                    cur,
                    project_id,
                    "RESEARCH_SITUATION_CREATED",
                    situation_id,
                    persisted_situation,
                )
            if portfolio_id is not None and portfolio_data is not None:
                persisted_portfolio = _json_document(
                    {
                        **portfolio_data,
                        "created_from_decision_id": str(decision_id),
                        "generated_candidate_ids": [str(item) for item in candidate_ids],
                    }
                )
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'CandidatePortfolio','ACTIVE',%s,%s)",
                    (portfolio_id, project_id, json.dumps(persisted_portfolio), now),
                )
                self._event(
                    cur,
                    project_id,
                    "CANDIDATE_PORTFOLIO_CREATED",
                    portfolio_id,
                    persisted_portfolio,
                )
            for candidate, candidate_id in zip(candidates, candidate_ids, strict=True):
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'ResearchActionCandidate','PROPOSED',%s,%s)",
                    (candidate_id, project_id, json.dumps(candidate), now),
                )
                self._event(
                    cur,
                    project_id,
                    "RESEARCH_ACTION_CANDIDATE_CREATED",
                    candidate_id,
                    candidate,
                )
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ResearchDecision','SELECTED',%s,%s)",
                (decision_id, project_id, json.dumps(materialized), now),
            )
            self._event(
                cur,
                project_id,
                "RESEARCH_DECISION_SELECTED",
                decision_id,
                materialized,
            )
            regime = materialized.get("search_regime")
            if regime:
                self._event(
                    cur,
                    project_id,
                    "SEARCH_REGIME_CHANGED",
                    decision_id,
                    {
                        "search_regime": regime,
                        "reason": materialized.get("search_regime_reason", []),
                        "candidate_portfolio_id": str(portfolio_id) if portfolio_id else None,
                    },
                )
            if regime == "PARADIGM_RESET":
                self._event(
                    cur,
                    project_id,
                    "PARADIGM_RESET_TRIGGERED",
                    decision_id,
                    {
                        "candidate_portfolio_id": str(portfolio_id) if portfolio_id else None,
                        "frontier_gap": materialized.get("frontier_gap", {}),
                        "stagnation_state": materialized.get("stagnation_state", {}),
                    },
                )
        return {
            "decision": {
                "id": str(decision_id),
                "project_id": project_id,
                "kind": "ResearchDecision",
                "status": "SELECTED",
                "data": materialized,
            },
            "candidates": persisted,
            "selected": persisted[selected_index],
            "situation": (
                {
                    "id": str(situation_id),
                    "project_id": project_id,
                    "kind": "ResearchSituation",
                    "status": "ACTIVE",
                    "data": _json_document(
                        {**situation_data, "created_from_decision_id": str(decision_id)}
                    ),
                }
                if situation_id is not None and situation_data is not None
                else None
            ),
            "portfolio": (
                {
                    "id": str(portfolio_id),
                    "project_id": project_id,
                    "kind": "CandidatePortfolio",
                    "status": "ACTIVE",
                    "data": _json_document(
                        {
                            **portfolio_data,
                            "created_from_decision_id": str(decision_id),
                            "generated_candidate_ids": [str(item) for item in candidate_ids],
                        }
                    ),
                }
                if portfolio_id is not None and portfolio_data is not None
                else None
            ),
        }

    def breakthrough_signal_record(
        self, project_id: str, signal: dict[str, Any], decision_id: str | None = None
    ) -> dict[str, Any]:
        """Persist strategic discovery value without mutating hypothesis status."""
        data = _json_document(
            {
                **signal,
                "decision_id": decision_id,
                "scientific_truth_independent": True,
                "recording_policy": "brain-v3.1-discovery-search-v1",
            }
        )
        if signal.get("hypothesis_status") == "SUPPORTED" and signal.get("type") == "PARTIAL_METRIC_BREAKTHROUGH":
            raise GPUError("BREAKTHROUGH_CANNOT_RECLASSIFY_HYPOTHESIS", "Record a partial breakthrough without changing scientific truth")
        return self.object_create(project_id, "BreakthroughSignal", data, "BREAKTHROUGH_SIGNAL_RECORDED")

    def epistemic_reclassification(self, project_id: str | None = None) -> dict[str, Any]:
        """Classify legacy decision records without changing scientific observations.

        The operation is idempotent: an already equivalent v2.1 classification
        produces no new event or object version.
        """
        with self._connect() as conn, conn.cursor() as cur:
            sql = (
                "SELECT id,project_id,status,data FROM research_objects "
                "WHERE kind='ResearchDecision'"
            )
            args: list[Any] = []
            if project_id:
                sql += " AND project_id=%s"
                args.append(project_id)
            sql += " ORDER BY created_at FOR UPDATE"
            cur.execute(sql, args)
            decisions = cur.fetchall()
            cur.execute(
                "SELECT project_id,status,data FROM research_objects "
                "WHERE kind='ResearchDecisionOutcome'"
                + (" AND project_id=%s" if project_id else ""),
                args,
            )
            outcomes = cur.fetchall()
            outcome_by_decision = {
                str(item["data"].get("decision_id")): item for item in outcomes
            }
            counts: dict[str, int] = {role: 0 for role in DECISION_ROLES}
            changed = 0
            eligible = 0
            closed = 0
            for decision in decisions:
                classification = classify_decision_data(decision["data"])
                outcome = outcome_by_decision.get(str(decision["id"]))
                if outcome and str(outcome["status"]) in {"RESULT_INSPECTED", "ASSESSED", "CLOSED"}:
                    classification["cycle_status"] = "CLOSED"
                    if classification["scientific_role"] not in {"NOT_SCIENTIFIC", "SYSTEM_SMOKE", "CONTRACT_TEST"}:
                        classification["scientific_verification"] = "RESULT_INSPECTED"
                eligibility = strategy_learning_eligibility(
                    {"data": {**decision["data"], **classification}}, outcome
                )
                classification["strategy_learning_role"] = (
                    "SCIENTIFIC_ACTION_OUTCOME"
                    if eligibility["eligible"]
                    else "EXCLUDE"
                )
                classification["epistemic_classification_version"] = "v2.1"
                counts[classification["decision_role"]] += 1
                eligible += int(eligibility["eligible"])
                closed += int(classification["cycle_status"] == "CLOSED")
                needs_update = any(
                    decision["data"].get(key) != value for key, value in classification.items()
                )
                if not needs_update:
                    continue
                data = _json_document({**decision["data"], **classification})
                cur.execute(
                    "UPDATE research_objects SET data=%s WHERE id=%s",
                    (json.dumps(data), decision["id"]),
                )
                self._event(
                    cur,
                    decision["project_id"],
                    "EPISTEMIC_CLASSIFICATION_BACKFILLED",
                    decision["id"],
                    {"classification": classification, "strategy_learning_eligible": eligibility["eligible"]},
                )
                changed += 1
            return {
                "project_id": project_id,
                "total_decisions": len(decisions),
                "changed": changed,
                "decision_roles": counts,
                "strategy_learning_eligible": eligible,
                "closed_cycles": closed,
                "open_or_incomplete_cycles": len(decisions) - closed,
                "idempotent": changed == 0,
            }

    def decision_outcome_apply(
        self,
        decision_id: str,
        outcome_data: dict[str, Any],
        after_situation_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist S_t+1, a reassessable outcome, and strategy updates atomically."""
        now = datetime.now(UTC)
        positive_labels = {"HIGH_VALUE", "USEFUL"}
        negative_labels = {
            "LOW_VALUE",
            "ZERO_INFORMATION",
            "REDUNDANT",
            "PREMATURE",
            "INVALID",
        }

        def strategy_key(scope_level: str, discriminator: str) -> str:
            material = "|".join(
                (
                    "brain-v2-strategy-v1",
                    scope_level,
                    discriminator,
                    str(outcome_data["problem_signature"]),
                    str(outcome_data["action_type"]),
                )
            )
            return hashlib.sha256(material.encode()).hexdigest()

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE id=%s FOR UPDATE",
                (decision_id,),
            )
            decision = cur.fetchone()
            if not decision:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", decision_id)
            if decision["kind"] != "ResearchDecision":
                raise GPUError("NOT_A_RESEARCHDECISION", decision_id)
            project_id = str(decision["project_id"])
            if str(outcome_data.get("project_id")) != project_id:
                raise GPUError("RESEARCH_PROJECT_MISMATCH", decision_id)

            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE kind='ResearchDecisionOutcome' AND data->>'decision_id'=%s FOR UPDATE",
                (decision_id,),
            )
            existing_outcome = cur.fetchone()
            assessment_history = []
            if existing_outcome:
                assessment_history = list(existing_outcome["data"].get("assessment_history", []))
                previous = {
                    key: existing_outcome["data"].get(key)
                    for key in (
                        "label",
                        "realized_information_gain",
                        "hindsight_assessment",
                        "assessed_at",
                    )
                }
                if previous not in assessment_history:
                    assessment_history.append(previous)
                after_situation_id = existing_outcome["data"].get("after_situation_id")
                if not after_situation_id:
                    after_situation_id = str(uuid.uuid4())
                    after_payload = _json_document(
                        {
                            **after_situation_data,
                            "created_from_decision_id": decision_id,
                            "transition_role": "S_t+1",
                        }
                    )
                    cur.execute(
                        "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                        "VALUES(%s,%s,'ResearchSituation','ACTIVE',%s,%s)",
                        (after_situation_id, project_id, json.dumps(after_payload), now),
                    )
                    self._event(
                        cur,
                        project_id,
                        "RESEARCH_SITUATION_CREATED",
                        after_situation_id,
                        after_payload,
                    )
                outcome_id = existing_outcome["id"]
                materialized_outcome = _json_document(
                    {
                        **existing_outcome["data"],
                        **outcome_data,
                        "after_situation_id": str(after_situation_id),
                        "assessment_history": assessment_history,
                        "assessed_at": now.isoformat(),
                    }
                )
                cur.execute(
                    "UPDATE research_objects SET status='RESULT_INSPECTED',data=%s WHERE id=%s",
                    (json.dumps(materialized_outcome), outcome_id),
                )
            else:
                after_situation_id, outcome_id = uuid.uuid4(), uuid.uuid4()
                after_payload = _json_document(
                    {
                        **after_situation_data,
                        "created_from_decision_id": decision_id,
                        "transition_role": "S_t+1",
                    }
                )
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'ResearchSituation','ACTIVE',%s,%s)",
                    (after_situation_id, project_id, json.dumps(after_payload), now),
                )
                self._event(
                    cur,
                    project_id,
                    "RESEARCH_SITUATION_CREATED",
                    after_situation_id,
                    after_payload,
                )
                materialized_outcome = _json_document(
                    {
                        **outcome_data,
                        "after_situation_id": str(after_situation_id),
                        "assessment_history": [],
                        "assessed_at": now.isoformat(),
                    }
                )
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'ResearchDecisionOutcome','RESULT_INSPECTED',%s,%s)",
                    (outcome_id, project_id, json.dumps(materialized_outcome), now),
                )

            # A failure-analysis decision is a terminal inspection of its linked
            # failed run, even though it deliberately produces no scientific
            # evidence. Preserve the execution status and add the normal
            # inspection marker so INSPECT_FAILURE is not selected indefinitely.
            technically_inspected_runs: list[dict[str, Any]] = []
            try:
                run_ids = sorted(
                    {
                        str(uuid.UUID(str(run_id)))
                        for run_id in outcome_data.get("experiment_run_ids", [])
                    }
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise GPUError("INVALID_RESEARCH_OBJECT_ID", str(exc)) from exc
            if run_ids:
                cur.execute(
                    "SELECT id,project_id,kind,status,data FROM research_objects "
                    "WHERE id=ANY(%s::uuid[]) FOR UPDATE",
                    (run_ids,),
                )
                runs = {str(item["id"]): item for item in cur.fetchall()}
                for run_id in run_ids:
                    run = runs.get(run_id)
                    if not run:
                        raise GPUError("RESEARCH_OBJECT_NOT_FOUND", run_id)
                    if run["kind"] != "ExperimentRun":
                        raise GPUError("NOT_AN_EXPERIMENTRUN", run_id)
                    if str(run["project_id"]) != project_id:
                        raise GPUError("RESEARCH_PROJECT_MISMATCH", run_id)
                    if run["status"] not in {"failed", "cancelled", "unknown"}:
                        continue
                    if run["data"].get("inspection") is not None:
                        continue
                    inspection_data = {
                        "inspection_kind": "TECHNICAL_FAILURE",
                        "decision_id": decision_id,
                        "decision_outcome_id": str(outcome_id),
                        "assessed_at": now.isoformat(),
                        "execution_status": run["status"],
                        "scientific_result": "NOT_ASSESSED",
                    }
                    run_data = _json_document({**run["data"], "inspection": inspection_data})
                    cur.execute(
                        "UPDATE research_objects SET data=%s WHERE id=%s",
                        (json.dumps(run_data), run_id),
                    )
                    self._event(
                        cur,
                        project_id,
                        "EXPERIMENT_FAILURE_INSPECTED",
                        run_id,
                        inspection_data,
                    )
                    technically_inspected_runs.append(
                        {"id": run_id, "status": run["status"], "data": run_data}
                    )
            self._event(
                cur,
                project_id,
                "DECISION_OUTCOME_ASSESSED",
                outcome_id,
                materialized_outcome,
            )

            decision_data = _json_document(
                {
                    **decision["data"],
                    **classify_decision_data(decision["data"]),
                    "cycle_status": "CLOSED",
                    "decision_outcome_id": str(outcome_id),
                    "actual_information_gain": outcome_data.get("realized_information_gain"),
                    "hindsight_assessment": outcome_data.get("hindsight_assessment"),
                    "outcome": outcome_data.get("observed_result"),
                }
            )
            cur.execute(
                "UPDATE research_objects SET status='COMPLETED',data=%s WHERE id=%s",
                (json.dumps(decision_data), decision_id),
            )

            cur.execute(
                "SELECT id,project_id,status,data FROM research_objects "
                "WHERE kind='ResearchDecisionOutcome' "
                "AND data->>'problem_signature'=%s AND data->>'action_type'=%s FOR UPDATE",
                (outcome_data["problem_signature"], outcome_data["action_type"]),
            )
            related_outcomes = cur.fetchall()

            # Only closed, prospective scientific cycles may contribute to
            # production strategy memory. Operational recovery, smokes,
            # contract tests, benchmarks, and incomplete outcomes remain
            # durable evidence but have zero strategy weight.
            decision_for_learning = {**decision, "data": decision_data}
            current_outcome = {
                "id": outcome_id,
                "status": "RESULT_INSPECTED",
                "data": materialized_outcome,
            }
            eligible_current = strategy_learning_eligibility(
                decision_for_learning, current_outcome
            )
            related_decision_ids = [
                str(item["data"].get("decision_id"))
                for item in related_outcomes
                if item["data"].get("decision_id")
            ]
            cur.execute(
                "SELECT id,data FROM research_objects WHERE kind='ResearchDecision' "
                "AND id=ANY(%s::uuid[])",
                (related_decision_ids,),
            )
            decisions_by_id = {str(item["id"]): item for item in cur.fetchall()}
            related_outcomes = [
                item
                for item in related_outcomes
                if strategy_learning_eligibility(
                    decisions_by_id.get(str(item["data"].get("decision_id")), {}), item
                )["eligible"]
            ]

            def cohort(scope_level: str) -> list[dict[str, Any]]:
                if scope_level == "PROJECT":
                    return [
                        item
                        for item in related_outcomes
                        if str(item["project_id"]) == project_id
                    ]
                if scope_level == "DOMAIN":
                    return [
                        item
                        for item in related_outcomes
                        if item["data"].get("domain") == outcome_data.get("domain")
                    ]
                return related_outcomes

            domain_rows = cohort("DOMAIN")
            global_rows = cohort("GLOBAL")
            eligible_for_strategy = (
                eligible_current["eligible"]
                and outcome_data.get("label") in (positive_labels | negative_labels)
            )
            scope_levels = ["PROJECT"] if eligible_for_strategy else []
            if eligible_for_strategy and len({str(item["project_id"]) for item in domain_rows}) >= 2:
                scope_levels.append("DOMAIN")
            if (
                eligible_for_strategy
                and
                len({str(item["project_id"]) for item in global_rows}) >= 3
                and len({str(item["data"].get("domain")) for item in global_rows}) >= 2
            ):
                scope_levels.append("GLOBAL")

            patterns: list[dict[str, Any]] = []
            for scope_level in scope_levels:
                rows = cohort(scope_level)
                labels = [str(item["data"].get("label", "UNKNOWN")) for item in rows]
                successes = sum(label in positive_labels for label in labels)
                failures = sum(label in negative_labels for label in labels)
                projects = sorted({str(item["project_id"]) for item in rows})
                domains = sorted(
                    {
                        str(item["data"].get("domain"))
                        for item in rows
                        if item["data"].get("domain")
                    }
                )
                discriminator = (
                    project_id
                    if scope_level == "PROJECT"
                    else str(outcome_data.get("domain"))
                    if scope_level == "DOMAIN"
                    else "GLOBAL"
                )
                key = strategy_key(scope_level, discriminator)
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
                cur.execute(
                    "SELECT id,project_id,status,data,created_at FROM research_objects "
                    "WHERE kind='ResearchStrategyPattern' AND data->>'strategy_key'=%s "
                    "FOR UPDATE",
                    (key,),
                )
                existing_pattern = cur.fetchone()
                if failures and successes:
                    support_level = "CONTEXT_DEPENDENT"
                elif failures > successes:
                    support_level = "REFUTED" if failures >= 2 else "WEAKENED"
                elif scope_level == "PROJECT":
                    support_level = (
                        "PROJECT_REPEATED" if successes >= 2 else "PROJECT_OBSERVATION"
                    )
                elif scope_level == "DOMAIN":
                    support_level = "DOMAIN_REPEATED"
                else:
                    support_level = (
                        "GLOBAL_SUPPORTED"
                        if len(projects) >= 5 and len(domains) >= 3
                        else "CROSS_DOMAIN_CANDIDATE"
                    )
                counterexamples = [
                    {
                        "outcome_id": str(item["id"]),
                        "decision_id": item["data"].get("decision_id"),
                        "project_id": str(item["project_id"]),
                        "domain": item["data"].get("domain"),
                        "label": item["data"].get("label"),
                        "failure_conditions": item["data"].get("failure_conditions", []),
                    }
                    for item in rows
                    if str(item["data"].get("label")) in negative_labels
                ]
                counts = {
                    "historical_successes": successes,
                    "historical_failures": failures,
                    "high_information_count": sum(
                        item["data"].get("realized_information_gain") == "HIGH" for item in rows
                    ),
                    "medium_information_count": sum(
                        item["data"].get("realized_information_gain") == "MEDIUM"
                        for item in rows
                    ),
                    "low_information_count": sum(
                        item["data"].get("realized_information_gain") == "LOW" for item in rows
                    ),
                    "zero_information_count": sum(
                        item["data"].get("realized_information_gain") == "ZERO"
                        for item in rows
                    ),
                }
                pattern_data = _json_document(
                    {
                        "strategy_key": key,
                        "scope_level": scope_level,
                        "scope_discriminator": discriminator,
                        "problem_signature": outcome_data["problem_signature"],
                        "research_stage": outcome_data.get("research_stage"),
                        "conditions": outcome_data.get("conditions", {}),
                        "action_type": outcome_data["action_type"],
                        "action_parameters_pattern": outcome_data.get(
                            "action_parameters_pattern", {}
                        ),
                        **counts,
                        "mean_compute_cost": self._mean_numeric(
                            rows, "actual_compute_cost"
                        ),
                        "mean_engineering_cost": self._mean_numeric(
                            rows, "actual_engineering_cost"
                        ),
                        "projects_observed": projects,
                        "domains_observed": domains,
                        "counterexamples": counterexamples,
                        "applicability_conditions": outcome_data.get(
                            "applicability_conditions", {}
                        ),
                        "failure_conditions": sorted(
                            {
                                str(condition)
                                for item in rows
                                for condition in item["data"].get("failure_conditions", [])
                            }
                        ),
                        "support_level": support_level,
                        "decision_ids": sorted(
                            {
                                str(item["data"].get("decision_id"))
                                for item in rows
                                if item["data"].get("decision_id")
                            }
                        ),
                        "outcome_ids": sorted(str(item["id"]) for item in rows),
                        "experiment_run_ids": sorted(
                            {
                                str(run_id)
                                for item in rows
                                for run_id in item["data"].get("experiment_run_ids", [])
                            }
                        ),
                        "evidence_family_ids": sorted(
                            {
                                str(family_id)
                                for item in rows
                                for family_id in item["data"].get(
                                    "evidence_family_ids", []
                                )
                            }
                        ),
                        "updated_at": now.isoformat(),
                    }
                )
                object_status = "WEAKENED" if support_level in {"WEAKENED", "REFUTED"} else "ACTIVE"
                if existing_pattern:
                    pattern_id = existing_pattern["id"]
                    previous_support = existing_pattern["data"].get("support_level")
                    cur.execute(
                        "UPDATE research_objects SET status=%s,data=%s WHERE id=%s",
                        (object_status, json.dumps(pattern_data), pattern_id),
                    )
                    event_type = (
                        "STRATEGY_PATTERN_WEAKENED"
                        if support_level in {"WEAKENED", "REFUTED", "CONTEXT_DEPENDENT"}
                        else "STRATEGY_PATTERN_STRENGTHENED"
                    )
                    if previous_support != support_level and scope_level != "PROJECT":
                        self._event(
                            cur,
                            project_id,
                            "STRATEGY_PATTERN_SCOPE_CHANGED",
                            pattern_id,
                            {"from": previous_support, "to": support_level},
                        )
                else:
                    pattern_id = uuid.uuid4()
                    cur.execute(
                        "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                        "VALUES(%s,%s,'ResearchStrategyPattern',%s,%s,%s)",
                        (
                            pattern_id,
                            project_id,
                            object_status,
                            json.dumps(pattern_data),
                            now,
                        ),
                    )
                    event_type = "STRATEGY_PATTERN_CREATED"
                self._event(cur, project_id, event_type, pattern_id, pattern_data)
                strategy_outcome_data = {
                    "strategy_pattern_id": str(pattern_id),
                    "decision_outcome_id": str(outcome_id),
                    "decision_id": decision_id,
                    "label": outcome_data.get("label"),
                    "scope_level": scope_level,
                    "eligible_for_learning": eligible_current["eligible"],
                    "eligibility": eligible_current,
                }
                cur.execute(
                    "SELECT id,data FROM research_objects WHERE kind='StrategyOutcome' "
                    "AND data->>'decision_outcome_id'=%s AND data->>'scope_level'=%s FOR UPDATE",
                    (str(outcome_id), scope_level),
                )
                existing_strategy_outcome = cur.fetchone()
                if existing_strategy_outcome:
                    strategy_outcome_id = existing_strategy_outcome["id"]
                    strategy_outcome_data = {
                        **existing_strategy_outcome["data"],
                        **strategy_outcome_data,
                        "updated_at": now.isoformat(),
                    }
                    cur.execute(
                        "UPDATE research_objects SET data=%s WHERE id=%s",
                        (json.dumps(strategy_outcome_data), strategy_outcome_id),
                    )
                else:
                    strategy_outcome_id = uuid.uuid4()
                    cur.execute(
                        "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                        "VALUES(%s,%s,'StrategyOutcome','RESULT_INSPECTED',%s,%s)",
                        (
                            strategy_outcome_id,
                            project_id,
                            json.dumps(strategy_outcome_data),
                            now,
                        ),
                    )
                self._event(
                    cur,
                    project_id,
                    "STRATEGY_OUTCOME_RECORDED",
                    strategy_outcome_id,
                    strategy_outcome_data,
                )
                patterns.append(
                    {
                        "id": str(pattern_id),
                        "project_id": project_id,
                        "kind": "ResearchStrategyPattern",
                        "status": object_status,
                        "data": pattern_data,
                    }
                )

            if eligible_current["eligible"] and outcome_data.get("label") in negative_labels:
                for pattern_id in outcome_data.get("retrieved_strategy_pattern_ids", []):
                    cur.execute(
                        "SELECT id,project_id,status,data FROM research_objects "
                        "WHERE id=%s AND kind='ResearchStrategyPattern' FOR UPDATE",
                        (pattern_id,),
                    )
                    transferred = cur.fetchone()
                    if not transferred or str(transferred["project_id"]) == project_id:
                        continue
                    counterexample = {
                        "outcome_id": str(outcome_id),
                        "decision_id": decision_id,
                        "project_id": project_id,
                        "domain": outcome_data.get("domain"),
                        "label": outcome_data.get("label"),
                        "failure_conditions": outcome_data.get("failure_conditions", []),
                    }
                    data = {**transferred["data"]}
                    existing_counterexamples = list(data.get("counterexamples", []))
                    if not any(
                        item.get("outcome_id") == str(outcome_id)
                        for item in existing_counterexamples
                    ):
                        existing_counterexamples.append(counterexample)
                    data.update(
                        {
                            "counterexamples": existing_counterexamples,
                            "support_level": "CONTEXT_DEPENDENT",
                            "updated_at": now.isoformat(),
                        }
                    )
                    cur.execute(
                        "UPDATE research_objects SET data=%s WHERE id=%s",
                        (json.dumps(data), pattern_id),
                    )
                    self._event(
                        cur,
                        transferred["project_id"],
                        "STRATEGY_NEGATIVE_TRANSFER_RECORDED",
                        pattern_id,
                        counterexample,
                    )

        return {
            "decision": {"id": decision_id, "status": "COMPLETED", "data": decision_data},
            "outcome": {
                "id": str(outcome_id),
                "project_id": project_id,
                "kind": "ResearchDecisionOutcome",
                "status": "RESULT_INSPECTED",
                "data": materialized_outcome,
            },
            "after_situation": {
                "id": str(after_situation_id),
                "project_id": project_id,
                "kind": "ResearchSituation",
                "status": "ACTIVE",
                "data": after_situation_data,
            },
            "strategy_patterns": patterns,
            "technically_inspected_experiment_runs": technically_inspected_runs,
        }

    @staticmethod
    def _mean_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
        values = [
            float(item["data"][field])
            for item in rows
            if isinstance(item["data"].get(field), (int, float))
            and not isinstance(item["data"].get(field), bool)
        ]
        return round(sum(values) / len(values), 6) if values else None

    def technical_result_inspection_apply(
        self,
        *,
        run_id: str,
        decision_id: str,
        actual_information_gain: str,
        information_gain_basis: list[str],
        inspection: dict[str, Any],
    ) -> dict:
        """Close a failed execution without manufacturing scientific evidence."""
        if actual_information_gain not in {"ZERO", "INVALID", "UNKNOWN"}:
            raise GPUError("INVALID_INFORMATION_GAIN", actual_information_gain)
        if not information_gain_basis:
            raise GPUError(
                "INFORMATION_GAIN_BASIS_REQUIRED",
                "Provide what technical uncertainty or evidence changed",
            )
        try:
            run_id = str(uuid.UUID(run_id))
            decision_id = str(uuid.UUID(decision_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise GPUError("INVALID_RESEARCH_OBJECT_ID", str(exc)) from exc
        now = datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects "
                "WHERE id=ANY(%s::uuid[]) FOR UPDATE",
                ([run_id, decision_id],),
            )
            rows = {str(row["id"]): row for row in cur.fetchall()}
            if len(rows) != 2:
                missing = next(item for item in (run_id, decision_id) if item not in rows)
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", missing)
            run, decision = rows[run_id], rows[decision_id]
            if run["kind"] != "ExperimentRun":
                raise GPUError("NOT_AN_EXPERIMENTRUN", run_id)
            if decision["kind"] != "ResearchDecision":
                raise GPUError("NOT_A_RESEARCHDECISION", decision_id)
            if str(run["project_id"]) != str(decision["project_id"]):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", "Run and decision differ")
            if run["status"] not in {"failed", "cancelled", "unknown"}:
                raise GPUError("TECHNICAL_INSPECTION_REQUIRES_FAILED_RUN", run["status"])
            prior = run["data"].get("inspection")
            if prior is not None:
                if (
                    prior.get("inspection_kind") == "TECHNICAL_FAILURE"
                    and prior.get("decision_id") == decision_id
                ):
                    return {
                        "run": {"id": run_id, "status": run["status"], "data": run["data"]},
                        "decision": {
                            "id": decision_id,
                            "status": decision["status"],
                            "data": decision["data"],
                        },
                        "idempotent_replay": True,
                    }
                raise GPUError("EXPERIMENT_RESULT_ALREADY_INSPECTED", run_id)
            inspection_data = _json_document(
                {
                    **inspection,
                    "inspection_kind": "TECHNICAL_FAILURE",
                    "decision_id": decision_id,
                    "assessed_at": now.isoformat(),
                    "execution_status": run["status"],
                    "scientific_result": "NOT_ASSESSED",
                    "actual_information_gain": actual_information_gain,
                    "information_gain_basis": information_gain_basis,
                }
            )
            run_data = _json_document({**run["data"], "inspection": inspection_data})
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(run_data), run_id),
            )
            self._event(
                cur,
                str(run["project_id"]),
                "EXPERIMENT_FAILURE_INSPECTED",
                run_id,
                inspection_data,
            )
        return {
            "run": {"id": run_id, "status": run["status"], "data": run_data},
            "decision": {
                "id": decision_id,
                "status": decision["status"],
                "data": decision["data"],
            },
            "technical_non_scientific": True,
        }

    def result_assessment_apply(
        self,
        *,
        run_id: str,
        decision_id: str,
        hypothesis_id: str,
        agenda_item_id: str,
        evidence_data: dict[str, Any],
        hypothesis_transition: str,
        rationale: str,
        inspection: dict[str, Any],
        agenda_status: str,
        actual_information_gain: str,
        information_gain_basis: list[str] | None = None,
        causal_edge_id: str | None = None,
        causal_edge_status: str | None = None,
    ) -> dict:
        """Apply a complete scientific result assessment atomically."""
        if hypothesis_transition not in {"SUPPORTED", "SURVIVES_INITIAL_TEST", "WEAKENED", "REFUTED", "INCONCLUSIVE"}:
            raise GPUError("INVALID_RESEARCH_OBJECT_STATUS", hypothesis_transition)
        if agenda_status not in {"OPEN", "ACTIVE", "RESOLVED", "BLOCKED", "DEFERRED"}:
            raise GPUError("INVALID_RESEARCH_OBJECT_STATUS", agenda_status)
        if actual_information_gain not in {"HIGH", "MEDIUM", "LOW", "ZERO", "INVALID", "UNKNOWN"}:
            raise GPUError("INVALID_INFORMATION_GAIN", actual_information_gain)
        if information_gain_basis is not None and (
            not isinstance(information_gain_basis, list)
            or any(not isinstance(item, str) for item in information_gain_basis)
        ):
            raise GPUError("INVALID_INFORMATION_GAIN_BASIS", "information_gain_basis must be a list of strings")
        basis = list(information_gain_basis or [])
        if actual_information_gain not in {"ZERO", "INVALID", "UNKNOWN"} and not basis:
            basis = ["HYPOTHESIS_TRANSITION"] if hypothesis_transition != "INCONCLUSIVE" else []
        if actual_information_gain not in {"ZERO", "INVALID", "UNKNOWN"} and not basis:
            raise GPUError("INFORMATION_GAIN_BASIS_REQUIRED", "Provide what uncertainty or evidence changed")
        if causal_edge_status == "INTERVENTION_SUPPORTED":
            scope = evidence_data.get("scope")
            bounded_dimensions = (
                "models",
                "architectures",
                "checkpoints",
                "datasets",
                "objects",
            )
            if (
                not isinstance(scope, dict)
                or not any(scope.get(field) for field in bounded_dimensions)
                or not scope.get("interventions")
                or not scope.get("metrics")
            ):
                raise GPUError(
                    "CAUSAL_PROMOTION_SCOPE_REQUIRED",
                    "Intervention support requires explicit bounded experimental scope",
                )
        if causal_edge_status == "REFUTED" and hypothesis_transition in {
            "SUPPORTED",
            "SURVIVES_INITIAL_TEST",
        }:
            raise GPUError(
                "WORLD_MODEL_CONSISTENCY_ERROR",
                "A positive hypothesis transition cannot require a refuted causal edge",
            )
        if causal_edge_status == "INTERVENTION_SUPPORTED" and hypothesis_transition in {
            "WEAKENED",
            "REFUTED",
        }:
            raise GPUError(
                "WORLD_MODEL_CONSISTENCY_ERROR",
                "A weakened hypothesis cannot simultaneously promote its causal edge",
            )
        now, evidence_id = datetime.now(UTC), uuid.uuid4()
        try:
            run_id = str(uuid.UUID(run_id))
            decision_id = str(uuid.UUID(decision_id))
            hypothesis_id = str(uuid.UUID(hypothesis_id))
            agenda_item_id = str(uuid.UUID(agenda_item_id))
            causal_edge_id = str(uuid.UUID(causal_edge_id)) if causal_edge_id else None
        except (ValueError, AttributeError) as exc:
            raise GPUError("INVALID_RESEARCH_OBJECT_ID", str(exc)) from exc
        identifiers = [run_id, decision_id, hypothesis_id, agenda_item_id]
        if causal_edge_id:
            identifiers.append(causal_edge_id)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects "
                "WHERE id=ANY(%s::uuid[]) FOR UPDATE",
                (identifiers,),
            )
            rows = {str(row["id"]): row for row in cur.fetchall()}
            if len(rows) != len(set(identifiers)):
                missing = next(item for item in identifiers if item not in rows)
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", missing)
            run, decision = rows[run_id], rows[decision_id]
            hypothesis, agenda = rows[hypothesis_id], rows[agenda_item_id]
            expected_kinds = (
                (run, "ExperimentRun"),
                (decision, "ResearchDecision"),
                (hypothesis, "Hypothesis"),
                (agenda, "AgendaItem"),
            )
            for row, kind in expected_kinds:
                if row["kind"] != kind:
                    raise GPUError(f"NOT_A_{kind.upper()}", str(row["id"]))
            project_id = str(run["project_id"])
            if any(str(row["project_id"]) != project_id for row, _ in expected_kinds):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", "Assessment inputs differ")
            if run["status"] == "RESULT_INSPECTED":
                prior = run["data"].get("inspection", {})
                if self._assessment_replay_matches(prior, decision["data"], run_id, decision_id):
                    return {
                        "run": {"id": run_id, "status": run["status"], "data": run["data"]},
                        "evidence": self._object_row(cur, prior["evidence_id"]),
                        "hypothesis": {
                            "id": hypothesis_id,
                            "status": hypothesis["status"],
                            "data": hypothesis["data"],
                        },
                        "agenda_item": {
                            "id": agenda_item_id,
                            "status": agenda["status"],
                            "data": agenda["data"],
                        },
                        "decision": {
                            "id": decision_id,
                            "status": decision["status"],
                            "data": decision["data"],
                        },
                        "world_model_update": None,
                        "idempotent_replay": True,
                    }
                raise GPUError("EXPERIMENT_RESULT_ALREADY_INSPECTED", run_id)
            if run["status"] not in {"completed", "RESULT_NOT_INSPECTED", "failed", "cancelled", "unknown"}:
                raise GPUError("EXPERIMENT_RESULT_NOT_READY", run["status"])

            independence_key = f"ORIGIN:{run_id}"
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtext(%s))",
                (f"evidence-family:{project_id}:{independence_key}",),
            )
            cur.execute(
                "SELECT id,data FROM research_objects WHERE project_id=%s "
                "AND kind='EvidenceFamily' AND data->>'independence_key'=%s FOR UPDATE",
                (project_id, independence_key),
            )
            family = cur.fetchone()
            supporting_transition = hypothesis_transition in {
                "SUPPORTED",
                "SURVIVES_INITIAL_TEST",
            }
            contradicting_transition = hypothesis_transition in {"WEAKENED", "REFUTED"}
            if family:
                canonical_family = family["data"]
                if (
                    str(canonical_family.get("origin_id")) != run_id
                    or canonical_family.get("origin_type") != "EXPERIMENT"
                    or "EXPERIMENT" not in canonical_family.get("origin_roles", [])
                    or canonical_family.get("derived_from_evidence_family_id") is not None
                    or canonical_family.get("dependency_note") is not None
                ):
                    raise GPUError("EVIDENCE_FAMILY_KEY_CONFLICT", independence_key)
                family_id = family["id"]
                family_data = {**family["data"]}
            else:
                family_id = uuid.uuid4()
                family_data = {
                    "origin_type": "EXPERIMENT",
                    "origin_roles": ["EXPERIMENT"],
                    "origin_id": run_id,
                    "independence_key": independence_key,
                    "description": "Inspected result from one canonical ExperimentRun",
                    "derived_from_evidence_family_id": None,
                    "dependency_note": None,
                    "supporting_entity_ids": [],
                    "contradicting_entity_ids": [],
                    "derived_record_ids": [],
                }
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'EvidenceFamily','ACTIVE',%s,%s)",
                    (family_id, project_id, json.dumps(family_data), now),
                )
                self._event(
                    cur,
                    project_id,
                    "EVIDENCE_FAMILY_CREATED",
                    family_id,
                    {
                        "origin_type": "EXPERIMENT",
                        "origin_id": run_id,
                        "independence_key": independence_key,
                    },
                )
            family_id_text = str(family_id)
            family_data["derived_record_ids"] = list(
                dict.fromkeys([*family_data.get("derived_record_ids", []), str(evidence_id)])
            )
            if supporting_transition:
                family_data["supporting_entity_ids"] = list(
                    dict.fromkeys(
                        [
                            *family_data.get("supporting_entity_ids", []),
                            hypothesis_id,
                            *([causal_edge_id] if causal_edge_id else []),
                        ]
                    )
                )
            if contradicting_transition:
                family_data["contradicting_entity_ids"] = list(
                    dict.fromkeys(
                        [
                            *family_data.get("contradicting_entity_ids", []),
                            hypothesis_id,
                            *([causal_edge_id] if causal_edge_id else []),
                        ]
                    )
                )
            cur.execute(
                "UPDATE research_objects SET data=%s WHERE id=%s",
                (json.dumps(family_data), family_id),
            )
            evidence_data = {
                **evidence_data,
                "evidence_family_ids": [family_id_text],
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'EvidenceUnit','VERIFIED_REAL',%s,%s)",
                (evidence_id, project_id, json.dumps(evidence_data), now),
            )
            self._event(
                cur, project_id, "EXPERIMENT_EVIDENCE_INSPECTED", evidence_id, evidence_data
            )
            hypothesis_data = {**hypothesis["data"], "assessment_rationale": rationale}
            hypothesis_data["evidence_family_ids"] = list(
                dict.fromkeys(
                    [*hypothesis_data.get("evidence_family_ids", []), family_id_text]
                )
            )
            if supporting_transition:
                hypothesis_data["supporting_evidence_family_ids"] = list(
                    dict.fromkeys(
                        [
                            *hypothesis_data.get("supporting_evidence_family_ids", []),
                            family_id_text,
                        ]
                    )
                )
            if contradicting_transition:
                hypothesis_data["contradicting_evidence_family_ids"] = list(
                    dict.fromkeys(
                        [
                            *hypothesis_data.get("contradicting_evidence_family_ids", []),
                            family_id_text,
                        ]
                    )
                )
            cur.execute(
                "UPDATE research_objects SET status=%s,data=%s WHERE id=%s",
                (hypothesis_transition, json.dumps(hypothesis_data), hypothesis_id),
            )
            self._event(
                cur,
                project_id,
                f"HYPOTHESIS_{hypothesis_transition}",
                hypothesis_id,
                {"rationale": rationale},
            )
            inspection_data = {
                **inspection,
                "evidence_id": str(evidence_id),
                "decision_id": decision_id,
            }
            run_data = {**run["data"], "inspection": inspection_data}
            cur.execute(
                "UPDATE research_objects SET status='RESULT_INSPECTED',data=%s WHERE id=%s",
                (json.dumps(run_data), run_id),
            )
            self._event(
                cur,
                project_id,
                "EXPERIMENT_RESULT_INSPECTED",
                run_id,
                inspection_data,
            )
            agenda_data = {**agenda["data"], "status_rationale": rationale}
            cur.execute(
                "UPDATE research_objects SET status=%s,data=%s WHERE id=%s",
                (agenda_status, json.dumps(agenda_data), agenda_item_id),
            )
            self._event(
                cur,
                project_id,
                "AGENDA_ITEM_STATUS_CHANGED",
                agenda_item_id,
                {"status": agenda_status, "status_rationale": rationale},
            )

            edge_result = None
            if causal_edge_id:
                edge = rows[causal_edge_id]
                if edge["kind"] != "CausalEdge":
                    raise GPUError("NOT_A_CAUSALEDGE", causal_edge_id)
                if str(edge["project_id"]) != project_id:
                    raise GPUError("RESEARCH_PROJECT_MISMATCH", causal_edge_id)
                supporting = list(edge["data"].get("supporting_ids", []))
                against = list(edge["data"].get("against_ids", []))
                evidence_targets = {
                    "OBSERVED_ASSOCIATION": supporting,
                    "HYPOTHESIZED_CAUSAL": supporting,
                    "INTERVENTION_SUPPORTED": supporting,
                    "WEAKENED": against,
                    "REFUTED": against,
                    "UNKNOWN": None,
                }
                target = evidence_targets.get(str(causal_edge_status))
                if target is None and causal_edge_status != "UNKNOWN":
                    raise GPUError("INVALID_CAUSAL_EDGE_STATUS", str(causal_edge_status))
                if target is not None and str(evidence_id) not in target:
                    target.append(str(evidence_id))
                edge_data = {
                    **edge["data"],
                    "edge_status": causal_edge_status,
                    "supporting_ids": supporting,
                    "against_ids": against,
                    "decision_id": decision_id,
                    "last_update_rationale": rationale,
                }
                edge_data["evidence_family_ids"] = list(
                    dict.fromkeys(
                        [*edge_data.get("evidence_family_ids", []), family_id_text]
                    )
                )
                family_field = None
                if causal_edge_status in {
                    "OBSERVED_ASSOCIATION",
                    "HYPOTHESIZED_CAUSAL",
                    "INTERVENTION_SUPPORTED",
                }:
                    family_field = "supporting_evidence_family_ids"
                elif causal_edge_status in {"WEAKENED", "REFUTED"}:
                    family_field = "contradicting_evidence_family_ids"
                if family_field:
                    edge_data[family_field] = list(
                        dict.fromkeys([*edge_data.get(family_field, []), family_id_text])
                    )
                cur.execute(
                    "SELECT id,data FROM research_objects WHERE project_id=%s "
                    "AND kind='EvidenceFamily' FOR SHARE",
                    (project_id,),
                )
                all_families = {str(item["id"]): item["data"] for item in cur.fetchall()}

                def family_roots(family_ids: list[str]) -> dict[str, dict[str, Any]]:
                    roots = {}
                    for linked_id in family_ids:
                        current, visited = str(linked_id), set()
                        while current:
                            if current in visited:
                                raise GPUError(
                                    "EVIDENCE_FAMILY_DEPENDENCY_CYCLE", current
                                )
                            visited.add(current)
                            data = all_families.get(current)
                            if data is None:
                                raise GPUError("NOT_AN_EVIDENCEFAMILY", current)
                            parent = data.get("derived_from_evidence_family_id")
                            if not parent:
                                roots[current] = data
                                break
                            current = str(parent)
                    return roots

                support_roots = family_roots(
                    edge_data.get("supporting_evidence_family_ids", [])
                )
                against_roots = family_roots(
                    edge_data.get("contradicting_evidence_family_ids", [])
                )
                support_count = len(support_roots)
                against_count = len(against_roots)
                scope = evidence_data["scope"]
                architectures = {str(item) for item in scope.get("architectures", [])}
                intervention_roles = {"EXPERIMENT", "INDEPENDENT_REPLICATION"}
                if causal_edge_status == "INTERVENTION_SUPPORTED" and any(
                    not intervention_roles.intersection(
                        {root.get("origin_type"), *root.get("origin_roles", [])}
                    )
                    for root in support_roots.values()
                ):
                    raise GPUError(
                        "CAUSAL_PROMOTION_INTERVENTION_FAMILY_REQUIRED",
                        "Every independent causal support root must be experimental",
                    )
                if support_count == 0:
                    support_level = "NONE"
                elif support_count == 1:
                    support_level = "SINGLE_INTERVENTION"
                elif len(architectures) > 1:
                    support_level = "CROSS_ARCHITECTURE_SUPPORT"
                elif support_count >= 3:
                    support_level = "ROBUST_WITHIN_ARCHITECTURE"
                else:
                    support_level = "REPLICATED_WITHIN_SCOPE"
                edge_data.update(
                    {
                        "scope": scope,
                        "support_level": support_level,
                        "independent_support_family_count": support_count,
                        "independent_contradicting_family_count": against_count,
                        "replication_count": max(support_count - 1, 0),
                        "matched_control_count": int(
                            edge["data"].get("matched_control_count", 0)
                        )
                        + int(evidence_data.get("matched_control_passed") is True),
                        "counter_intervention_count": int(
                            edge["data"].get("counter_intervention_count", 0)
                        ),
                    }
                )
                if causal_edge_status == "INTERVENTION_SUPPORTED" and against_count:
                    raise GPUError(
                        "WORLD_MODEL_CONSISTENCY_ERROR",
                        "Resolve contradicting EvidenceFamilies before causal promotion",
                    )
                if causal_edge_status == "INTERVENTION_SUPPORTED":
                    cur.execute(
                        "SELECT id FROM research_objects WHERE project_id=%s "
                        "AND kind='CausalEdge' AND id<>%s "
                        "AND data->>'source_id'=%s AND data->>'target_id'=%s "
                        "AND data->>'relation'=%s AND coalesce(data->'scope','{}'::jsonb)=%s::jsonb "
                        "AND data->>'edge_status'='REFUTED' LIMIT 1",
                        (
                            project_id,
                            causal_edge_id,
                            edge_data.get("source_id"),
                            edge_data.get("target_id"),
                            edge_data.get("relation"),
                            json.dumps(scope),
                        ),
                    )
                    if cur.fetchone():
                        raise GPUError(
                            "WORLD_MODEL_CONSISTENCY_ERROR",
                            "An identical-scope refuted edge blocks causal promotion",
                        )
                if causal_edge_status == "REFUTED":
                    cur.execute(
                        "SELECT id FROM research_objects WHERE project_id=%s "
                        "AND kind='Hypothesis' AND status=ANY(%s) "
                        "AND ((coalesce(data->'required_edge_ids','[]'::jsonb) ? %s) "
                        "OR (coalesce(data->'causal_edge_ids','[]'::jsonb) ? %s)) LIMIT 1",
                        (
                            project_id,
                            ["ACTIVE", "SURVIVES_INITIAL_TEST", "SUPPORTED"],
                            causal_edge_id,
                            causal_edge_id,
                        ),
                    )
                    active_dependency = cur.fetchone()
                    if active_dependency and str(active_dependency["id"]) != hypothesis_id:
                        raise GPUError(
                            "WORLD_MODEL_CONSISTENCY_ERROR",
                            "Another active hypothesis still requires the refuted edge",
                        )
                edge_status = (
                    "VERIFIED_REAL"
                    if causal_edge_status == "INTERVENTION_SUPPORTED"
                    else "RESULT_INSPECTED"
                )
                cur.execute(
                    "UPDATE research_objects SET status=%s,data=%s WHERE id=%s",
                    (edge_status, json.dumps(edge_data), causal_edge_id),
                )
                self._event(
                    cur,
                    project_id,
                    "CAUSAL_EDGE_STATUS_CHANGED",
                    causal_edge_id,
                    {"status": edge_status, "edge_status": causal_edge_status},
                )
                if family_field:
                    cur.execute(
                        "INSERT INTO research_edges VALUES(%s,%s,%s,%s) "
                        "ON CONFLICT DO NOTHING",
                        (
                            family_id,
                            causal_edge_id,
                            (
                                "SUPPORTS"
                                if family_field == "supporting_evidence_family_ids"
                                else "CONTRADICTS"
                            ),
                            now,
                        ),
                    )
                world_model_id = edge_data["world_model_id"]
                cur.execute(
                    "SELECT project_id,status,data FROM research_objects "
                    "WHERE id=%s AND kind='WorldModel' FOR UPDATE",
                    (world_model_id,),
                )
                model = cur.fetchone()
                if not model:
                    raise GPUError("NOT_A_WORLDMODEL", world_model_id)
                if str(model["project_id"]) != project_id:
                    raise GPUError("RESEARCH_PROJECT_MISMATCH", world_model_id)
                version_id = uuid.uuid4()
                version_number = int(model["data"].get("version", 0)) + 1
                version_data = {
                    "world_model_id": world_model_id,
                    "version": version_number,
                    "parent_version_id": model["data"].get("current_version_id"),
                    "changes": {
                        "edges_status_changed": [
                            {
                                "edge_id": causal_edge_id,
                                "from": edge["data"].get("edge_status"),
                                "to": causal_edge_status,
                            }
                        ]
                    },
                    "evidence_ids": [str(evidence_id)],
                    "decision_id": decision_id,
                    "timestamp": now.isoformat(),
                }
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'WorldModelVersion','IMPLEMENTED_UNVERIFIED',%s,%s)",
                    (version_id, project_id, json.dumps(version_data), now),
                )
                self._event(
                    cur,
                    project_id,
                    "WORLD_MODEL_VERSION_CREATED",
                    version_id,
                    version_data,
                )
                model_data = {
                    **model["data"],
                    "current_version_id": str(version_id),
                    "version": version_number,
                }
                cur.execute(
                    "UPDATE research_objects SET data=%s WHERE id=%s",
                    (json.dumps(model_data), world_model_id),
                )
                self._event(
                    cur,
                    project_id,
                    "WORLD_MODEL_VERSION_ADVANCED",
                    world_model_id,
                    {"version": version_number},
                )
                edge_result = {
                    "edge": {"id": causal_edge_id, "status": edge_status, "data": edge_data},
                    "version": {
                        "id": str(version_id),
                        "status": "IMPLEMENTED_UNVERIFIED",
                        "data": version_data,
                    },
                }

            decision_data = {
                **decision["data"],
                **classify_decision_data(decision["data"]),
                "cycle_status": "CLOSED",
                "scientific_verification": "RESULT_INSPECTED",
                "scientific_result": {
                    "status": "INSPECTED",
                    "hypothesis_transition": hypothesis_transition,
                    "prediction_outcome": evidence_data["prediction_outcome"],
                    "guard_condition_outcome": evidence_data.get("guard_condition_outcome"),
                },
                "system_result": {
                    "execution_success": run["status"] in {"completed", "RESULT_NOT_INSPECTED"},
                    "run_status": run["status"],
                },
                "strategy_result": {
                    "eligible_for_learning": False,
                    "reason": "DecisionOutcome required for strategy learning eligibility",
                },
                "actual_information_gain": actual_information_gain,
                "information_gain_basis": basis,
                "outcome": {
                    "run_id": run_id,
                    "evidence_id": str(evidence_id),
                    "hypothesis_transition": hypothesis_transition,
                    "prediction_outcome": evidence_data["prediction_outcome"],
                },
                "hindsight_assessment": rationale,
            }
            cur.execute(
                "UPDATE research_objects SET status='COMPLETED',data=%s WHERE id=%s",
                (json.dumps(decision_data), decision_id),
            )
            self._event(
                cur,
                project_id,
                "RESEARCH_DECISION_OUTCOME_RECORDED",
                decision_id,
                {"status": "COMPLETED", "outcome": decision_data["outcome"]},
            )
            for target_id, relationship in (
                (str(evidence_id), "DERIVED"),
                (
                    hypothesis_id,
                    "SUPPORTS"
                    if supporting_transition
                    else "CONTRADICTS"
                    if contradicting_transition
                    else "DERIVED",
                ),
            ):
                cur.execute(
                    "INSERT INTO research_edges VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    (family_id, target_id, relationship, now),
                )
            cur.execute("SELECT state FROM research_projects WHERE id=%s FOR UPDATE", (project_id,))
            project = cur.fetchone()
            if not project:
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            fact = {
                "evidence_id": str(evidence_id),
                "statement": evidence_data["prediction_outcome"],
                "scope": evidence_data["scope"],
                "verification_status": "VERIFIED_REAL",
            }
            project_state = {
                **project["state"],
                "established_facts": [*project["state"].get("established_facts", []), fact],
            }
            cur.execute(
                "UPDATE research_projects SET state=%s WHERE id=%s",
                (json.dumps(project_state), project_id),
            )
            self._event(
                cur,
                project_id,
                "RESEARCH_STATE_UPDATED",
                None,
                {"append": {"established_facts": [fact]}},
            )
        return {
            "run": {"id": run_id, "status": "RESULT_INSPECTED", "data": run_data},
            "evidence": {
                "id": str(evidence_id),
                "project_id": project_id,
                "kind": "EvidenceUnit",
                "status": "VERIFIED_REAL",
                "data": evidence_data,
            },
            "hypothesis": {
                "id": hypothesis_id,
                "status": hypothesis_transition,
                "data": hypothesis_data,
            },
            "agenda_item": {"id": agenda_item_id, "status": agenda_status, "data": agenda_data},
            "world_model_update": edge_result,
            "decision": {"id": decision_id, "status": "COMPLETED", "data": decision_data},
        }

    @staticmethod
    def _object_row(cur, object_id: str) -> dict:
        cur.execute(
            "SELECT id,project_id,kind,status,data,created_at FROM research_objects WHERE id=%s",
            (object_id,),
        )
        row = cur.fetchone()
        if not row:
            raise GPUError("RESEARCH_OBJECT_NOT_FOUND", object_id)
        return row

    @staticmethod
    def _assessment_replay_matches(
        inspection: dict[str, Any], decision_data: dict[str, Any], run_id: str, decision_id: str
    ) -> bool:
        """Match current snapshots and legacy inspected runs to their original decision."""
        if inspection.get("decision_id") == decision_id:
            return True
        outcome = decision_data.get("outcome", {})
        return (
            "decision_id" not in inspection
            and outcome.get("run_id") == run_id
            and outcome.get("evidence_id") == inspection.get("evidence_id")
        )

    def objects_list(
        self,
        project_id: str,
        kind: str | None = None,
        statuses: set[str] | None = None,
        limit: int | None = 500,
        data_filters: dict[str, Any] | None = None,
        as_of: datetime | str | None = None,
    ) -> list[dict]:
        """List canonical research objects with typed filters for Brain services."""
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                sql = (
                    "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                    "WHERE project_id=%s"
                )
                args: list[Any] = [project_id]
            else:
                sql = (
                    "WITH latest AS ("
                    "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.project_id,v.kind,"
                    "v.status,v.data,v.object_created_at AS created_at "
                    "FROM research_object_versions v WHERE v.project_id=%s "
                    "AND v.committed_at<=%s "
                    "ORDER BY v.object_id,v.committed_at DESC,"
                    "v.revision_id DESC) "
                    "SELECT id,project_id,kind,status,data,created_at FROM latest WHERE TRUE"
                )
                args = [project_id, cutoff]
            if kind:
                sql += " AND kind=%s"
                args.append(kind)
            if statuses:
                sql += " AND status=ANY(%s)"
                args.append(list(statuses))
            if data_filters:
                sql += " AND data @> %s::jsonb"
                args.append(json.dumps(data_filters))
            sql += " ORDER BY created_at DESC"
            if limit is not None:
                sql += " LIMIT %s"
                args.append(max(limit, 1))
            cur.execute(sql, args)
            return cur.fetchall()

    def objects_global_list(
        self,
        kind: str,
        statuses: set[str] | None = None,
        limit: int | None = 1000,
        as_of: datetime | str | None = None,
    ) -> list[dict]:
        """List one object kind across projects for explicitly scoped strategy transfer."""
        if kind not in RESEARCH_OBJECT_KINDS:
            raise GPUError("INVALID_RESEARCH_OBJECT_KIND", kind)
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                sql = (
                    "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                    "WHERE kind=%s"
                )
                args: list[Any] = [kind]
            else:
                sql = (
                    "WITH latest AS ("
                    "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.project_id,v.kind,"
                    "v.status,v.data,v.object_created_at AS created_at "
                    "FROM research_object_versions v WHERE v.committed_at<=%s "
                    "ORDER BY v.object_id,v.committed_at DESC,v.revision_id DESC) "
                    "SELECT id,project_id,kind,status,data,created_at FROM latest WHERE kind=%s"
                )
                args = [cutoff, kind]
            if statuses:
                sql += " AND status=ANY(%s)"
                args.append(list(statuses))
            sql += " ORDER BY created_at DESC"
            if limit is not None:
                sql += " LIMIT %s"
                args.append(max(limit, 1))
            cur.execute(sql, args)
            return cur.fetchall()

    def objects_count(
        self,
        project_id: str,
        kind: str,
        statuses: set[str] | None = None,
        data_filters: dict[str, Any] | None = None,
        as_of: datetime | str | None = None,
    ) -> int:
        """Count fully filtered canonical objects without materializing a bounded page."""
        cutoff = self._prepare_as_of(as_of)
        if cutoff is None:
            sql = "SELECT count(*) AS count FROM research_objects WHERE project_id=%s AND kind=%s"
            args: list[Any] = [project_id, kind]
        else:
            sql = (
                "WITH latest AS ("
                "SELECT DISTINCT ON (v.object_id) v.object_id,v.kind,v.status,v.data "
                "FROM research_object_versions v WHERE v.project_id=%s "
                "AND v.committed_at<=%s "
                "ORDER BY v.object_id,v.committed_at DESC,"
                "v.revision_id DESC) "
                "SELECT count(*) AS count FROM latest WHERE kind=%s"
            )
            args = [project_id, cutoff, kind]
        if statuses:
            sql += " AND status=ANY(%s)"
            args.append(list(statuses))
        if data_filters:
            sql += " AND data @> %s::jsonb"
            args.append(json.dumps(data_filters))
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            return int(cur.fetchone()["count"])

    def objects_identifiers(
        self,
        project_id: str,
        kind: str,
        statuses: set[str] | None = None,
        as_of: datetime | str | None = None,
    ) -> list[dict]:
        """Return only identity and status fields for complete lightweight classifications."""
        cutoff = self._prepare_as_of(as_of)
        if cutoff is None:
            sql = "SELECT id,status FROM research_objects WHERE project_id=%s AND kind=%s"
            args: list[Any] = [project_id, kind]
        else:
            sql = (
                "WITH latest AS ("
                "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.kind,v.status,"
                "v.object_created_at AS created_at FROM research_object_versions v "
                "WHERE v.project_id=%s AND v.committed_at<=%s "
                "ORDER BY v.object_id,v.committed_at DESC,"
                "v.revision_id DESC) "
                "SELECT id,status FROM latest WHERE kind=%s"
            )
            args = [project_id, cutoff, kind]
        if statuses:
            sql += " AND status=ANY(%s)"
            args.append(list(statuses))
        sql += " ORDER BY created_at DESC"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()

    def experiment_run_first(
        self,
        project_id: str,
        statuses: set[str],
        inspected: bool | None = None,
        as_of: datetime | str | None = None,
    ) -> dict | None:
        """Return the newest matching run without loading its JSON payload."""
        cutoff = self._prepare_as_of(as_of)
        if cutoff is None:
            sql = (
                "SELECT id,status,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='ExperimentRun' AND status=ANY(%s)"
            )
            args: list[Any] = [project_id, list(statuses)]
        else:
            sql = (
                "WITH latest AS ("
                "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.kind,v.status,v.data,"
                "v.object_created_at AS created_at FROM research_object_versions v "
                "WHERE v.project_id=%s AND v.committed_at<=%s "
                "ORDER BY v.object_id,v.committed_at DESC,"
                "v.revision_id DESC) "
                "SELECT id,status,created_at FROM latest "
                "WHERE kind='ExperimentRun' AND status=ANY(%s)"
            )
            args = [project_id, cutoff, list(statuses)]
        if inspected is not None:
            comparison = "<>" if inspected else "="
            sql += (
                " AND coalesce(data->'inspection','null'::jsonb) "
                f"{comparison} 'null'::jsonb"
            )
        sql += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchone()

    @staticmethod
    def run_has_terminal_technical_inspection(run: dict[str, Any]) -> bool:
        """Whether a run has already been closed as non-scientific operations work.

        A failed launch must not be reintroduced as scientific work merely because
        an older runner record still says ``unknown`` or ``running``.  Keep this
        deliberately narrow: ordinary completed scientific results remain subject
        to normal result inspection.
        """
        if run.get("status") in {"TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED"}:
            return True
        data = run.get("data") or {}
        inspection = data.get("inspection")
        if not isinstance(inspection, dict):
            return False
        outcome = str(inspection.get("prediction_outcome", ""))
        return bool(
            inspection.get("technical_non_scientific") is True
            or outcome.startswith("TECHNICAL_")
            or (
                inspection.get("scientific_result") == "NOT_ASSESSED"
                and inspection.get("actual_information_gain") in {"ZERO", "INVALID"}
            )
        )

    @classmethod
    def experiment_run_is_operationally_active(cls, run: dict[str, Any]) -> bool:
        """Single canonical predicate for an execution that can still be recovered."""
        return (
            run.get("status") in {"ACTIVE", "RESERVED", "running", "unknown", "RUNNING"}
            and not cls.run_has_terminal_technical_inspection(run)
        )

    def references_get(
        self, identifiers: list[str], as_of: datetime | str | None = None
    ) -> dict[str, dict]:
        """Fetch a bounded reference set with one database round trip."""
        if len(identifiers) > 200:
            raise GPUError("TOO_MANY_RESEARCH_REFERENCES", "At most 200 references are allowed")
        if not identifiers:
            return {}
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                cur.execute(
                    "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                    "WHERE id=ANY(%s::uuid[])",
                    (identifiers,),
                )
            else:
                cur.execute(
                    "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.project_id,v.kind,"
                    "v.status,v.data,v.object_created_at AS created_at "
                    "FROM research_object_versions v "
                    "WHERE v.object_id=ANY(%s::uuid[]) "
                    "AND v.committed_at<=%s "
                    "ORDER BY v.object_id,v.committed_at DESC,"
                    "v.revision_id DESC",
                    (identifiers, cutoff),
                )
            return {str(row["id"]): row for row in cur.fetchall()}

    def state_get(self, project_id: str, as_of: datetime | str | None = None) -> dict:
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                cur.execute(
                    "SELECT name,question,state FROM research_projects WHERE id=%s", (project_id,)
                )
            else:
                cur.execute(
                    "SELECT name,question,state,valid_from,"
                    "committed_at,legacy_backfill "
                    "FROM research_project_versions "
                    "WHERE project_id=%s AND committed_at<=%s "
                    "ORDER BY committed_at DESC,revision_id DESC LIMIT 1",
                    (project_id, cutoff),
                )
            row = cur.fetchone()
            if not row:
                if cutoff is not None:
                    raise GPUError(
                        "TEMPORAL_SNAPSHOT_UNAVAILABLE", f"{project_id} at {cutoff.isoformat()}"
                    )
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            if cutoff is None:
                cur.execute(
                    "SELECT id,kind,status,data,created_at FROM research_objects "
                    "WHERE project_id=%s ORDER BY created_at DESC",
                    (project_id,),
                )
            else:
                cur.execute(
                    "WITH latest AS ("
                    "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.kind,v.status,v.data,"
                    "v.object_created_at AS created_at FROM research_object_versions v "
                    "WHERE v.project_id=%s AND v.committed_at<=%s "
                    "ORDER BY v.object_id,v.committed_at DESC,"
                    "v.revision_id DESC) "
                    "SELECT id,kind,status,data,created_at FROM latest ORDER BY created_at DESC",
                    (project_id, cutoff),
                )
            objects = cur.fetchall()
            cur.execute(
                "SELECT MAX(created_at) AS latest_scientific_event_at, COUNT(*) AS event_count "
                "FROM research_events WHERE project_id=%s",
                (project_id,),
            )
            event_freshness = cur.fetchone()
            cur.execute(
                "SELECT MAX(created_at) AS latest_object_at, COUNT(*) AS object_count "
                "FROM research_objects WHERE project_id=%s",
                (project_id,),
            )
            object_freshness = cur.fetchone()
            by_kind = lambda kind, statuses=None: [item for item in objects if item["kind"] == kind and (not statuses or item["status"] in statuses)]
            canonical = {
                "research_question": row["question"],
                "established_facts": row["state"].get("established_facts", []),
                "supported_claims": by_kind("Claim", {"SUPPORTED"}),
                "weakened_claims": by_kind("Claim", {"WEAKENED"}),
                "refuted_claims": by_kind("Claim", {"REFUTED"}),
                "unresolved_claims": by_kind("Claim", {"ACTIVE"}),
                "active_anomalies": by_kind("Anomaly", {"ACTIVE"}),
                "active_contradictions": by_kind("Contradiction", {"ACTIVE"}),
                "active_hypotheses": by_kind("Hypothesis", {"ACTIVE", "SURVIVES_INITIAL_TEST"}),
                "refuted_lineages": by_kind("Hypothesis", {"REFUTED"}),
                "completed_experiments": by_kind("ExperimentRun", {"completed"}),
                "active_experiments": [
                    item
                    for item in by_kind("ExperimentRun")
                    if self.experiment_run_is_operationally_active(item)
                ],
                "terminal_non_success_experiments": by_kind(
                    "ExperimentRun", {"failed", "cancelled", "TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED"}
                ),
                "open_experiments": by_kind("Experiment", {"ACTIVE"}),
                "reproduction_status": by_kind("Reproduction"),
                "negative_results": by_kind("NegativeResult"),
                "lessons": by_kind("Lesson"),
                "comparative_lessons": by_kind("ComparativeLesson"),
                "meta_lessons": by_kind("MetaLesson"),
                "experiment_branches": by_kind("ExperimentBranch"),
                "current_best_explanation": row["state"].get("current_best_explanation"),
                "highest_value_unknown": row["state"].get("highest_value_unknown"),
                "next_discriminating_experiments": row["state"].get("next_discriminating_experiments", []),
            }
            world_models = by_kind("WorldModel")
            world_version = (
                world_models[0]["data"].get("version") if world_models else None
            )
            freshness = {
                "canonical_source": "research_objects_and_immutable_events",
                "project_state_role": "CACHE_ONLY",
                "research_state_version": int(object_freshness["object_count"] or 0),
                "world_model_version": world_version,
                "state_updated_at": (
                    object_freshness["latest_object_at"].isoformat()
                    if object_freshness["latest_object_at"] else None
                ),
                "latest_scientific_event_at": (
                    event_freshness["latest_scientific_event_at"].isoformat()
                    if event_freshness["latest_scientific_event_at"] else None
                ),
                "scientific_event_count": int(event_freshness["event_count"] or 0),
            }
            return {
                **row,
                "canonical_state": canonical,
                "state_freshness": freshness,
                "project_state_cache": row["state"],
                "objects": objects,
                "as_of": cutoff.isoformat() if cutoff is not None else None,
            }

    def project_state_update(
        self,
        project_id: str,
        update: dict[str, Any],
        append: dict[str, list[Any]] | None = None,
    ) -> dict:
        allowed = {
            "established_facts",
            "current_best_explanation",
            "highest_value_unknown",
            "next_discriminating_experiments",
        }
        append = append or {}
        unexpected = sorted((set(update) | set(append)) - allowed)
        if unexpected:
            raise GPUError("INVALID_RESEARCH_STATE_FIELDS", ", ".join(unexpected))
        for field in ("established_facts", "next_discriminating_experiments"):
            if field in update and not isinstance(update[field], list):
                raise GPUError("INVALID_RESEARCH_STATE_FIELDS", f"{field} must be a list")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT state FROM research_projects WHERE id=%s FOR UPDATE", (project_id,))
            row = cur.fetchone()
            if not row:
                raise GPUError("RESEARCH_PROJECT_NOT_FOUND", project_id)
            state = {**row["state"]}
            for field, values in append.items():
                if field not in {"established_facts", "next_discriminating_experiments"}:
                    raise GPUError("INVALID_RESEARCH_STATE_FIELDS", f"{field} cannot be appended")
                if not isinstance(values, list):
                    raise GPUError("INVALID_RESEARCH_STATE_FIELDS", f"{field} append must be a list")
                state[field] = [*state.get(field, []), *values]
            state.update(update)
            cur.execute("UPDATE research_projects SET state=%s WHERE id=%s", (json.dumps(state), project_id))
            self._event(
                cur,
                project_id,
                "RESEARCH_STATE_UPDATED",
                None,
                {"update": update, "append": append},
            )
        return state

    def object_get(self, object_id: str, as_of: datetime | str | None = None) -> dict:
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                cur.execute(
                    "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                    "WHERE id=%s",
                    (object_id,),
                )
            else:
                cur.execute(
                    "SELECT object_id AS id,project_id,kind,status,data,"
                    "object_created_at AS created_at,valid_from,"
                    "committed_at,legacy_backfill "
                    "FROM research_object_versions WHERE object_id=%s "
                    "AND committed_at<=%s "
                    "ORDER BY committed_at DESC,revision_id DESC LIMIT 1",
                    (object_id, cutoff),
                )
            row = cur.fetchone()
            if not row:
                if cutoff is not None:
                    raise GPUError(
                        "TEMPORAL_OBJECT_NOT_VISIBLE", f"{object_id} at {cutoff.isoformat()}"
                    )
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

    def run_reserve(
        self,
        experiment_id: str,
        idempotency_key: str,
        job_id: str,
        request_fingerprint: str,
        execution: dict[str, Any],
    ) -> dict:
        """Atomically reserve canonical run/job identity before process submission."""
        if not idempotency_key or len(idempotency_key) > 200:
            raise GPUError("INVALID_IDEMPOTENCY_KEY", "Use a non-empty key up to 200 characters")
        now = datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT project_id,kind,data FROM research_objects WHERE id=%s FOR UPDATE",
                (experiment_id,),
            )
            experiment = cur.fetchone()
            if not experiment:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", experiment_id)
            if experiment["kind"] != "Experiment" or not experiment["data"].get("frozen"):
                raise GPUError("EXPERIMENT_NOT_PREREGISTERED", experiment_id)
            cur.execute(
                "SELECT run_id,job_id,request_fingerprint,status FROM research_execution_attempts "
                "WHERE experiment_id=%s AND idempotency_key=%s",
                (experiment_id, idempotency_key),
            )
            existing = cur.fetchone()
            if existing:
                if existing["request_fingerprint"] != request_fingerprint:
                    raise GPUError(
                        "IDEMPOTENCY_KEY_REUSED",
                        "The execution key is already bound to a different request",
                    )
                return self._execution_mapping(
                    cur, existing["run_id"], existing["job_id"], idempotency_key, True
                )

            run_id = uuid.uuid4()
            run_data = {
                "experiment_id": experiment_id,
                "job_id": job_id,
                "idempotency_key": idempotency_key,
                "request_fingerprint": request_fingerprint,
                **execution,
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ExperimentRun','RESERVED',%s,%s)",
                (run_id, experiment["project_id"], json.dumps(run_data), now),
            )
            cur.execute(
                "INSERT INTO research_execution_attempts "
                "VALUES(%s,%s,%s,%s,%s,'RESERVED',%s,%s)",
                (
                    experiment_id,
                    idempotency_key,
                    run_id,
                    job_id,
                    request_fingerprint,
                    now,
                    now,
                ),
            )
            cur.execute(
                "INSERT INTO research_edges VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (experiment_id, run_id, "HAS_RUN", now),
            )
            self._event(
                cur,
                experiment["project_id"],
                "EXPERIMENT_EXECUTION_RESERVED",
                run_id,
                {
                    "experiment_id": experiment_id,
                    "run_id": str(run_id),
                    "job_id": job_id,
                    "idempotency_key": idempotency_key,
                },
            )
            return self._execution_mapping(cur, run_id, job_id, idempotency_key, False)

    def run_mark_submitted(self, run_id: str) -> dict:
        """Idempotently mark a reserved attempt as submitted and emit its canonical IDs."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT a.experiment_id,a.idempotency_key,a.job_id,a.status,o.project_id,o.data "
                "FROM research_execution_attempts a JOIN research_objects o ON o.id=a.run_id "
                "WHERE a.run_id=%s FOR UPDATE",
                (run_id,),
            )
            row = cur.fetchone()
            if not row:
                raise GPUError("EXPERIMENT_RUN_NOT_FOUND", run_id)
            if row["status"] in {"RESERVED", "unknown"}:
                data = {**row["data"], "submission_status": "SUBMITTED"}
                now = datetime.now(UTC)
                cur.execute(
                    "UPDATE research_objects SET status='running',data=%s WHERE id=%s",
                    (json.dumps(data), run_id),
                )
                cur.execute(
                    "UPDATE research_execution_attempts SET status='SUBMITTED',updated_at=%s "
                    "WHERE run_id=%s",
                    (now, run_id),
                )
                self._event(
                    cur,
                    row["project_id"],
                    "EXPERIMENT_STARTED"
                    if row["status"] == "RESERVED"
                    else "EXPERIMENT_EXECUTION_RECOVERED",
                    run_id,
                    {
                        "experiment_id": str(row["experiment_id"]),
                        "run_id": run_id,
                        "job_id": row["job_id"],
                        "idempotency_key": row["idempotency_key"],
                    },
                )
            return self._execution_mapping(
                cur,
                run_id,
                row["job_id"],
                row["idempotency_key"],
                row["status"] not in {"RESERVED", "unknown"},
            )

    def run_resolve(self, identifier: str) -> dict:
        """Resolve a canonical execution mapping from either run ID or local job ID."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT run_id,job_id,idempotency_key FROM research_execution_attempts "
                "WHERE run_id::text=%s OR job_id=%s",
                (identifier, identifier),
            )
            row = cur.fetchone()
            if row:
                return self._execution_mapping(
                    cur, row["run_id"], row["job_id"], row["idempotency_key"], True
                )
        # Backward compatibility for runs created before execution-attempt mappings existed.
        if re.fullmatch(r"[0-9a-fA-F-]{36}", identifier):
            run = self.object_get(identifier)
            if run["kind"] == "ExperimentRun" and run["data"].get("job_id"):
                return {
                    "experiment_id": run["data"].get("experiment_id"),
                    "run_id": str(run["id"]),
                    "job_id": run["data"]["job_id"],
                    "idempotency_key": run["data"].get("idempotency_key"),
                    "status": run["status"],
                    "run": run,
                    "legacy_mapping": True,
                }
        raise GPUError("EXPERIMENT_EXECUTION_NOT_FOUND", identifier)

    @staticmethod
    def _execution_mapping(cur, run_id, job_id, idempotency_key, idempotent_replay) -> dict:
        cur.execute(
            "SELECT id,project_id,kind,status,data,created_at FROM research_objects WHERE id=%s",
            (run_id,),
        )
        run = cur.fetchone()
        if not run:
            raise GPUError("EXPERIMENT_RUN_NOT_FOUND", str(run_id))
        return {
            "experiment_id": str(run["data"]["experiment_id"]),
            "run_id": str(run["id"]),
            "job_id": job_id,
            "idempotency_key": idempotency_key,
            "status": run["status"],
            "run": run,
            "idempotent_replay": idempotent_replay,
        }

    def run_update(self, run_id: str, result: dict) -> dict:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects "
                "WHERE id=%s FOR UPDATE",
                (run_id,),
            )
            run = cur.fetchone()
            if not run or run["kind"] != "ExperimentRun":
                raise GPUError("EXPERIMENT_RUN_NOT_FOUND", run_id)
            if run["status"] in {
                "completed", "failed", "cancelled", "RESULT_INSPECTED",
                "TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED",
            }:
                return {
                    "id": run_id,
                    "status": run["status"],
                    "data": run["data"],
                    "already_final": True,
                }
            data = {**run["data"], **result}
            status = result.get("status", run["status"])
            self._validate_status(status)
            if status == "completed":
                event = "EXPERIMENT_COMPLETED"
            elif status in {"failed", "cancelled"}:
                event = "EXPERIMENT_FAILED"
            elif status == "unknown":
                event = "EXPERIMENT_STATUS_UNKNOWN"
            else:
                event = "EXPERIMENT_PROGRESS_SYNCED"
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, json.dumps(data), run_id))
            cur.execute(
                "UPDATE research_execution_attempts SET status=%s,updated_at=%s WHERE run_id=%s",
                (status, datetime.now(UTC), run_id),
            )
            self._event(cur, run["project_id"], event, run_id, result)
        return {"id": run_id, "status": status, "data": data}

    def run_reconcile_orphan(
        self,
        run_id: str,
        job_id: str,
        terminal_status: str,
        runner_status: str,
        rationale: str,
    ) -> dict:
        """Terminally close a submitted run proved to be a non-scientific orphan.

        This is intentionally stricter than normal sync: it cannot discard a
        canonical artifact, a live job, or an unsubmitted reservation.
        """
        if terminal_status not in {"TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED"}:
            raise GPUError("INVALID_TECHNICAL_TERMINAL_STATUS", terminal_status)
        if runner_status not in {"cancelled", "missing", "unknown"}:
            raise GPUError("RUNNER_STATUS_NOT_ORPHANED", runner_status)
        if not rationale.strip():
            raise GPUError("TECHNICAL_RECONCILIATION_RATIONALE_REQUIRED", run_id)
        now = datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects WHERE id=%s FOR UPDATE",
                (run_id,),
            )
            run = cur.fetchone()
            if not run or run["kind"] != "ExperimentRun":
                raise GPUError("EXPERIMENT_RUN_NOT_FOUND", run_id)
            if run["data"].get("job_id") != job_id:
                raise GPUError("EXPERIMENT_RUN_JOB_MISMATCH", job_id)
            if not run["data"].get("submission_status"):
                raise GPUError("EXPERIMENT_RUN_NOT_SUBMITTED", run_id)
            if run["status"] in {"completed", "RESULT_INSPECTED"}:
                raise GPUError("EXPERIMENT_RUN_ALREADY_SCIENTIFICALLY_FINAL", run_id)
            if run["status"] in {"TECHNICAL_CANCELLED", "TECHNICAL_ORPHANED"}:
                return {"id": run_id, "status": run["status"], "data": run["data"], "idempotent_replay": True}
            cur.execute(
                "SELECT 1 FROM research_objects WHERE project_id=%s AND kind='Artifact' "
                "AND data->>'run_id'=%s LIMIT 1",
                (run["project_id"], run_id),
            )
            if cur.fetchone():
                raise GPUError("EXPERIMENT_RUN_ARTIFACT_PRESENT", run_id)
            inspection = {
                "mode": "TECHNICAL_ORPHAN_RECONCILIATION",
                "technical_non_scientific": True,
                "scientific_result": "NOT_ASSESSED",
                "prediction_outcome": "TECHNICAL_STALE_DUPLICATE_NO_RESULT",
                "actual_information_gain": "ZERO",
                "recorded_at": now.isoformat(),
                "rationale": rationale.strip(),
                "runner_status": runner_status,
            }
            data = {**run["data"], "inspection": inspection, "technical_reconciliation": inspection}
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (terminal_status, json.dumps(data), run_id))
            cur.execute("UPDATE research_execution_attempts SET status=%s,updated_at=%s WHERE run_id=%s", (terminal_status, now, run_id))
            self._event(cur, run["project_id"], "EXPERIMENT_RUN_TECHNICALLY_RECONCILED", run_id, {"status": terminal_status, "runner_status": runner_status, "rationale": rationale.strip()})
        return {"id": run_id, "status": terminal_status, "data": data, "idempotent_replay": False}

    def legacy_reserved_run_abandon(
        self, run_id: str, job_id: str, rationale: str, provenance: dict[str, Any]
    ) -> dict:
        """Terminally cancel one server-verified legacy reservation with no submitted job."""
        now = datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects WHERE id=%s FOR UPDATE",
                (run_id,),
            )
            run = cur.fetchone()
            if not run or run["kind"] != "ExperimentRun":
                raise GPUError("EXPERIMENT_RUN_NOT_FOUND", run_id)
            if run["status"] == "cancelled":
                abandoned = run["data"].get("legacy_abandonment", {})
                if abandoned.get("job_id") == job_id:
                    if run["data"].get("inspection") is None:
                        data = {
                            **run["data"],
                            "inspection": {
                                "mode": "EXECUTION_NOT_SUBMITTED",
                                "recorded_at": now.isoformat(),
                                "rationale": abandoned.get("rationale", rationale),
                                "limitations": [
                                    "This is operational bookkeeping, not a scientific result assessment."
                                ],
                            },
                        }
                        cur.execute(
                            "UPDATE research_objects SET data=%s WHERE id=%s",
                            (json.dumps(data), run_id),
                        )
                        self._event(
                            cur,
                            run["project_id"],
                            "EXPERIMENT_RUN_LEGACY_ABANDONMENT_RECORDED",
                            run_id,
                            {"job_id": job_id, "mode": "EXECUTION_NOT_SUBMITTED"},
                        )
                        return {
                            "id": run_id,
                            "status": "cancelled",
                            "data": data,
                            "idempotent_replay": True,
                        }
                    return {
                        "id": run_id,
                        "status": "cancelled",
                        "data": run["data"],
                        "idempotent_replay": True,
                    }
                raise GPUError("LEGACY_RUN_ALREADY_FINAL", run_id)
            if run["status"] != "RESERVED":
                raise GPUError("LEGACY_RUN_NOT_ABANDONABLE", run["status"])
            if run["data"].get("job_id") != job_id:
                raise GPUError("LEGACY_RUN_JOB_MISMATCH", job_id)
            if run["data"].get("submission_status"):
                raise GPUError("LEGACY_RUN_SUBMISSION_RECORDED", run_id)
            data = {
                **run["data"],
                "legacy_abandonment": {
                    "job_id": job_id,
                    "verified_missing_backing_job": True,
                    "rationale": rationale,
                    "provenance": provenance,
                    "abandoned_at": now.isoformat(),
                    "limitations": [
                        "No backing local job record existed at abandonment.",
                        "Cancellation is execution bookkeeping, not scientific evidence.",
                    ],
                },
                "inspection": {
                    "mode": "EXECUTION_NOT_SUBMITTED",
                    "recorded_at": now.isoformat(),
                    "rationale": rationale,
                    "limitations": [
                        "This is operational bookkeeping, not a scientific result assessment."
                    ],
                },
            }
            cur.execute(
                "UPDATE research_objects SET status='cancelled',data=%s WHERE id=%s",
                (json.dumps(data), run_id),
            )
            cur.execute(
                "UPDATE research_execution_attempts SET status='cancelled',updated_at=%s WHERE run_id=%s",
                (now, run_id),
            )
            self._event(
                cur,
                run["project_id"],
                "EXPERIMENT_RUN_LEGACY_ABANDONED",
                run_id,
                {"job_id": job_id, "rationale": rationale, "provenance": provenance},
            )
        return {"id": run_id, "status": "cancelled", "data": data, "idempotent_replay": False}

    def artifact_record(self, run_id: str, job_id: str, artifact: dict[str, Any]) -> dict:
        """Idempotently persist one run artifact and its provenance edge."""
        path = artifact.get("path")
        if not isinstance(path, str) or not path:
            raise GPUError("INVALID_ARTIFACT_PATH", "Artifact path is required")
        now = datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects WHERE id=%s FOR UPDATE",
                (run_id,),
            )
            run = cur.fetchone()
            if not run or run["kind"] != "ExperimentRun":
                raise GPUError("EXPERIMENT_RUN_NOT_FOUND", run_id)
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='Artifact' AND data->>'run_id'=%s "
                "AND data->>'path'=%s ORDER BY created_at LIMIT 1",
                (run["project_id"], run_id, path),
            )
            existing = cur.fetchone()
            if existing:
                return {**existing, "idempotent_replay": True}
            artifact_id = uuid.uuid4()
            data = {"run_id": run_id, "job_id": job_id, **artifact}
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'Artifact','VERIFIED_INTEGRATION',%s,%s)",
                (artifact_id, run["project_id"], json.dumps(data), now),
            )
            cur.execute(
                "INSERT INTO research_edges VALUES(%s,%s,'PRODUCED',%s) ON CONFLICT DO NOTHING",
                (run_id, artifact_id, now),
            )
            self._event(
                cur,
                run["project_id"],
                "ARTIFACT_RECORDED",
                artifact_id,
                data,
            )
            return {
                "id": str(artifact_id),
                "project_id": str(run["project_id"]),
                "kind": "Artifact",
                "status": "VERIFIED_INTEGRATION",
                "data": data,
                "created_at": now,
                "idempotent_replay": False,
            }

    def assess(self, object_id: str, status: str, rationale: str) -> dict:
        if status not in {"SUPPORTED", "WEAKENED", "REFUTED", "SURVIVES_INITIAL_TEST", "INCONCLUSIVE", "BLOCKED"}:
            raise GPUError("INVALID_SCIENTIFIC_STATUS", status)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data FROM research_objects "
                "WHERE id=%s FOR UPDATE",
                (object_id,),
            )
            item = cur.fetchone()
            if not item:
                raise GPUError("RESEARCH_OBJECT_NOT_FOUND", object_id)
            if item["kind"] == "Hypothesis" and status in {
                "SUPPORTED",
                "SURVIVES_INITIAL_TEST",
            }:
                required_edges = list(
                    dict.fromkeys(
                        [
                            *item["data"].get("required_edge_ids", []),
                            *item["data"].get("causal_edge_ids", []),
                        ]
                    )
                )
                if required_edges:
                    cur.execute(
                        "SELECT id FROM research_objects WHERE id=ANY(%s::uuid[]) "
                        "AND kind='CausalEdge' AND data->>'edge_status'='REFUTED' LIMIT 1",
                        (required_edges,),
                    )
                    if cur.fetchone():
                        raise GPUError(
                            "WORLD_MODEL_CONSISTENCY_ERROR",
                            "A positive hypothesis assessment requires no refuted causal edge",
                        )
                if not item["data"].get("supporting_evidence_family_ids"):
                    raise GPUError(
                        "SCIENTIFIC_PROMOTION_EVIDENCE_FAMILY_REQUIRED",
                        "Positive hypothesis assessment requires independent evidence provenance",
                    )
            event = f"{item['kind'].upper()}_{status}"
            data = {**item["data"], "assessment_rationale": rationale}
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, json.dumps(data), object_id))
            self._event(cur, item["project_id"], event, object_id, {"rationale": rationale})
        return {"id": object_id, "status": status, "rationale": rationale}

    def object_update(self, object_id: str, data_update: dict[str, Any], status: str, event_type: str) -> dict:
        """Materialize a new current view while preserving the scientific change as an event."""
        self._validate_status(status)
        item = self.object_get(object_id)
        data = {**item["data"], **data_update}
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, json.dumps(data), object_id))
            event_update = dict(data_update)
            if "logs_tail" in event_update:
                event_update["logs_tail"] = {"updated": True, "bytes": len(event_update["logs_tail"])}
            self._event(cur, item["project_id"], event_type, object_id, {"status": status, **event_update})
        return {"id": object_id, "status": status, "data": data}

    def events(
        self, project_id: str, limit: int = 100, as_of: datetime | str | None = None
    ) -> list[dict]:
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            sql = (
                "SELECT event_type,subject_id,payload,created_at FROM research_events "
                "WHERE project_id=%s"
            )
            args: list[Any] = [project_id]
            if cutoff is not None:
                sql += " AND committed_at<=%s"
                args.append(cutoff)
            sql += " ORDER BY created_at DESC LIMIT %s"
            args.append(limit)
            cur.execute(sql, args)
            return cur.fetchall()

    def search(
        self,
        project_id: str,
        query: str,
        kind: str | None = None,
        limit: int = 25,
        as_of: datetime | str | None = None,
    ) -> list[dict]:
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                sql = (
                    "SELECT id,kind,status,data,created_at FROM research_objects "
                    "WHERE project_id=%s"
                )
                args: list[Any] = [project_id]
            else:
                sql = (
                    "WITH latest AS ("
                    "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.kind,v.status,v.data,"
                    "v.object_created_at AS created_at FROM research_object_versions v "
                    "WHERE v.project_id=%s AND v.committed_at<=%s "
                    "ORDER BY v.object_id,v.committed_at DESC,"
                    "v.revision_id DESC) "
                    "SELECT id,kind,status,data,created_at FROM latest WHERE TRUE"
                )
                args = [project_id, cutoff]
            if kind:
                sql += " AND kind=%s"
                args.append(kind)
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += " AND data::text ILIKE %s ESCAPE '\\' ORDER BY created_at DESC LIMIT %s"
            args.extend((f"%{escaped}%", limit))
            cur.execute(sql, args)
            return cur.fetchall()

    def executable_paper_by_fingerprint(
        self, project_id: str, build_fingerprint: str
    ) -> dict | None:
        """Look up a paid build by exact identity without lossy text-search limits."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,project_id,kind,status,data,created_at FROM research_objects "
                "WHERE project_id=%s AND kind='ExecutablePaper' "
                "AND data->>'build_fingerprint'=%s ORDER BY created_at DESC LIMIT 1",
                (project_id, build_fingerprint),
            )
            return cur.fetchone()

    def related_hypotheses(
        self,
        project_id: str,
        mechanism: str,
        limit: int = 10,
        as_of: datetime | str | None = None,
    ) -> list[dict]:
        """Lexically screen active and failed ideas until pgvector is available for semantic ranking."""
        query_terms = self.terms(mechanism)
        candidates = [
            *self.objects_list(project_id, "Hypothesis", limit=None, as_of=as_of),
            *self.objects_list(project_id, "NegativeResult", limit=None, as_of=as_of),
        ]
        related = []
        for item in candidates:
            text = item["data"].get("mechanism") or item["data"].get("proposal", "")
            terms = self.terms(text)
            overlap = (
                len(query_terms & terms) / len(query_terms | terms) if query_terms | terms else 0.0
            )
            containment = len(query_terms & terms) / len(query_terms) if query_terms else 0.0
            if overlap:
                related.append(
                    {
                        **item,
                        "lexical_similarity": round(overlap, 3),
                        "containment_similarity": round(containment, 3),
                    }
                )
        return sorted(related, key=lambda item: item["lexical_similarity"], reverse=True)[:limit]

    def embedding_metadata_get(self, object_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT embedding_metadata FROM research_objects WHERE id=%s", (object_id,)
            )
            row = cur.fetchone()
        if not row:
            raise GPUError("RESEARCH_OBJECT_NOT_FOUND", object_id)
        return row["embedding_metadata"]

    def embedding_set(
        self,
        object_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        if not self.vector_available:
            raise GPUError("PGVECTOR_UNAVAILABLE", "The database does not have the vector extension")
        if not embedding or len(embedding) > 4096:
            raise GPUError("INVALID_EMBEDDING", "Embedding must contain 1 to 4096 numeric values")
        if not all(isinstance(value, (int, float)) for value in embedding):
            raise GPUError("INVALID_EMBEDDING", "Embedding values must be numeric")
        item = self.object_get(object_id)
        metadata = _json_document(metadata or {})
        if metadata and int(metadata.get("dimension", len(embedding))) != len(embedding):
            raise GPUError(
                "INVALID_EMBEDDING_METADATA", "Metadata dimension differs from the vector"
            )
        literal = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        now = datetime.now(UTC)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE research_objects SET embedding=%s::vector,embedding_updated_at=%s,"
                "embedding_metadata=%s "
                "WHERE id=%s",
                (literal, now, json.dumps(metadata), object_id),
            )
            self._event(
                cur,
                item["project_id"],
                "EMBEDDING_STORED",
                object_id,
                {
                    "dimensions": len(embedding),
                    "provider": metadata.get("provider"),
                    "model": metadata.get("model"),
                    "model_version": metadata.get("model_version"),
                    "source_text_hash": metadata.get("source_text_hash"),
                },
            )
        return {
            "id": object_id,
            "dimensions": len(embedding),
            "updated_at": now,
            "metadata": metadata,
        }

    def semantic_search(
        self,
        project_id: str,
        embedding: list[float],
        kind: str | None = None,
        limit: int = 25,
        as_of: datetime | str | None = None,
    ) -> list[dict]:
        if not self.vector_available:
            raise GPUError("PGVECTOR_UNAVAILABLE", "The database does not have the vector extension")
        if not embedding or len(embedding) > 4096 or not all(
            isinstance(value, (int, float)) for value in embedding
        ):
            raise GPUError("INVALID_EMBEDDING", "Query embedding must contain numeric values")
        literal = "[" + ",".join(str(float(value)) for value in embedding) + "]"
        cutoff = self._prepare_as_of(as_of)
        with self._connect() as conn, conn.cursor() as cur:
            if cutoff is None:
                sql = (
                    "SELECT id,kind,status,data,created_at,embedding <=> %s::vector AS distance "
                    "FROM research_objects WHERE project_id=%s AND embedding IS NOT NULL "
                    "AND vector_dims(embedding)=%s"
                )
                args: list[Any] = [literal, project_id, len(embedding)]
            else:
                sql = (
                    "WITH latest AS ("
                    "SELECT DISTINCT ON (v.object_id) v.object_id AS id,v.kind,v.status,v.data,"
                    "v.object_created_at AS created_at,v.embedding_text,v.embedding_updated_at "
                    "FROM research_object_versions v "
                    "WHERE v.project_id=%s AND v.committed_at<=%s "
                    "ORDER BY v.object_id,v.committed_at DESC,"
                    "v.revision_id DESC) "
                    "SELECT id,kind,status,data,created_at,"
                    "embedding_text::vector <=> %s::vector AS distance FROM latest "
                    "WHERE embedding_text IS NOT NULL AND embedding_updated_at<=%s "
                    "AND vector_dims(embedding_text::vector)=%s"
                )
                args = [project_id, cutoff, literal, cutoff, len(embedding)]
            if kind:
                sql += " AND kind=%s"
                args.append(kind)
            if cutoff is None:
                sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
                args.extend((literal, limit))
            else:
                sql += " ORDER BY embedding_text::vector <=> %s::vector LIMIT %s"
                args.extend((literal, limit))
            cur.execute(sql, args)
            return cur.fetchall()

    @staticmethod
    def terms(value: str) -> set[str]:
        return {term for term in re.findall(r"[a-z0-9]{4,}", value.lower())}

    @staticmethod
    def _validate_status(status: str) -> None:
        if status not in RESEARCH_OBJECT_STATUSES:
            raise GPUError("INVALID_RESEARCH_OBJECT_STATUS", status)

    @staticmethod
    def _canonical_uuid(value: str) -> str:
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise GPUError("INVALID_RESEARCH_OBJECT_ID", str(value)) from exc

    @staticmethod
    def _event(cur, project_id, event_type, subject_id, payload):
        cur.execute(
            "INSERT INTO research_events(id,project_id,event_type,subject_id,payload,created_at) "
            "VALUES(%s,%s,%s,%s,%s,%s)",
            (
                uuid.uuid4(),
                project_id,
                event_type,
                subject_id,
                json.dumps(payload),
                datetime.now(UTC),
            ),
        )
