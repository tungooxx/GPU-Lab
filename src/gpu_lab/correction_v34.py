"""Durable v3.4 correction workflow: challenge, verify, then adjudicate.

This module deliberately never changes the target scientific object.  A
CorrectionCase is an auditable evidence path; a later existing evidence or
experiment workflow remains responsible for changing scientific belief.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg

from .errors import GPUError
from .research import ResearchStore

CORRECTION_VERSION = "distributed-verification-correction-engine-v3.4"
CORRECTION_OPERATORS = {
    "CAUSAL_LOGIC", "STRONGEST_NULL", "EXPERIMENT_VALIDITY", "IMPLEMENTATION_VALIDITY",
    "EVIDENCE_SCOPE", "CONTRADICTION_SEARCH", "METRIC_VALIDITY", "REPLICATION_VALIDITY",
    "GENERALIZATION_VALIDITY", "LITERATURE_CONTRADICTION", "NOVELTY_VALIDITY",
    "WORLD_MODEL_CONSISTENCY", "COUNTERFACTUAL_VALIDITY", "ASSUMPTION_AUDIT",
    "DATA_PROVENANCE", "STATISTICAL_VALIDITY",
}
ISSUE_TYPES = {
    "CAUSAL_OVERREACH", "CORRELATION_AS_CAUSATION", "SCOPE_OVERREACH", "MISSING_NULL",
    "MISSING_CONTROL", "INVALID_INTERVENTION", "IMPLEMENTATION_CONFOUND", "METRIC_ARTIFACT",
    "EVIDENCE_CONTRADICTION", "EVIDENCE_DEPENDENCE", "INVALID_REPLICATION",
    "WORLD_MODEL_INCONSISTENCY", "LITERATURE_CONTRADICTION", "NOVELTY_OVERCLAIM",
    "GENERALIZATION_OVERCLAIM", "DATASET_ARTIFACT", "CHECKPOINT_DEPENDENCE",
    "EVALUATOR_DEPENDENCE", "UNJUSTIFIED_ARCHITECTURE_INFERENCE", "COUNTERFACTUAL_OVERCLAIM",
    "OTHER_STRUCTURED",
}
VERIFICATION_STATUSES = {"VERIFIED", "PARTIALLY_VERIFIED", "NOT_VERIFIED", "REFUTED_CHALLENGE", "INCONCLUSIVE", "INVALID_VERIFICATION"}
ADJUDICATIONS = {"KEEP", "REVISE", "REJECT", "NARROW_SCOPE", "EXPERIMENT_REQUIRED"}
ACTIVE_CASE_STATUSES = {"OPEN", "CRITIQUE_GENERATION", "CRITIQUES_FROZEN", "VERIFICATION", "ADJUDICATION", "NEEDS_EXPERIMENT", "INCONCLUSIVE"}


class DistributedCorrectionService:
    """Coordinates independent correction work without granting critics truth authority."""

    def __init__(self, store: ResearchStore, *, migrate: bool = True):
        self.store = store
        if migrate:
            self._migrate()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, default=str)

    @staticmethod
    def _hash(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    def _migrate(self) -> None:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_correction_v34_migration'))")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS correction_case_memberships (
                    correction_case_id UUID NOT NULL REFERENCES research_objects(id),
                    worker_id UUID NOT NULL, worker_session_id UUID NOT NULL,
                    correction_operator TEXT NOT NULL, status TEXT NOT NULL,
                    challenge_id UUID UNIQUE REFERENCES research_objects(id),
                    joined_at TIMESTAMPTZ NOT NULL, frozen_at TIMESTAMPTZ,
                    PRIMARY KEY(correction_case_id, worker_session_id)
                );
                CREATE INDEX IF NOT EXISTS correction_case_memberships_case_idx
                    ON correction_case_memberships(correction_case_id,status);
                CREATE TABLE IF NOT EXISTS correction_case_adjudications (
                    correction_case_id UUID PRIMARY KEY REFERENCES research_objects(id),
                    correction_record_id UUID NOT NULL UNIQUE REFERENCES research_objects(id),
                    created_at TIMESTAMPTZ NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS correction_case_active_authority_unique
                    ON research_objects(project_id,(data->>'authority_key'))
                    WHERE kind='CorrectionCase' AND status IN
                    ('OPEN','CRITIQUE_GENERATION','CRITIQUES_FROZEN','VERIFICATION','ADJUDICATION','NEEDS_EXPERIMENT','INCONCLUSIVE');
            """)

    def _case(self, case_id: str) -> dict:
        case = self.store.object_get(str(case_id))
        if case["kind"] != "CorrectionCase":
            raise GPUError("CORRECTION_CASE_NOT_FOUND", str(case_id))
        return case

    def _members(self, case_id: str) -> list[dict]:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM correction_case_memberships WHERE correction_case_id=%s ORDER BY joined_at", (case_id,))
            return [{key: str(value) if isinstance(value, uuid.UUID) else value for key, value in row.items()} for row in cur.fetchall()]

    def _session(self, cur, project_id: str, worker_id: str, session_id: str) -> None:
        cur.execute("SELECT id FROM research_worker_sessions WHERE id=%s AND worker_id=%s AND current_project_id=%s AND status NOT IN ('DISCONNECTED','EXPIRED')", (session_id, worker_id, project_id))
        if not cur.fetchone():
            raise GPUError("CORRECTION_WORKER_SESSION_INVALID", str(session_id))

    def create_case(self, project_id: str, target_id: str, opened_by: str | None = None,
                    purpose: str = "SCIENTIFIC_CORRECTION", severity: str = "MEDIUM") -> dict:
        target = self.store.object_get(str(target_id))
        if str(target["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", str(target_id))
        severity = severity.upper()
        if severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise GPUError("CORRECTION_SEVERITY_INVALID", severity)
        target_snapshot = {
            "id": str(target["id"]), "kind": target["kind"], "status": target["status"],
            "data": target["data"], "created_at": target["created_at"].isoformat(),
        }
        key = f"v34:{target['id']}:{self._hash(target_snapshot)}:{purpose.strip().upper()}"
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM research_objects WHERE project_id=%s AND kind='CorrectionCase' AND data->>'authority_key'=%s AND status=ANY(%s) ORDER BY created_at LIMIT 1", (project_id, key, list(ACTIVE_CASE_STATUSES)))
            existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["id"]))
        data = {
            "implementation_version": CORRECTION_VERSION, "authority_key": key,
            "target_type": target["kind"], "target_id": str(target["id"]),
            "target_status": target["status"], "target_version": self._hash(target_snapshot),
            "target_snapshot_hash": self._hash(target_snapshot), "target_snapshot": target_snapshot,
            "opened_by": opened_by, "purpose": purpose.strip().upper(), "severity": severity,
            "correction_stage": "INDEPENDENT_CRITIQUE", "peer_visibility": "HIDDEN",
            "opened_at": self._now().isoformat(), "agreement_is_not_evidence": True,
        }
        # The partial unique index is the authority under concurrent callers.
        # On collision return its canonical case instead of creating a second
        # competing correction process for the same frozen target/version.
        try:
            return self.store.object_create(project_id, "CorrectionCase", data, "CORRECTION_CASE_OPENED", "CRITIQUE_GENERATION")
        except psycopg.errors.UniqueViolation:
            with self.store._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT id FROM research_objects WHERE project_id=%s AND kind='CorrectionCase' AND data->>'authority_key'=%s AND status=ANY(%s) ORDER BY created_at LIMIT 1", (project_id, key, list(ACTIVE_CASE_STATUSES)))
                existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["id"]))
            raise

    def join_case(self, case_id: str, worker_id: str, session_id: str, correction_operator: str) -> dict:
        case = self._case(case_id)
        operator = correction_operator.upper()
        if operator not in CORRECTION_OPERATORS:
            raise GPUError("CORRECTION_OPERATOR_INVALID", operator)
        if case["status"] not in {"OPEN", "CRITIQUE_GENERATION"}:
            raise GPUError("CORRECTION_CASE_NOT_GENERATING", str(case_id))
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, str(case["project_id"]), worker_id, session_id)
            cur.execute("SELECT * FROM correction_case_memberships WHERE correction_case_id=%s AND worker_session_id=%s FOR UPDATE", (case_id, session_id))
            existing = cur.fetchone()
            if existing:
                return {key: str(value) if isinstance(value, uuid.UUID) else value for key, value in existing.items()}
            cur.execute("INSERT INTO correction_case_memberships(correction_case_id,worker_id,worker_session_id,correction_operator,status,joined_at) VALUES(%s,%s,%s,%s,'JOINED',%s)", (case_id, worker_id, session_id, operator, now))
            self.store._event(cur, case["project_id"], "CORRECTION_CRITIC_JOINED", str(case_id), {"operator": operator})
        return next(member for member in self._members(str(case_id)) if member["worker_session_id"] == str(session_id))

    def submit_challenge(self, case_id: str, worker_id: str, session_id: str, challenge: dict[str, Any]) -> dict:
        case = self._case(case_id)
        if case["status"] not in {"OPEN", "CRITIQUE_GENERATION"}:
            raise GPUError("CORRECTION_CASE_NOT_GENERATING", str(case_id))
        issue_type = str(challenge.get("issue_type", "")).upper()
        statement = str(challenge.get("issue_statement", "")).strip()
        if issue_type not in ISSUE_TYPES:
            raise GPUError(
                "CORRECTION_ISSUE_TYPE_INVALID",
                issue_type,
                details={"allowed_issue_types": sorted(ISSUE_TYPES)},
            )
        if not statement:
            raise GPUError("CORRECTION_ISSUE_STATEMENT_REQUIRED", str(case_id))
        evidence_refs = [str(item) for item in challenge.get("evidence_refs", []) if str(item)]
        reasoning_only = bool(challenge.get("reasoning_only", not evidence_refs))
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, str(case["project_id"]), worker_id, session_id)
            cur.execute("SELECT * FROM correction_case_memberships WHERE correction_case_id=%s AND worker_id=%s AND worker_session_id=%s FOR UPDATE", (case_id, worker_id, session_id))
            member = cur.fetchone()
            if not member or member["status"] != "JOINED":
                raise GPUError("CORRECTION_CHALLENGE_NOT_WRITABLE", str(case_id))
            if member["challenge_id"]:
                return self.store.object_get(str(member["challenge_id"]))
            challenge_id = str(uuid.uuid4())
            data = {
                "implementation_version": CORRECTION_VERSION, "correction_case_id": str(case_id),
                "critic_worker_id": str(worker_id), "critic_session_id": str(session_id),
                "correction_operator": member["correction_operator"], "target_component": str(challenge.get("target_component", "WHOLE_TARGET")),
                "issue_type": issue_type, "issue_statement": statement[:12000], "severity": str(challenge.get("severity", "MEDIUM")).upper(),
                "evidence_refs": evidence_refs, "missing_evidence": [str(item) for item in challenge.get("missing_evidence", [])],
                "proposed_counterexample": challenge.get("proposed_counterexample"), "proposed_null": challenge.get("proposed_null"),
                "proposed_discriminating_test": challenge.get("proposed_discriminating_test"),
                "reasoning_only": reasoning_only, "grounded": bool(evidence_refs), "confidence": challenge.get("confidence"),
                "created_at": now.isoformat(), "frozen_at": None,
            }
            cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'CorrectionChallenge','PROPOSED',%s,%s)", (challenge_id, case["project_id"], self._json(data), now))
            cur.execute("UPDATE correction_case_memberships SET challenge_id=%s WHERE correction_case_id=%s AND worker_session_id=%s", (challenge_id, case_id, session_id))
            self.store._event(cur, case["project_id"], "CORRECTION_CHALLENGE_SUBMITTED", challenge_id, {"case_id": str(case_id), "issue_type": issue_type, "reasoning_only": reasoning_only})
        return self.store.object_get(challenge_id)

    def challenge_get(self, case_id: str, challenge_id: str, requester_session_id: str) -> dict:
        case, challenge = self._case(case_id), self.store.object_get(str(challenge_id))
        if challenge["kind"] != "CorrectionChallenge" or str(challenge["data"].get("correction_case_id")) != str(case_id):
            raise GPUError("CORRECTION_CHALLENGE_NOT_FOUND", str(challenge_id))
        if case["data"].get("peer_visibility") == "HIDDEN" and str(challenge["data"].get("critic_session_id")) != str(requester_session_id):
            raise GPUError("CORRECTION_PEER_ISOLATION_ACTIVE", "Peer critiques are hidden until every critic freezes")
        return challenge

    def freeze_challenge(self, case_id: str, worker_id: str, session_id: str) -> dict:
        case = self._case(case_id)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            self._session(cur, str(case["project_id"]), worker_id, session_id)
            cur.execute("SELECT * FROM correction_case_memberships WHERE correction_case_id=%s AND worker_id=%s AND worker_session_id=%s FOR UPDATE", (case_id, worker_id, session_id))
            member = cur.fetchone()
            if not member or not member["challenge_id"]:
                raise GPUError("CORRECTION_CHALLENGE_REQUIRED", str(case_id))
            if member["status"] == "FROZEN":
                return self.store.object_get(str(member["challenge_id"]))
            cur.execute("UPDATE correction_case_memberships SET status='FROZEN',frozen_at=%s WHERE correction_case_id=%s AND worker_session_id=%s", (now, case_id, session_id))
            cur.execute("SELECT data FROM research_objects WHERE id=%s FOR UPDATE", (member["challenge_id"],))
            data = cur.fetchone()["data"]
            cur.execute("UPDATE research_objects SET status='FROZEN',data=%s WHERE id=%s", (self._json({**data, "frozen_at": now.isoformat()}), member["challenge_id"]))
            self.store._event(cur, case["project_id"], "CORRECTION_CHALLENGE_FROZEN", member["challenge_id"], {"case_id": str(case_id)})
            cur.execute("SELECT count(*) FILTER (WHERE status='JOINED') AS pending FROM correction_case_memberships WHERE correction_case_id=%s", (case_id,))
            if int(cur.fetchone()["pending"]) == 0:
                data = {**case["data"], "correction_stage": "VERIFICATION", "peer_visibility": "VISIBLE_FOR_VERIFICATION", "critiques_frozen_at": now.isoformat()}
                cur.execute("UPDATE research_objects SET status='VERIFICATION',data=%s WHERE id=%s", (self._json(data), case_id))
                self.store._event(cur, case["project_id"], "CORRECTION_CRITIQUES_FROZEN", str(case_id), {"peer_visibility": "VISIBLE_FOR_VERIFICATION"})
        return self.store.object_get(str(member["challenge_id"]))

    def verify(self, challenge_id: str, verifier_worker_id: str, verification: dict[str, Any]) -> dict:
        challenge = self.store.object_get(str(challenge_id))
        if challenge["kind"] != "CorrectionChallenge" or challenge["status"] != "FROZEN":
            raise GPUError("CORRECTION_CHALLENGE_NOT_VERIFIABLE", str(challenge_id))
        status = str(verification.get("verification_status", "")).upper()
        if status not in VERIFICATION_STATUSES:
            raise GPUError("CORRECTION_VERIFICATION_STATUS_INVALID", status)
        refs = [str(item) for item in verification.get("evidence_refs", []) if str(item)]
        checks = [str(item) for item in verification.get("deterministic_check_refs", []) if str(item)]
        if status in {"VERIFIED", "PARTIALLY_VERIFIED", "REFUTED_CHALLENGE"} and not (refs or checks):
            raise GPUError("CORRECTION_VERIFICATION_GROUNDING_REQUIRED", status)
        if str(challenge["data"].get("critic_worker_id")) == str(verifier_worker_id) and not checks:
            raise GPUError("CORRECTION_INDEPENDENT_VERIFIER_REQUIRED", "A critic may only self-check through a recorded deterministic check")
        for ref in refs + checks:
            evidence = self.store.object_get(ref)
            if str(evidence["project_id"]) != str(challenge["project_id"]):
                raise GPUError("CORRECTION_EVIDENCE_REF_PROJECT_MISMATCH", ref)
        data = {"implementation_version": CORRECTION_VERSION, "challenge_id": str(challenge_id), "verifier_worker_id": str(verifier_worker_id), "verification_operator": str(verification.get("verification_operator", "EVIDENCE_GROUNDING")).upper(), "evidence_refs": refs, "deterministic_check_refs": checks, "artifact_refs": [str(item) for item in verification.get("artifact_refs", [])], "verification_status": status, "verification_reason": str(verification.get("verification_reason", ""))[:12000], "created_at": self._now().isoformat()}
        result = self.store.object_create(str(challenge["project_id"]), "CorrectionVerification", data, "CORRECTION_CHALLENGE_VERIFIED" if status in {"VERIFIED", "PARTIALLY_VERIFIED"} else "CORRECTION_CHALLENGE_REFUTED" if status == "REFUTED_CHALLENGE" else "CORRECTION_VERIFICATION_RECORDED", "COMPLETED")
        return result

    def adjudicate(self, case_id: str, resolution: str, rationale: str, *, discriminating_test: str | None = None) -> dict:
        case = self._case(case_id)
        resolution = resolution.upper()
        if resolution not in ADJUDICATIONS or not rationale.strip():
            raise GPUError("CORRECTION_ADJUDICATION_INVALID", resolution)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"correction-adjudicate:{case_id}",))
            cur.execute("SELECT * FROM correction_case_adjudications WHERE correction_case_id=%s", (case_id,))
            existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["correction_record_id"]))
            target = self.store.object_get(str(case["data"]["target_id"]))
            target_snapshot = {
                "id": str(target["id"]), "kind": target["kind"], "status": target["status"],
                "data": target["data"], "created_at": target["created_at"].isoformat(),
            }
            if self._hash(target_snapshot) != case["data"].get("target_snapshot_hash"):
                cur.execute("UPDATE research_objects SET status='SUPERSEDED',data=%s WHERE id=%s", (self._json({**case["data"], "superseded_at": self._now().isoformat(), "superseded_by_target_version": self._hash(target_snapshot)}), case_id))
                self.store._event(cur, case["project_id"], "CORRECTION_CASE_SUPERSEDED", str(case_id), {"target_id": str(target["id"])})
                raise GPUError("CORRECTION_TARGET_SUPERSEDED", "Target changed after the correction case was frozen")
            cur.execute("SELECT id,data FROM research_objects WHERE project_id=%s AND kind='CorrectionChallenge' AND data->>'correction_case_id'=%s", (case["project_id"], str(case_id)))
            challenges = cur.fetchall()
            if not challenges or any(self.store.object_get(str(row["id"]))["status"] != "FROZEN" for row in challenges):
                raise GPUError("CORRECTION_ADJUDICATION_BEFORE_FREEZE", str(case_id))
            challenge_ids = [str(row["id"]) for row in challenges]
            verifications = self.store.objects_list(str(case["project_id"]), "CorrectionVerification", limit=None)
            linked = [item for item in verifications if str(item["data"].get("challenge_id")) in challenge_ids]
            grounded = [item for item in linked if item["data"].get("verification_status") in {"VERIFIED", "PARTIALLY_VERIFIED"}]
            if resolution in {"REVISE", "REJECT", "NARROW_SCOPE"} and not grounded:
                raise GPUError("CORRECTION_ADJUDICATION_EVIDENCE_REQUIRED", resolution)
            if resolution == "EXPERIMENT_REQUIRED" and not (discriminating_test or "").strip():
                raise GPUError("CORRECTION_DISCRIMINATING_TEST_REQUIRED", str(case_id))
            literature_issue = any(str(row["data"].get("issue_type")) in {"LITERATURE_CONTRADICTION", "NOVELTY_OVERCLAIM"} for row in challenges)
            if literature_issue and resolution in {"REVISE", "REJECT", "NARROW_SCOPE"}:
                literature_grounded = any(item["data"].get("literature_verified") or item["data"].get("literature_refs") for item in grounded)
                if not literature_grounded:
                    raise GPUError("CORRECTION_LITERATURE_GROUNDING_REQUIRED", "Novelty or literature critiques need independently verified literature grounding")
            disagreement_id = None
            if resolution == "EXPERIMENT_REQUIRED":
                disagreement = {"correction_case_id": str(case_id), "target_id": case["data"]["target_id"], "challenge_ids": challenge_ids, "predicted_separation": discriminating_test.strip(), "status": "PROPOSED", "not_evidence": True}
                disagreement_id = str(uuid.uuid4())
                cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'ScientificDisagreement','PROPOSED',%s,%s)", (disagreement_id, case["project_id"], self._json(disagreement), self._now()))
                self.store._event(cur, case["project_id"], "CORRECTION_EXPERIMENT_REQUIRED", disagreement_id, {"case_id": str(case_id)})
            record_id = str(uuid.uuid4())
            unique_evidence_refs = sorted({ref for item in grounded for ref in item["data"].get("evidence_refs", []) + item["data"].get("deterministic_check_refs", [])})
            record = {"implementation_version": CORRECTION_VERSION, "correction_case_id": str(case_id), "target_id": case["data"]["target_id"], "target_version": case["data"]["target_version"], "resolution": resolution, "rationale": rationale[:12000], "challenge_ids": challenge_ids, "verification_ids": [str(item["id"]) for item in linked], "grounded_verification_ids": [str(item["id"]) for item in grounded], "unique_evidence_refs": unique_evidence_refs, "agreement_is_not_evidence": True, "scientific_disagreement_id": disagreement_id, "target_mutated": False, "resolved_at": self._now().isoformat()}
            status = "NEEDS_EXPERIMENT" if resolution == "EXPERIMENT_REQUIRED" else f"RESOLVED_{resolution}"
            cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'CorrectionRecord','COMPLETED',%s,%s)", (record_id, case["project_id"], self._json(record), self._now()))
            data = {**case["data"], "correction_stage": "RESOLVED" if resolution != "EXPERIMENT_REQUIRED" else "EXPERIMENT_REQUIRED", "resolution": resolution, "resolution_reason": rationale[:12000], "resolved_at": self._now().isoformat(), "correction_record_id": record_id}
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, self._json(data), case_id))
            cur.execute("INSERT INTO correction_case_adjudications(correction_case_id,correction_record_id,created_at) VALUES(%s,%s,%s)", (case_id, record_id, self._now()))
            self.store._event(cur, case["project_id"], "CORRECTION_RESOLVED", str(case_id), {"resolution": resolution, "target_mutated": False})
        return self.store.object_get(record_id)

    def hindsight_record(self, correction_record_id: str, hindsight: dict[str, Any]) -> dict:
        record = self.store.object_get(str(correction_record_id))
        if record["kind"] != "CorrectionRecord":
            raise GPUError("CORRECTION_RECORD_NOT_FOUND", str(correction_record_id))
        data = {"correction_record_id": str(correction_record_id), "outcome": str(hindsight.get("outcome", "UNKNOWN")).upper(), "prevented_invalid_compute": bool(hindsight.get("prevented_invalid_compute", False)), "false_alarm": bool(hindsight.get("false_alarm", False)), "later_reversed": bool(hindsight.get("later_reversed", False)), "notes": str(hindsight.get("notes", ""))[:12000], "recorded_at": self._now().isoformat()}
        return self.store.object_create(str(record["project_id"]), "CorrectionHindsight", data, "CORRECTION_HINDSIGHT_UPDATED", "COMPLETED")

    def case_get(self, case_id: str, requester_session_id: str | None = None) -> dict:
        case = self._case(case_id)
        members = self._members(str(case_id))
        result = {**case, "members": members, "peer_challenges_visible": case["data"].get("peer_visibility") != "HIDDEN"}
        if requester_session_id and case["data"].get("peer_visibility") == "HIDDEN":
            for member in result["members"]:
                if member["worker_session_id"] != str(requester_session_id):
                    member.pop("challenge_id", None)
        return result

    def shadow_preview(self, project_id: str, target_id: str) -> dict:
        """Read-only correction planning; it proposes no fact and writes nothing."""
        target = self.store.object_get(str(target_id))
        if str(target["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", str(target_id))
        operators = (["CAUSAL_LOGIC", "STRONGEST_NULL", "EXPERIMENT_VALIDITY", "IMPLEMENTATION_VALIDITY"] if target["kind"] in {"Hypothesis", "Claim", "Mechanism", "CausalEdge"} else ["EVIDENCE_SCOPE", "METRIC_VALIDITY", "DATA_PROVENANCE", "WORLD_MODEL_CONSISTENCY"])
        return {"read_only": True, "implementation_version": CORRECTION_VERSION, "target": {"id": str(target["id"]), "kind": target["kind"], "status": target["status"]}, "target_snapshot_hash": self._hash({"kind": target["kind"], "status": target["status"], "data": target["data"]}), "independent_correction_operators": operators, "proposed_challenges": [], "evidence_grounding_required": True, "agreement_is_not_evidence": True, "executed_experiments": 0}
