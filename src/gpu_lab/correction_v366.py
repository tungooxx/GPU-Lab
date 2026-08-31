"""v3.6.6 scientific self-correction.

This is intentionally additive to v3.4 CorrectionCase.  It separates three
non-interchangeable questions: whether a scientific design is worth testing,
whether exact bytes implement an accepted design, and whether a result supports
a claim.  None of these records mutates a hypothesis or admits evidence.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from .errors import GPUError
from .research import ResearchStore

VERSION = "scientific-self-correction-v3.6.6"
DESIGN_RISKS = {"LOW", "MODERATE", "HIGH", "CONFIRMATORY"}
ENGINEERING_RISKS = {"DETERMINISTIC", "SCIENCE_CRITICAL", "HIGH_AMBIGUITY"}
ASSUMPTION_TYPES = {
    "NECESSITY", "SUFFICIENCY", "IDENTIFIABILITY", "REPRESENTATION", "CAUSALITY",
    "CONTROL_VALIDITY", "INVARIANCE", "GENERALIZATION", "MEASUREMENT", "OPTIMIZATION",
    "CAPACITY", "TEMPORAL",
}
CHALLENGE_TYPES = {
    "ALTERNATIVE_MECHANISM", "DEGENERATE_SOLUTION", "NECESSITY_CHALLENGE",
    "IDENTIFIABILITY_CHALLENGE", "CONTROL_FAILURE", "NULL_MISSING",
    "REPRESENTATION_CONFUSION", "CAPACITY_CONFOUND", "OPTIMIZATION_CONFOUND",
    "TEMPORAL_CONFOUND", "MEASUREMENT_INVALIDITY",
}
ATTACK_VERDICTS = {
    "SURVIVES_CURRENT_ATTACK", "MATERIAL_ASSUMPTION_CHALLENGED", "REVISE_DESIGN",
    "NARROW_CLAIM", "STRONG_NULL_MISSING", "NON_IDENTIFIABLE",
    "DEGENERATE_ALTERNATIVE_EXISTS", "DISCRIMINATING_EXPERIMENT_REQUIRED",
    "SUPERSEDE_DESIGN", "INSUFFICIENT_ATTACK_COVERAGE",
}
SYNTHESIS_VERDICTS = {
    "RETAIN_FOR_TEST", "RETAIN_WITH_NARROWER_CLAIM", "REVISE_BEFORE_TEST",
    "DISCRIMINATING_EXPERIMENT_REQUIRED", "SUPERSEDE", "INSUFFICIENT_CORRECTION",
}
FIDELITY_VERDICTS = {
    "FIDELITY_CONFIRMED", "FIDELITY_VIOLATION", "SCIENTIFIC_DRIFT_DETECTED",
    "DETERMINISTIC_TEST_MISSING", "INCONCLUSIVE",
}
INTERPRETATION_VERDICTS = {
    "SURVIVES_CURRENT_ATTACK", "MECHANISM_NONSPECIFIC", "NARROW_CLAIM",
    "IMPLEMENTATION_VALIDITY_REQUIRED", "INSUFFICIENT_EVIDENCE", "INCONCLUSIVE",
}
INTERPRETATION_OPERATORS = {
    "STRONGEST_NULL", "CAUSAL_IDENTIFIABILITY", "INTERVENTION_VALIDITY",
    "GENERALIZATION_SCOPE", "FAILURE_CLASSIFICATION", "MEASUREMENT_VALIDITY",
}


class ScientificSelfCorrectionService:
    """Coordinates independently submitted idea attacks without majority voting."""

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

    @staticmethod
    def _normal(value: str) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _record(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        return {key: str(value) if isinstance(value, uuid.UUID) else value for key, value in row.items()}

    def _migrate(self) -> None:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_correction_v366_migration'))")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS design_attack_batches (
                    id UUID PRIMARY KEY, project_id UUID NOT NULL REFERENCES research_projects(id),
                    design_id UUID NOT NULL REFERENCES research_objects(id),
                    design_snapshot_hash TEXT NOT NULL, status TEXT NOT NULL,
                    peer_visibility TEXT NOT NULL, required_assumption_ids JSONB NOT NULL,
                    frozen_lineage_context JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL, frozen_at TIMESTAMPTZ
                );
                ALTER TABLE design_attack_batches
                    ADD COLUMN IF NOT EXISTS frozen_lineage_context JSONB NOT NULL DEFAULT '{}'::jsonb;
                CREATE TABLE IF NOT EXISTS design_attack_memberships (
                    batch_id UUID NOT NULL REFERENCES design_attack_batches(id),
                    worker_id UUID NOT NULL, worker_session_id UUID NOT NULL,
                    attack_operator TEXT NOT NULL, attack_id UUID UNIQUE REFERENCES research_objects(id),
                    status TEXT NOT NULL, joined_at TIMESTAMPTZ NOT NULL, frozen_at TIMESTAMPTZ,
                    PRIMARY KEY(batch_id, worker_session_id)
                );
                CREATE INDEX IF NOT EXISTS design_attack_memberships_batch_idx
                    ON design_attack_memberships(batch_id,status);
                CREATE TABLE IF NOT EXISTS design_attack_syntheses (
                    batch_id UUID PRIMARY KEY REFERENCES design_attack_batches(id),
                    synthesis_id UUID NOT NULL UNIQUE REFERENCES research_objects(id), created_at TIMESTAMPTZ NOT NULL
                );
            """)

    def _object(self, ident: str, kind: str) -> dict:
        value = self.store.object_get(str(ident))
        if value["kind"] != kind:
            raise GPUError(f"{kind.upper()}_NOT_FOUND", str(ident))
        return value

    @staticmethod
    def _attack_intensity(risk: str) -> dict[str, Any]:
        """Return the consequence-proportionate independence requirement.

        This is deliberately a minimum coverage rule, not a voting rule.  A
        material challenge still blocks acceptance even when every required
        independent attack has been collected.
        """
        return {
            "LOW": {"minimum_independent_attacks": 1, "operators": ["NECESSITY_ATTACKER"]},
            "MODERATE": {"minimum_independent_attacks": 1, "operators": ["STRONGEST_NULL_ATTACKER"]},
            "HIGH": {"minimum_independent_attacks": 3, "operators": ["NECESSITY_ATTACKER", "CAUSAL_IDENTIFIABILITY_ATTACKER", "DEGENERATE_SOLUTION_ATTACKER"]},
            "CONFIRMATORY": {"minimum_independent_attacks": 3, "operators": ["STRONGEST_NULL_ATTACKER", "CAUSAL_IDENTIFIABILITY_ATTACKER", "MEASUREMENT_ATTACKER"]},
        }[risk]

    def _session(self, cur, project_id: str, worker_id: str, session_id: str) -> None:
        cur.execute(
            "SELECT id FROM research_worker_sessions WHERE id=%s AND worker_id=%s "
            "AND current_project_id=%s AND status NOT IN ('DISCONNECTED','EXPIRED')",
            (session_id, worker_id, project_id),
        )
        if not cur.fetchone():
            raise GPUError("DESIGN_ATTACK_WORKER_SESSION_INVALID", str(session_id))

    def _frozen_lineage_context(self, project_id: str, source_object_id: str | None = None) -> dict[str, Any]:
        """Capture current design-relevant history without treating technical failure as science.

        The context is an immutable list of durable references, not a verdict
        on the design.  It makes post-design NegativeResults visible to the
        critics while explicitly separating records that identify themselves
        as technical-only or not scientifically assessed.
        """
        scientific_negative_result_ids: list[str] = []
        technical_only_negative_result_ids: list[str] = []
        for item in self.store.objects_list(project_id, "NegativeResult", limit=None):
            data = item.get("data") or {}
            marker = " ".join(str(data.get(key) or "") for key in (
                "scientific_result", "classification", "failure_class", "result_type", "validity",
            )).upper()
            if any(token in marker for token in ("TECHNICAL", "NOT_ASSESSED", "INVALID_EXECUTION")):
                technical_only_negative_result_ids.append(str(item["id"]))
            else:
                scientific_negative_result_ids.append(str(item["id"]))
        return {
            "captured_at": self._now().isoformat(),
            "source_object_id": str(source_object_id) if source_object_id else None,
            "scientific_negative_result_ids": sorted(scientific_negative_result_ids),
            "technical_only_negative_result_ids": sorted(technical_only_negative_result_ids),
            "technical_failures_are_not_scientific_design_evidence": True,
        }

    def create_design(self, project_id: str, name: str, specification: dict[str, Any], *,
                      source_object_id: str | None = None, scientific_risk: str = "MODERATE",
                      engineering_risk: str = "DETERMINISTIC") -> dict:
        scientific_risk, engineering_risk = self._normal(scientific_risk), self._normal(engineering_risk)
        if scientific_risk not in DESIGN_RISKS:
            raise GPUError("SCIENTIFIC_DESIGN_RISK_INVALID", scientific_risk)
        if engineering_risk not in ENGINEERING_RISKS:
            raise GPUError("ENGINEERING_VERIFICATION_RISK_INVALID", engineering_risk)
        if not name.strip() or not specification:
            raise GPUError("SCIENTIFIC_DESIGN_SPEC_REQUIRED", "name and specification are required")
        source = None
        if source_object_id:
            source = self.store.object_get(str(source_object_id))
            if str(source["project_id"]) != str(project_id):
                raise GPUError("RESEARCH_PROJECT_MISMATCH", str(source_object_id))
        frozen = {"name": name.strip(), "specification": specification, "source_object_id": str(source_object_id) if source else None}
        data = {
            "implementation_version": VERSION, "name": name.strip(), "specification": specification,
            "source_object_id": str(source_object_id) if source else None,
            "scientific_design_risk": scientific_risk, "engineering_verification_risk": engineering_risk,
            "semantic_hash": self._hash(frozen), "design_status": "FROZEN", "authority": "NOT_ACCEPTED_FOR_TEST",
            "design_self_correction_status": "INCOMPLETE", "created_at": self._now().isoformat(),
            "original_lineage_context": self._frozen_lineage_context(project_id, source_object_id),
        }
        return self.store.object_create(project_id, "ScientificDesign", data, "SCIENTIFIC_DESIGN_FROZEN", "FROZEN")

    def assumption_create(self, design_id: str, statement: str, assumption_type: str, *,
                          centrality: str = "MATERIAL", falsifiable_prediction: str | None = None,
                          provenance: dict[str, Any] | None = None) -> dict:
        design = self._object(design_id, "ScientificDesign")
        assumption_type = self._normal(assumption_type)
        if assumption_type not in ASSUMPTION_TYPES or not statement.strip():
            raise GPUError("DESIGN_ASSUMPTION_INVALID", assumption_type)
        if self._normal(centrality) not in {"PERIPHERAL", "MATERIAL", "CENTRAL"}:
            raise GPUError("DESIGN_ASSUMPTION_CENTRALITY_INVALID", str(centrality))
        data = {
            "implementation_version": VERSION, "scientific_design_id": str(design_id),
            "scientific_design_version": design["data"]["semantic_hash"], "statement": statement.strip(),
            "assumption_type": assumption_type, "centrality": self._normal(centrality),
            "falsifiable_prediction": falsifiable_prediction, "known_support": [], "known_challenges": [],
            "status": "OPEN", "provenance": provenance or {},
        }
        return self.store.object_create(str(design["project_id"]), "DesignAssumption", data, "DESIGN_ASSUMPTION_CREATED", "OPEN")

    def attack_batch_create(self, design_id: str, assumption_ids: list[str] | None = None) -> dict:
        design = self._object(design_id, "ScientificDesign")
        if design["status"] not in {"FROZEN", "UNDER_ATTACK"}:
            raise GPUError("SCIENTIFIC_DESIGN_NOT_ATTACKABLE", str(design_id))
        assumptions = self.store.objects_list(str(design["project_id"]), "DesignAssumption", limit=None)
        design_assumptions = [item for item in assumptions if str(item["data"].get("scientific_design_id")) == str(design_id)]
        required = [str(item) for item in (assumption_ids or [x["id"] for x in design_assumptions])]
        known = {str(item["id"]) for item in design_assumptions}
        if not required or not set(required).issubset(known):
            raise GPUError("DESIGN_ATTACK_ASSUMPTIONS_REQUIRED", "Attack only frozen assumptions of this design")
        risk = str(design["data"].get("scientific_design_risk") or "MODERATE").upper()
        intensity = self._attack_intensity(risk)
        now, batch_id = self._now(), uuid.uuid4()
        snapshot_hash = design["data"]["semantic_hash"]
        lineage_context = self._frozen_lineage_context(str(design["project_id"]), design["data"].get("source_object_id"))
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM design_attack_batches WHERE design_id=%s AND design_snapshot_hash=%s AND status='GENERATING' FOR UPDATE", (design_id, snapshot_hash))
            existing = cur.fetchone()
            if existing:
                return self.attack_batch_get(str(existing["id"]))
            cur.execute("INSERT INTO design_attack_batches(id,project_id,design_id,design_snapshot_hash,status,peer_visibility,required_assumption_ids,frozen_lineage_context,created_at) VALUES(%s,%s,%s,%s,'GENERATING','HIDDEN',%s,%s,%s)", (batch_id, design["project_id"], design_id, snapshot_hash, self._json(required), self._json(lineage_context), now))
            self.store._event(cur, design["project_id"], "DESIGN_ATTACK_INTENSITY_SELECTED", design_id, {"scientific_design_risk": risk, **intensity})
            cur.execute("UPDATE research_objects SET status='UNDER_ATTACK',data=%s WHERE id=%s", (self._json({**design["data"], "design_status": "UNDER_ATTACK"}), design_id))
            self.store._event(cur, design["project_id"], "DESIGN_ATTACK_STARTED", design_id, {"batch_id": str(batch_id), "required_assumption_ids": required, "scientific_negative_result_count": len(lineage_context["scientific_negative_result_ids"]), "technical_negative_result_count": len(lineage_context["technical_only_negative_result_ids"])})
        result = self.attack_batch_get(str(batch_id))
        result["scientific_design_risk"] = risk
        result["recommended_attack_intensity"] = intensity
        return result

    def attack_batch_get(self, batch_id: str, requester_session_id: str | None = None) -> dict:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM design_attack_batches WHERE id=%s", (batch_id,))
            batch = cur.fetchone()
            if not batch:
                raise GPUError("DESIGN_ATTACK_BATCH_NOT_FOUND", str(batch_id))
            cur.execute("SELECT * FROM design_attack_memberships WHERE batch_id=%s ORDER BY joined_at", (batch_id,))
            members = cur.fetchall()
        result = {key: str(value) if isinstance(value, uuid.UUID) else value for key, value in batch.items()}
        result["members"] = []
        for member in members:
            item = {key: str(value) if isinstance(value, uuid.UUID) else value for key, value in member.items()}
            if result["peer_visibility"] == "HIDDEN" and str(item["worker_session_id"]) != str(requester_session_id):
                item.pop("attack_id", None)
            result["members"].append(item)
        result["peer_attacks_visible"] = result["peer_visibility"] != "HIDDEN"
        design = self._object(str(result["design_id"]), "ScientificDesign")
        risk = str(design["data"].get("scientific_design_risk") or "MODERATE").upper()
        result["scientific_design_risk"] = risk
        result["recommended_attack_intensity"] = self._attack_intensity(risk)
        return result

    def attack_join(self, batch_id: str, worker_id: str, session_id: str, attack_operator: str) -> dict:
        operator, now = self._normal(attack_operator), self._now()
        if not operator:
            raise GPUError("DESIGN_ATTACK_OPERATOR_REQUIRED", "operator")
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM design_attack_batches WHERE id=%s FOR UPDATE", (batch_id,))
            batch = cur.fetchone()
            if not batch or batch["status"] != "GENERATING":
                raise GPUError("DESIGN_ATTACK_BATCH_NOT_GENERATING", str(batch_id))
            self._session(cur, str(batch["project_id"]), worker_id, session_id)
            cur.execute("SELECT * FROM design_attack_memberships WHERE batch_id=%s AND worker_session_id=%s FOR UPDATE", (batch_id, session_id))
            member = cur.fetchone()
            if member:
                return {key: str(value) if isinstance(value, uuid.UUID) else value for key, value in member.items()}
            # A new browser/session for the same worker is not an independent
            # scientific critic.  Keeping this invariant at join time prevents
            # a session-restart from inflating high-risk attack coverage.
            cur.execute("SELECT worker_session_id FROM design_attack_memberships WHERE batch_id=%s AND worker_id=%s FOR UPDATE", (batch_id, worker_id))
            existing_worker = cur.fetchone()
            if existing_worker:
                raise GPUError("DESIGN_ATTACK_WORKER_ALREADY_JOINED", str(worker_id))
            design = self._object(str(batch["design_id"]), "ScientificDesign")
            attack_id = uuid.uuid4()
            data = {
                "implementation_version": VERSION, "design_attack_batch_id": str(batch_id), "target_design_id": str(batch["design_id"]),
                "target_design_version": batch["design_snapshot_hash"], "target_semantic_hash": batch["design_snapshot_hash"],
                "target_assumption_ids": list(batch["required_assumption_ids"]), "attacker_worker_id": str(worker_id),
                "attacker_session_id": str(session_id), "attack_operator": operator, "peer_exposure_state": "HIDDEN",
                "frozen_context_version": batch["design_snapshot_hash"], "design_snapshot": design["data"],
                "frozen_lineage_context": batch["frozen_lineage_context"],
                "challenges": [], "competing_explanations": [], "degenerate_explanations": [], "discriminating_tests": [],
                "verdict": None, "independence": {"peer_attack_visible_before_submission": False, "shared_message_visible": False},
            }
            cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'ScientificDesignAttack','UNDER_ATTACK',%s,%s)", (attack_id, batch["project_id"], self._json(data), now))
            cur.execute("INSERT INTO design_attack_memberships(batch_id,worker_id,worker_session_id,attack_operator,attack_id,status,joined_at) VALUES(%s,%s,%s,%s,%s,'JOINED',%s)", (batch_id, worker_id, session_id, operator, attack_id, now))
            self.store._event(cur, batch["project_id"], "DESIGN_ATTACK_STARTED", attack_id, {"batch_id": str(batch_id), "operator": operator})
        return self.store.object_get(str(attack_id))

    def attack_submit(self, batch_id: str, worker_id: str, session_id: str, *, verdict: str,
                      challenges: list[dict[str, Any]], competing_explanations: list[dict[str, Any]] | None = None,
                      degenerate_explanations: list[dict[str, Any]] | None = None,
                      discriminating_tests: list[dict[str, Any]] | None = None,
                      considered_assumption_ids: list[str] | None = None) -> dict:
        verdict = self._normal(verdict)
        if verdict not in ATTACK_VERDICTS:
            raise GPUError("DESIGN_ATTACK_VERDICT_INVALID", verdict)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT b.*,m.attack_id,m.status AS member_status FROM design_attack_batches b JOIN design_attack_memberships m ON m.batch_id=b.id WHERE b.id=%s AND m.worker_id=%s AND m.worker_session_id=%s FOR UPDATE", (batch_id, worker_id, session_id))
            member = cur.fetchone()
            if not member or member["status"] != "GENERATING" or member["member_status"] != "JOINED":
                raise GPUError("DESIGN_ATTACK_NOT_WRITABLE", str(batch_id))
            self._session(cur, str(member["project_id"]), worker_id, session_id)
            required = {str(item) for item in member["required_assumption_ids"]}
            considered = {str(item) for item in (considered_assumption_ids or required) if str(item)}
            if not considered or not considered.issubset(required):
                raise GPUError("DESIGN_ATTACK_ASSUMPTION_COVERAGE_INVALID", "considered assumptions must belong to the frozen batch")
            challenge_ids: list[str] = []
            for challenge in challenges:
                ctype = self._normal(challenge.get("challenge_type"))
                statement = str(challenge.get("statement") or "").strip()
                assumption_id = str(challenge.get("target_assumption_id") or "")
                if ctype not in CHALLENGE_TYPES or not statement or (assumption_id and assumption_id not in required):
                    raise GPUError("DESIGN_CHALLENGE_INVALID", ctype or "missing type/statement/assumption")
                challenge_id = uuid.uuid4()
                data = {"implementation_version": VERSION, "design_attack_id": str(member["attack_id"]), "target_assumption_id": assumption_id or None,
                        "challenge_type": ctype, "statement": statement[:12000], "competing_explanation": challenge.get("competing_explanation"),
                        "why_current_controls_fail": challenge.get("why_current_controls_fail"), "distinguishing_prediction": challenge.get("distinguishing_prediction"),
                        "proposed_discriminating_test": challenge.get("proposed_discriminating_test"), "severity": self._normal(challenge.get("severity", "MEDIUM")),
                        "verification_status": "UNRESOLVED", "resolution": None}
                cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'DesignChallenge','OPEN',%s,%s)", (challenge_id, member["project_id"], self._json(data), self._now()))
                challenge_ids.append(str(challenge_id))
                self.store._event(cur, member["project_id"], "DESIGN_CHALLENGE_CREATED", challenge_id, {"attack_id": str(member["attack_id"]), "challenge_type": ctype})
            cur.execute("SELECT data FROM research_objects WHERE id=%s FOR UPDATE", (member["attack_id"],))
            attack = cur.fetchone()["data"]
            update = {**attack, "challenge_ids": challenge_ids, "challenges": challenges, "competing_explanations": competing_explanations or [],
                      "degenerate_explanations": degenerate_explanations or [], "discriminating_tests": discriminating_tests or [], "verdict": verdict,
                      "considered_assumption_ids": sorted(considered),
                      "submitted_at": self._now().isoformat()}
            cur.execute("UPDATE research_objects SET data=%s WHERE id=%s", (self._json(update), member["attack_id"]))
        return self.store.object_get(str(member["attack_id"]))

    def attack_freeze(self, batch_id: str, worker_id: str, session_id: str) -> dict:
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT b.*,m.attack_id,m.status AS member_status FROM design_attack_batches b JOIN design_attack_memberships m ON m.batch_id=b.id WHERE b.id=%s AND m.worker_id=%s AND m.worker_session_id=%s FOR UPDATE", (batch_id, worker_id, session_id))
            member = cur.fetchone()
            if not member or member["member_status"] not in {"JOINED", "FROZEN"}:
                raise GPUError("DESIGN_ATTACK_NOT_JOINED", str(batch_id))
            self._session(cur, str(member["project_id"]), worker_id, session_id)
            attack = self.store.object_get(str(member["attack_id"]))
            if not attack["data"].get("verdict"):
                raise GPUError("DESIGN_ATTACK_SUBMISSION_REQUIRED", str(batch_id))
            if member["member_status"] == "FROZEN":
                return attack
            cur.execute("UPDATE design_attack_memberships SET status='FROZEN',frozen_at=%s WHERE batch_id=%s AND worker_session_id=%s", (now, batch_id, session_id))
            cur.execute("UPDATE research_objects SET status='FROZEN',data=%s WHERE id=%s", (self._json({**attack["data"], "frozen_at": now.isoformat()}), member["attack_id"]))
            cur.execute("SELECT count(*) FILTER (WHERE status='JOINED') AS pending FROM design_attack_memberships WHERE batch_id=%s", (batch_id,))
            if int(cur.fetchone()["pending"]) == 0:
                cur.execute("UPDATE design_attack_batches SET status='FROZEN',peer_visibility='VISIBLE',frozen_at=%s WHERE id=%s", (now, batch_id))
                self.store._event(cur, member["project_id"], "DESIGN_ATTACK_COMMITTED", member["design_id"], {"batch_id": str(batch_id)})
        return self.store.object_get(str(member["attack_id"]))

    def legacy_design_attack_extract(
        self, project_id: str, source_review_id: str, scientific_design_id: str,
        evidence_summary: str, considered_assumption_ids: list[str],
        challenges: list[dict[str, Any]] | None = None,
    ) -> dict:
        """Backfill a documented historical idea attack without inventing independence.

        This deliberately creates a frozen *partial* ScientificDesignAttack.
        It carries the original review/work identifier and its supplied evidence
        summary, but has no worker/session membership and can never inflate a
        live independent-batch quorum or authorize a synthesis by itself.
        """
        design = self._object(scientific_design_id, "ScientificDesign")
        if str(design["project_id"]) != str(project_id) or not evidence_summary.strip():
            raise GPUError("LEGACY_DESIGN_ATTACK_EXTRACTION_INVALID", "project, design, and evidence_summary are required")
        assumptions = [
            item for item in self.store.objects_list(project_id, "DesignAssumption", limit=None)
            if str(item["data"].get("scientific_design_id")) == str(scientific_design_id)
        ]
        known = {str(item["id"]) for item in assumptions}
        considered = {str(item) for item in considered_assumption_ids if str(item)}
        if not considered or not considered.issubset(known):
            raise GPUError("LEGACY_DESIGN_ATTACK_ASSUMPTION_COVERAGE_INVALID", "considered assumptions must belong to the design")
        supplied_challenges = challenges or []
        now = self._now()
        key = self._hash({"project_id": str(project_id), "source_review_id": str(source_review_id), "design_id": str(scientific_design_id)})
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"v366-legacy-design-attack:{key}",))
            cur.execute("SELECT kind FROM research_objects WHERE id=%s AND project_id=%s", (source_review_id, project_id))
            source = cur.fetchone()
            source_kind = str(source["kind"]) if source else None
            if not source:
                cur.execute("SELECT kind FROM lab_work_items WHERE id=%s AND project_id=%s", (source_review_id, project_id))
                source = cur.fetchone()
                source_kind = str(source["kind"]) if source else None
            if not source:
                raise GPUError("LEGACY_DESIGN_ATTACK_SOURCE_NOT_FOUND", str(source_review_id))
            cur.execute(
                "SELECT id FROM research_objects WHERE project_id=%s AND kind='ScientificDesignAttack' "
                "AND data->>'legacy_extraction_key'=%s FOR UPDATE",
                (project_id, key),
            )
            existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["id"]))
            attack_id = uuid.uuid4()
            challenge_ids: list[str] = []
            for challenge in supplied_challenges:
                ctype = self._normal(challenge.get("challenge_type"))
                statement = str(challenge.get("statement") or "").strip()
                assumption_id = str(challenge.get("target_assumption_id") or "")
                if ctype not in CHALLENGE_TYPES or not statement or (assumption_id and assumption_id not in considered):
                    raise GPUError("LEGACY_DESIGN_CHALLENGE_INVALID", ctype or "missing type/statement/assumption")
                challenge_id = uuid.uuid4()
                payload = {
                    "implementation_version": VERSION, "design_attack_id": str(attack_id),
                    "target_assumption_id": assumption_id or None, "challenge_type": ctype,
                    "statement": statement[:12000], "competing_explanation": challenge.get("competing_explanation"),
                    "why_current_controls_fail": challenge.get("why_current_controls_fail"),
                    "distinguishing_prediction": challenge.get("distinguishing_prediction"),
                    "proposed_discriminating_test": challenge.get("proposed_discriminating_test"),
                    "severity": self._normal(challenge.get("severity", "MEDIUM")),
                    "verification_status": "UNRESOLVED", "resolution": None,
                    "legacy_extracted_from": str(source_review_id),
                    "does_not_claim_new_independent_discovery": True,
                }
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'DesignChallenge','OPEN',%s,%s)",
                    (challenge_id, project_id, self._json(payload), now),
                )
                challenge_ids.append(str(challenge_id))
            data = {
                "implementation_version": VERSION, "legacy_extraction_key": key,
                "target_design_id": str(scientific_design_id),
                "target_design_version": design["data"]["semantic_hash"],
                "target_assumption_ids": sorted(considered), "considered_assumption_ids": sorted(considered),
                "source_review_id": str(source_review_id), "source_kind": source_kind,
                "historical_evidence_summary": evidence_summary.strip()[:12000],
                "challenge_ids": challenge_ids, "challenges": supplied_challenges,
                "verdict": "INSUFFICIENT_ATTACK_COVERAGE", "peer_exposure_state": "HISTORICAL_BACKFILL",
                "frozen_context_version": design["data"]["semantic_hash"],
                "independence": {
                    "historical_extraction": True, "attacker_identity": "UNATTRIBUTED_OR_PRESERVED_IN_SOURCE",
                    "does_not_count_toward_live_independent_batch_quorum": True,
                    "does_not_claim_peer_isolation": True,
                },
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ScientificDesignAttack','FROZEN',%s,%s)",
                (attack_id, project_id, self._json(data), now),
            )
            self.store._event(cur, project_id, "LEGACY_DESIGN_ATTACK_EXTRACTED", attack_id, {
                "source_review_id": str(source_review_id), "source_kind": source_kind,
                "design_id": str(scientific_design_id), "challenge_count": len(challenge_ids),
                "does_not_claim_independence": True,
            })
        return self.store.object_get(str(attack_id))

    def _materialize_transition_work(
        self, cur, batch: dict, design: dict, synthesis_id: str, status: str,
    ) -> tuple[dict[str, Any], bool]:
        """Create exactly one bounded follow-up for an attack synthesis.

        A synthesis has a deterministic operational consequence, but critics
        must not invent arbitrary adjacent science.  The resulting card is
        therefore keyed by the immutable synthesis identity and contains only
        the frozen design/synthesis references.  It is supporting reasoning,
        never execution authority.
        """
        accepted = status == "ACCEPTED_FOR_TEST"
        kind = "PRE_SCIENCE_DESIGN_FINALIZATION" if accepted else "SCIENTIFIC_DESIGN_REPAIR"
        role = "DESIGN_FINALIZER" if accepted else "SCIENTIFIC_DESIGN_REPAIR"
        action = "Finalize the accepted pre-science design binding" if accepted else "Repair the scientific design before any test"
        key = f"v366-design-transition:{synthesis_id}:{kind}"
        cur.execute(
            "SELECT * FROM lab_work_items WHERE project_id=%s AND equivalence_key=%s "
            "AND status=ANY(%s) FOR UPDATE",
            (
                batch["project_id"], key,
                ["DORMANT", "READY", "CLAIMED", "RUNNING", "RUNNING_DETACHED", "RESULT_READY", "WAITING_DEPENDENCY", "BLOCKED"],
            ),
        )
        existing = cur.fetchone()
        if existing:
            return self._record(existing), True
        now, work_id = self._now(), uuid.uuid4()
        refs = {
            "correction_layer": "DESIGN",
            "scientific_design_id": str(batch["design_id"]),
            "design_attack_batch_id": str(batch["id"]),
            "design_attack_synthesis_id": str(synthesis_id),
            "design_version": batch["design_snapshot_hash"],
            "deterministic_successor": True,
            "no_execution_authority": True,
            "transition": "DESIGN_ACCEPTED_FOR_TEST" if accepted else "DESIGN_REPAIR_REQUIRED",
        }
        cur.execute(
            "INSERT INTO lab_work_items("
            "id,project_id,kind,title,description,scientific_role,status,related_refs,equivalence_key,"
            "authority_status,subject_id,recovery_policy,speculation_class,speculation_condition,resource_class,"
            "dependency_scope,authority_class,promotion_state,created_at,updated_at"
            ") VALUES(%s,%s,%s,%s,%s,%s,'READY',%s,%s,'SUPPORTING',%s,'{}'::jsonb,"
            "'NON_SPECULATIVE','{}'::jsonb,'REASONING','WORKITEM_LOCAL','SUPPORTING','NOT_APPLICABLE',%s,%s) "
            "RETURNING *",
            (
                work_id, batch["project_id"], kind,
                f"{action}: {design['data'].get('name', 'Scientific design')}",
                "Deterministic successor of a frozen v3.6.6 design-attack synthesis. "
                "It may bind or repair the design, but cannot itself authorize execution.",
                role, self._json(refs), key, str(batch["design_id"]), now, now,
            ),
        )
        work = self._record(cur.fetchone())
        self.store._event(cur, batch["project_id"], "WORK_ITEM_SUCCESSOR_MATERIALIZED", str(work_id), {
            "source": "DESIGN_ATTACK_SYNTHESIS", "synthesis_id": str(synthesis_id),
            "design_id": str(batch["design_id"]), "transition": refs["transition"],
        })
        return work, False

    def synthesize(self, batch_id: str, synthesis_verdict: str, rationale: str) -> dict:
        verdict = self._normal(synthesis_verdict)
        if verdict not in SYNTHESIS_VERDICTS or not rationale.strip():
            raise GPUError("DESIGN_ATTACK_SYNTHESIS_INVALID", verdict)
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM design_attack_batches WHERE id=%s FOR UPDATE", (batch_id,))
            batch = cur.fetchone()
            if not batch or batch["status"] != "FROZEN":
                raise GPUError("DESIGN_ATTACK_SYNTHESIS_BEFORE_FREEZE", str(batch_id))
            cur.execute("SELECT synthesis_id FROM design_attack_syntheses WHERE batch_id=%s", (batch_id,))
            existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["synthesis_id"]))
            cur.execute("SELECT attack_id FROM design_attack_memberships WHERE batch_id=%s", (batch_id,))
            attacks = [self.store.object_get(str(row["attack_id"])) for row in cur.fetchall()]
            design = self._object(str(batch["design_id"]), "ScientificDesign")
            risk = str(design["data"].get("scientific_design_risk") or "MODERATE").upper()
            intensity = self._attack_intensity(risk)
            independent_workers = {str(attack["data"].get("attacker_worker_id")) for attack in attacks}
            distinct_operators = {str(attack["data"].get("attack_operator")) for attack in attacks}
            accepting = verdict in {"RETAIN_FOR_TEST", "RETAIN_WITH_NARROWER_CLAIM"}
            challenge_ids = [cid for attack in attacks for cid in attack["data"].get("challenge_ids", [])]
            challenges = [self.store.object_get(cid) for cid in challenge_ids]
            unique: dict[str, str] = {}
            material: list[str] = []
            challenged = set()
            for challenge in challenges:
                data = challenge["data"]
                key = self._hash({"type": data["challenge_type"], "assumption": data.get("target_assumption_id"), "statement": data["statement"].casefold(), "alternative": data.get("competing_explanation")})
                unique.setdefault(key, str(challenge["id"]))
                if data.get("target_assumption_id"):
                    challenged.add(str(data["target_assumption_id"]))
                if data.get("severity") in {"HIGH", "CRITICAL"} and data.get("verification_status") != "RESOLVED":
                    material.append(str(challenge["id"]))
            # No vote: one unresolved material challenge prevents acceptance.
            if verdict in {"RETAIN_FOR_TEST", "RETAIN_WITH_NARROWER_CLAIM"} and material:
                raise GPUError("DESIGN_ATTACK_MATERIAL_CHALLENGE_UNRESOLVED", "A material challenge cannot be outweighed by attack count")
            if accepting and len(independent_workers) < intensity["minimum_independent_attacks"]:
                raise GPUError(
                    "DESIGN_ATTACK_INDEPENDENT_COVERAGE_INCOMPLETE",
                    f"{risk} requires {intensity['minimum_independent_attacks']} independent workers; found {len(independent_workers)}",
                )
            if accepting and len(distinct_operators) < min(intensity["minimum_independent_attacks"], len(intensity["operators"])):
                raise GPUError(
                    "DESIGN_ATTACK_OPERATOR_DIVERSITY_INCOMPLETE",
                    f"{risk} requires orthogonal attack operators; found {len(distinct_operators)}",
                )
            required = {str(value) for value in batch["required_assumption_ids"]}
            # Coverage means an assumption was deliberately considered by an
            # isolated critic, whether that critic found a flaw or concluded
            # it currently survives.  Challenge count is not coverage.
            covered = {
                str(item)
                for attack in attacks
                for item in attack["data"].get("considered_assumption_ids", attack["data"].get("target_assumption_ids", []))
            }
            missing = sorted(required - covered)
            if verdict in {"RETAIN_FOR_TEST", "RETAIN_WITH_NARROWER_CLAIM"} and missing:
                raise GPUError("DESIGN_ATTACK_COVERAGE_INCOMPLETE", ",".join(missing))
            discriminator_ids = []
            for challenge in challenges:
                proposed = challenge["data"].get("proposed_discriminating_test")
                if proposed:
                    result = self.store.object_create(str(batch["project_id"]), "RequiredDiscriminator", {"design_id": str(batch["design_id"]), "challenge_id": str(challenge["id"]), "proposal": proposed, "status": "PROPOSED", "not_automatically_materialized": True}, "DISCRIMINATING_EXPERIMENT_REQUIRED", "PROPOSED")
                    discriminator_ids.append(result["id"])
            data = {"implementation_version": VERSION, "design_attack_batch_id": str(batch_id), "design_id": str(batch["design_id"]), "design_version": batch["design_snapshot_hash"], "attack_ids": [str(x["id"]) for x in attacks], "unique_challenge_ids": list(unique.values()), "duplicate_challenge_ids": [str(x["id"]) for x in challenges if str(x["id"]) not in unique.values()], "challenged_assumption_ids": sorted(challenged), "surviving_assumption_ids": sorted(required - challenged), "unresolved_material_challenge_ids": material, "missing_assumption_attack_ids": missing, "discriminating_experiment_ids": discriminator_ids, "synthesis_verdict": verdict, "rationale": rationale[:12000], "majority_vote_used": False}
            status = "ACCEPTED_FOR_TEST" if verdict in {"RETAIN_FOR_TEST", "RETAIN_WITH_NARROWER_CLAIM"} else "REPAIR_REQUIRED"
            synthesis = self.store.object_create(str(batch["project_id"]), "DesignAttackSynthesis", data, "DESIGN_ATTACK_SYNTHESIZED", status)
            transition_work, transition_reused = self._materialize_transition_work(
                cur, batch, design, synthesis["id"], status,
            )
            data["transition_work_item_id"] = str(transition_work["id"])
            data["transition_work_reused"] = transition_reused
            cur.execute("UPDATE research_objects SET data=%s WHERE id=%s", (self._json(data), synthesis["id"]))
            synthesis["data"] = data
            design_data = {**design["data"], "design_self_correction_status": "COMPLETE" if status == "ACCEPTED_FOR_TEST" else "REPAIR_REQUIRED", "design_status": status, "authority": "PASS_FOR_TEST" if status == "ACCEPTED_FOR_TEST" else "REPAIR_REQUIRED", "design_attack_synthesis_id": synthesis["id"], "transition_work_item_id": str(transition_work["id"])}
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, self._json(design_data), batch["design_id"]))
            cur.execute("INSERT INTO design_attack_syntheses(batch_id,synthesis_id,created_at) VALUES(%s,%s,%s)", (batch_id, synthesis["id"], self._now()))
            self.store._event(cur, batch["project_id"], "DESIGN_ACCEPTED_FOR_TEST" if status == "ACCEPTED_FOR_TEST" else "DESIGN_REPAIR_REQUIRED", batch["design_id"], {"synthesis_id": synthesis["id"], "verdict": verdict, "transition_work_item_id": str(transition_work["id"]), "transition_work_reused": transition_reused})
            synthesis["transition_work_item"] = transition_work
        return synthesis

    def fidelity_record(self, design_id: str, *, implementation_ref: str, science_to_code_map: dict[str, Any],
                        deterministic_checks: list[str], verdict: str, rationale: str) -> dict:
        design = self._object(design_id, "ScientificDesign")
        verdict = self._normal(verdict)
        if verdict not in FIDELITY_VERDICTS or not implementation_ref or not rationale.strip():
            raise GPUError("IMPLEMENTATION_FIDELITY_INVALID", verdict)
        data = {"implementation_version": VERSION, "scientific_design_id": str(design_id), "design_version": design["data"]["semantic_hash"], "implementation_ref": implementation_ref, "science_to_code_map": science_to_code_map, "deterministic_checks": deterministic_checks, "verdict": verdict, "rationale": rationale[:12000], "does_not_satisfy_design_attack": True}
        status = "FIDELITY_CONFIRMED" if verdict == "FIDELITY_CONFIRMED" else "INCONCLUSIVE"
        return self.store.object_create(str(design["project_id"]), "ImplementationFidelityCheck", data, "IMPLEMENTATION_FIDELITY_CONFIRMED" if status == "FIDELITY_CONFIRMED" else "IMPLEMENTATION_FIDELITY_RECORDED", status)

    def interpretation_attack_record(self, project_id: str, result_id: str, *, claim: str, verdict: str,
                                     strongest_alternative: str, rationale: str,
                                     attack_operator: str = "STRONGEST_NULL",
                                     reviewer_worker_id: str | None = None,
                                     reviewer_session_id: str | None = None) -> dict:
        """Persist one scoped challenge to an inspected result's inference.

        This never changes the result, admits evidence, or reviews code.  The
        same result/claim/operator is idempotent, while high-consequence work
        can still receive independent attacks under distinct operators.
        """
        result = self.store.object_get(str(result_id))
        verdict = self._normal(verdict)
        operator = self._normal(attack_operator)
        if str(result["project_id"]) != str(project_id) or verdict not in INTERPRETATION_VERDICTS or operator not in INTERPRETATION_OPERATORS or not claim.strip() or not strongest_alternative.strip() or not rationale.strip():
            raise GPUError("RESULT_INTERPRETATION_ATTACK_INVALID", verdict)
        if bool(reviewer_worker_id) != bool(reviewer_session_id):
            raise GPUError("RESULT_INTERPRETATION_REVIEWER_IDENTITY_REQUIRED", "worker_id and session_id must be supplied together")
        key = self._hash({
            "result_id": str(result_id), "claim": claim.strip().casefold(), "operator": operator,
        })
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"result-interpretation:{project_id}:{key}",))
            if reviewer_worker_id and reviewer_session_id:
                self._session(cur, project_id, str(reviewer_worker_id), str(reviewer_session_id))
            cur.execute(
                "SELECT id FROM research_objects WHERE project_id=%s AND kind='ResultInterpretationAttack' "
                "AND data->>'dedupe_key'=%s ORDER BY created_at LIMIT 1",
                (project_id, key),
            )
            existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["id"]))
            data = {
                "implementation_version": VERSION, "result_id": str(result_id),
                "claim": claim[:12000], "verdict": verdict,
                "strongest_alternative": strongest_alternative[:12000],
                "rationale": rationale[:12000], "attack_operator": operator,
                "dedupe_key": key,
                "reviewer_worker_id": str(reviewer_worker_id) if reviewer_worker_id else None,
                "reviewer_session_id": str(reviewer_session_id) if reviewer_session_id else None,
                "reviewer_identity_status": "VERIFIED" if reviewer_worker_id else "UNATTRIBUTED_LEGACY_COMPATIBILITY",
                "target_layer": "SCIENTIFIC_INTERPRETATION",
                "implementation_validity_is_held_separate": True,
                "does_not_admit_evidence": True,
            }
            self.store._event(cur, project_id, "RESULT_INTERPRETATION_ATTACK_STARTED", str(result_id), {
                "operator": operator, "dedupe_key": key,
            })
            attack_id, now = uuid.uuid4(), self._now()
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ResultInterpretationAttack','COMPLETED',%s,%s)",
                (attack_id, project_id, self._json(data), now),
            )
            self.store._event(cur, project_id, "RESULT_INTERPRETATION_ATTACK_COMMITTED", attack_id, data)
        return {
            "id": str(attack_id), "project_id": str(project_id),
            "kind": "ResultInterpretationAttack", "status": "COMPLETED", "data": data,
        }

    def challenge_hindsight_record(self, design_challenge_id: str, *, outcome: str,
                                   notes: str, evidence_refs: list[str] | None = None) -> dict:
        """Record whether later evidence validated an earlier idea-level challenge.

        The original frozen attack remains immutable.  Hindsight is a separate
        record so later results cannot be smuggled back into the critic's
        original independent submission.
        """
        challenge = self._object(design_challenge_id, "DesignChallenge")
        outcome = self._normal(outcome)
        if outcome not in {"VALIDATED", "PARTIALLY_VALIDATED", "FALSE_ALARM", "OVERBROAD", "STILL_UNRESOLVED"} or not notes.strip():
            raise GPUError("DESIGN_CHALLENGE_HINDSIGHT_INVALID", outcome)
        refs = [str(ref) for ref in (evidence_refs or []) if str(ref)]
        for ref in refs:
            item = self.store.object_get(ref)
            if str(item["project_id"]) != str(challenge["project_id"]):
                raise GPUError("DESIGN_CHALLENGE_HINDSIGHT_PROJECT_MISMATCH", ref)
        data = {
            "implementation_version": VERSION, "design_challenge_id": str(design_challenge_id),
            "outcome": outcome, "notes": notes[:12000], "evidence_refs": refs,
            "recorded_at": self._now().isoformat(), "does_not_rewrite_frozen_attack": True,
        }
        return self.store.object_create(str(challenge["project_id"]), "DesignChallengeHindsight", data,
                                        "DESIGN_CHALLENGE_HINDSIGHT_RECORDED", "COMPLETED")

    def regression_proposal_create(self, project_id: str, source_id: str, *, test_reference: str,
                                   invariant: str, rationale: str) -> dict:
        """Turn a repeatable engineering finding into a deterministic test proposal."""
        source = self.store.object_get(str(source_id))
        if str(source["project_id"]) != str(project_id) or source["kind"] not in {"ImplementationFidelityCheck", "EngineeringResult"}:
            raise GPUError("REVIEW_REGRESSION_SOURCE_INVALID", str(source_id))
        if not test_reference.strip() or not invariant.strip() or not rationale.strip():
            raise GPUError("REVIEW_REGRESSION_PROPOSAL_FIELDS_REQUIRED", "test_reference, invariant, rationale")
        data = {
            "implementation_version": VERSION, "source_id": str(source_id), "source_kind": source["kind"],
            "test_reference": test_reference.strip(), "invariant": invariant.strip(), "rationale": rationale[:12000],
            "status": "PROPOSED", "scientific_result": "NOT_ASSESSED",
        }
        return self.store.object_create(project_id, "DeterministicRegressionProposal", data,
                                        "REVIEW_FINDING_CONVERTED_TO_REGRESSION", "PROPOSED")

    def design_coverage(self, design_id: str) -> dict:
        design = self._object(design_id, "ScientificDesign")
        assumptions = [item for item in self.store.objects_list(str(design["project_id"]), "DesignAssumption", limit=None) if str(item["data"].get("scientific_design_id")) == str(design_id)]
        syntheses = [item for item in self.store.objects_list(str(design["project_id"]), "DesignAttackSynthesis", limit=None) if str(item["data"].get("design_id")) == str(design_id)]
        latest = syntheses[-1] if syntheses else None
        # Coverage is useful before synthesis as well: a frozen partial attack
        # must be visible as partial rather than appearing as no correction at
        # all.  Syntheses retain the authoritative accepted/repaired outcome.
        attacks = [
            item for item in self.store.objects_list(str(design["project_id"]), "ScientificDesignAttack", limit=None)
            if str(item["data"].get("target_design_id")) == str(design_id)
            and item["status"] in {"FROZEN", "COMPLETED"}
        ]
        covered = {
            str(assumption_id)
            for attack in attacks
            for assumption_id in attack["data"].get("considered_assumption_ids", attack["data"].get("target_assumption_ids", []))
        }
        if latest:
            covered.update(latest["data"].get("challenged_assumption_ids", []))
            covered.update(latest["data"].get("surviving_assumption_ids", []))
        all_ids = {str(item["id"]) for item in assumptions}
        return {"design_id": str(design_id), "design_status": design["data"].get("design_status"), "assumption_count": len(assumptions), "covered_assumption_ids": sorted(covered), "missing_assumption_ids": sorted(all_ids - covered), "synthesis": latest, "implementation_fidelity_is_not_design_correction": True}

    @staticmethod
    def _classify_review_target(kind: str, data: dict[str, Any]) -> str:
        text = " ".join(str(data.get(key, "")) for key in ("title", "description", "task_type", "purpose", "rationale", "scientific_role")).lower()
        if kind in {"ScientificDesignAttack", "DesignChallenge", "DesignAttackSynthesis"} or any(word in text for word in ("mechanism", "identifiability", "competing explanation", "design attack")):
            label = "DESIGN_ATTACK"
        elif kind in {"ImplementationFidelityCheck", "EngineeringResult"} or "science-to-code" in text or "fidelity" in text:
            label = "IMPLEMENTATION_FIDELITY"
        elif kind == "ResultInterpretationAttack" or "interpretation" in text or "claim" in text:
            label = "RESULT_INTERPRETATION"
        elif any(word in text for word in ("format", "path", "logging", "shell", "lint", "hash")):
            label = "DETERMINISTIC_ENGINEERING"
        else:
            label = "UNKNOWN"
        return label

    def review_target_classifier(self, project_id: str, target_id: str) -> dict:
        target = self.store.object_get(str(target_id))
        if str(target["project_id"]) != str(project_id):
            raise GPUError("RESEARCH_PROJECT_MISMATCH", str(target_id))
        label = self._classify_review_target(target["kind"], target["data"])
        return {"read_only": True, "implementation_version": VERSION, "target_id": str(target_id), "classification": label, "canonical_history_mutated": False}

    @staticmethod
    def _migration_label(classification: str) -> str:
        return {
            "DESIGN_ATTACK": "DESIGN",
            "IMPLEMENTATION_FIDELITY": "IMPLEMENTATION",
            "RESULT_INTERPRETATION": "INTERPRETATION",
            "DETERMINISTIC_ENGINEERING": "MECHANICAL",
        }.get(classification, "UNKNOWN")

    def review_target_classification_persist(self, project_id: str, target_id: str) -> dict:
        """Persist an inferred migration label without rewriting the source record.

        The original object or WorkItem remains untouched. This produces one
        durable BACKFILL record per target so dashboards can distinguish a
        historical inference from a contemporaneous independent review.
        """
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT id,kind,data FROM research_objects WHERE id=%s AND project_id=%s", (target_id, project_id))
            object_row = cur.fetchone()
            target_type = "RESEARCH_OBJECT"
            if object_row:
                kind, payload = str(object_row["kind"]), dict(object_row["data"] or {})
            else:
                cur.execute(
                    "SELECT id,kind,title,description,scientific_role,related_refs FROM lab_work_items "
                    "WHERE id=%s AND project_id=%s",
                    (target_id, project_id),
                )
                work_row = cur.fetchone()
                if not work_row:
                    raise GPUError("REVIEW_TARGET_NOT_FOUND", str(target_id))
                target_type, kind = "WORK_ITEM", str(work_row["kind"])
                payload = {
                    "title": work_row["title"], "description": work_row["description"],
                    "scientific_role": work_row["scientific_role"], **(work_row["related_refs"] or {}),
                }
            raw = self._classify_review_target(kind, payload)
            migration_label = self._migration_label(raw)
            confidence = "MEDIUM" if raw != "UNKNOWN" else "LOW"
            key = self._hash({"project_id": str(project_id), "target_type": target_type, "target_id": str(target_id)})
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"v366-review-classification:{key}",))
            cur.execute(
                "SELECT id FROM research_objects WHERE project_id=%s AND kind='ReviewTargetClassification' "
                "AND data->>'classification_key'=%s ORDER BY created_at LIMIT 1",
                (project_id, key),
            )
            existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["id"]))
            ident, now = uuid.uuid4(), self._now()
            data = {
                "implementation_version": VERSION, "classification_key": key,
                "target_id": str(target_id), "target_type": target_type,
                "source_kind": kind, "inferred_classification": migration_label,
                "classifier_label": raw, "confidence": confidence,
                "provenance": "BACKFILL_INFERRED_CLASSIFICATION",
                "preserves_original_artifact": True,
                "does_not_claim_independent_worker_separation": True,
                "classified_at": now.isoformat(),
            }
            cur.execute(
                "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                "VALUES(%s,%s,'ReviewTargetClassification','COMPLETED',%s,%s)",
                (ident, project_id, self._json(data), now),
            )
            self.store._event(cur, project_id, "REVIEW_TARGET_CLASSIFICATION_BACKFILLED", ident, data)
        return {"id": str(ident), "project_id": str(project_id), "kind": "ReviewTargetClassification", "status": "COMPLETED", "data": data}

    @staticmethod
    def _is_v5_design(design: dict) -> bool:
        data = design.get("data") or {}
        text = " ".join((
            str(data.get("name", "")), str(data.get("specification", "")),
            str(data.get("source_object_id", "")),
        )).casefold()
        return "v5" in text or "gru-256" in text or "learned g" in text

    def v5_missing_attack_work_preview(self, project_id: str) -> dict:
        """Return, but never create, the one bounded V5 attack successor."""
        focus = [
            "recurrence necessity", "256D rationale", "predictive-state identifiability",
            "learned G causal role", "static plus binding alternative",
            "transition-memory alternative", "short-history alternative",
            "time-permutation validity",
        ]
        designs = [
            design for design in self.store.objects_list(project_id, "ScientificDesign", limit=None)
            if self._is_v5_design(design)
            and design["status"] not in {"SUPERSEDED", "INVALID", "CANCELLED", "FAILED"}
        ]
        proposals = []
        for design in designs:
            coverage = self.design_coverage(str(design["id"]))
            if not coverage["missing_assumption_ids"]:
                continue
            version = str(design["data"].get("semantic_hash") or "unknown")
            proposals.append({
                "kind": "V5_SCIENTIFIC_DESIGN_ATTACK",
                "title": f"V5 bounded scientific design attack: {design['data'].get('name', 'V5')}",
                "scientific_role": "SCIENTIFIC_DESIGN_ATTACKER",
                "description": "One bounded attack lane covering all currently missing V5 design assumptions; not a code review or execution authorization.",
                "equivalence_key": f"v366-v5-design-attack:{design['id']}:{version}",
                "related_refs": {
                    "correction_layer": "DESIGN", "scientific_design_id": str(design["id"]),
                    "design_version": version, "required_focus": focus,
                    "no_execution_authority": True,
                },
                "missing_assumption_ids": coverage["missing_assumption_ids"],
                "materialization_requires": "authorized planner/work-proposal transition",
            })
        return {
            "read_only": True, "implementation_version": VERSION,
            "v5_design_count": len(designs), "bounded_work_proposals": proposals,
            "mutated": False,
        }

    def interpretation_attack_work_preview(self, project_id: str) -> dict:
        """Preview one bounded inference-attack card per inspected result."""
        attacked = {
            str(item["data"].get("result_id"))
            for item in self.store.objects_list(project_id, "ResultInterpretationAttack", limit=None)
        }
        proposals = []
        for run in self.store.objects_list(project_id, "ExperimentRun", limit=None):
            if run["status"] != "RESULT_INSPECTED" or str(run["id"]) in attacked:
                continue
            proposals.append({
                "kind": "RESULT_INTERPRETATION_ATTACK", "scientific_role": "RESULT_INTERPRETATION_ATTACK",
                "title": f"Interpretation attack: inspected result {run['id']}",
                "equivalence_key": f"v366-interpretation-attack:{run['id']}",
                "related_refs": {"correction_layer": "INTERPRETATION", "result_id": str(run["id"]), "no_execution_authority": True},
                "materialization_requires": "authorized planner/work-proposal transition",
            })
        return {"read_only": True, "implementation_version": VERSION, "bounded_work_proposals": proposals, "mutated": False}

    def production_audit(self, project_id: str) -> dict:
        """Read-only v3.6.6 audit; historical records are never rewritten."""
        kinds = ("EngineeringTask", "EngineeringResult", "ScientificDesignAttack",
                 "ImplementationFidelityCheck", "ResultInterpretationAttack", "CorrectionCase")
        records = [item for kind in kinds for item in self.store.objects_list(project_id, kind, limit=None)]
        classifications: dict[str, list[str]] = {
            "DESIGN_ATTACK": [], "IMPLEMENTATION_FIDELITY": [], "RESULT_INTERPRETATION": [],
            "DETERMINISTIC_ENGINEERING": [], "MIXED": [], "UNKNOWN": [],
        }
        for record in records:
            label = self._classify_review_target(record["kind"], record["data"])
            classifications.setdefault(label, []).append(str(record["id"]))
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,kind,status,title,description,scientific_role,related_refs FROM lab_work_items "
                "WHERE project_id=%s AND (kind ILIKE '%%REVIEW%%' OR scientific_role ILIKE '%%REVIEW%%' "
                "OR scientific_role ILIKE '%%ATTACK%%' OR scientific_role ILIKE '%%FIDELITY%%')",
                (project_id,),
            )
            work_items = cur.fetchall()
        for item in work_items:
            payload = {
                "title": item["title"], "description": item["description"],
                "scientific_role": item["scientific_role"], **(item["related_refs"] or {}),
            }
            label = self._classify_review_target(str(item["kind"]), payload)
            classifications.setdefault(label, []).append(f"work:{item['id']}")
        designs = self.store.objects_list(project_id, "ScientificDesign", limit=None)
        persisted = self.store.objects_list(project_id, "ReviewTargetClassification", limit=None)
        assumptions = self.store.objects_list(project_id, "DesignAssumption", limit=None)
        runs = self.store.objects_list(project_id, "ExperimentRun", limit=None)
        interpretation_attacks = self.store.objects_list(project_id, "ResultInterpretationAttack", limit=None)
        regression_proposals = self.store.objects_list(project_id, "DeterministicRegressionProposal", limit=None)
        coverage = [self.design_coverage(str(design["id"])) for design in designs]
        incomplete = [entry["design_id"] for entry in coverage if entry["missing_assumption_ids"] or not entry["synthesis"]]
        design_ids_with_assumptions = {str(item["data"].get("scientific_design_id")) for item in assumptions}
        designs_without_assumption_maps = [str(item["id"]) for item in designs if str(item["id"]) not in design_ids_with_assumptions]
        v5_ids = {str(item["id"]) for item in designs if self._is_v5_design(item)}
        v5_coverage = [entry for entry in coverage if entry["design_id"] in v5_ids]
        fidelity_ids = {
            str(item["data"].get("scientific_design_id"))
            for item in self.store.objects_list(project_id, "ImplementationFidelityCheck", limit=None)
        }
        fidelity_not_correction = [
            entry["design_id"] for entry in coverage
            if entry["design_id"] in fidelity_ids and (entry["missing_assumption_ids"] or not entry["synthesis"])
        ]
        interpretation_result_ids = {str(item["data"].get("result_id")) for item in interpretation_attacks}
        inspected_runs = [str(item["id"]) for item in runs if item["status"] == "RESULT_INSPECTED"]
        regression_sources = {str(item["data"].get("source_id")) for item in regression_proposals}
        engineering_results = self.store.objects_list(project_id, "EngineeringResult", limit=None)
        regression_opportunities = [str(item["id"]) for item in engineering_results if str(item["id"]) not in regression_sources]
        design_attacks = self.store.objects_list(project_id, "ScientificDesignAttack", limit=None)
        fidelity_checks = self.store.objects_list(project_id, "ImplementationFidelityCheck", limit=None)
        fidelity_by_design: dict[str, list[str]] = {}
        for check in fidelity_checks:
            fidelity_by_design.setdefault(str(check["data"].get("scientific_design_id")), []).append(str(check["id"]))
        implementation_overreview = {
            design_id: ids for design_id, ids in fidelity_by_design.items() if len(ids) > 1
        }
        generic_work_ids = [
            f"work:{item['id']}" for item in work_items
            if "REVIEW" in str(item["kind"]).upper()
            and not (item["related_refs"] or {}).get("correction_layer")
            and self._classify_review_target(str(item["kind"]), {
                "title": item["title"], "description": item["description"],
                "scientific_role": item["scientific_role"], **(item["related_refs"] or {}),
            }) == "UNKNOWN"
        ]
        legacy_attacks = [
            str(item["id"]) for item in design_attacks
            if (item["data"] or {}).get("legacy_extraction_key")
        ]
        required_discriminators = self.store.objects_list(project_id, "RequiredDiscriminator", limit=None)
        confirmation_authorizations = self.store.objects_list(project_id, "ConfirmationAuthorization", limit=None)
        design_deficits = []
        for design, entry in zip(designs, coverage):
            attack_ids = [
                str(attack["id"]) for attack in design_attacks
                if str((attack["data"] or {}).get("target_design_id")) == str(design["id"])
            ]
            discriminators = [
                str(record["id"]) for record in required_discriminators
                if str((record["data"] or {}).get("design_id")) == str(design["id"])
            ]
            missing = []
            if not entry["assumption_count"]:
                missing.append("ASSUMPTION_MAP")
            if not attack_ids:
                missing.append("INDEPENDENT_DESIGN_ATTACK")
            if not discriminators and not entry["synthesis"]:
                missing.append("DISCRIMINATOR_OR_EXPLICIT_SURVIVAL_RECORD")
            if missing:
                design_deficits.append({"design_id": str(design["id"]), "missing": missing})
        return {
            "read_only": True, "implementation_version": VERSION,
            "review_like_work_by_type": classifications,
            "review_like_work_item_count": len(work_items),
            "likely_design_attacks": classifications["DESIGN_ATTACK"],
            "likely_implementation_fidelity_checks": classifications["IMPLEMENTATION_FIDELITY"],
            "likely_result_interpretation_attacks": classifications["RESULT_INTERPRETATION"],
            "mechanical_reviews_that_should_be_deterministic": classifications["DETERMINISTIC_ENGINEERING"],
            "major_designs_lacking_explicit_self_correction": incomplete,
            "major_designs_lacking_assumption_maps": designs_without_assumption_maps,
            "implementation_reviews_not_counted_as_design_correction": fidelity_not_correction,
            "current_v5_correction_coverage": v5_coverage,
            "v5_missing_attack_preview": self.v5_missing_attack_work_preview(project_id),
            "result_interpretation_correction_coverage": {
                "inspected_result_ids": inspected_runs,
                "attacked_result_ids": sorted(interpretation_result_ids),
                "unattacked_inspected_result_ids": sorted(set(inspected_runs) - interpretation_result_ids),
            },
            "engineering_review_to_test_conversion_opportunity_ids": regression_opportunities,
            "design_coverage": coverage,
            "generic_code_review_frequency": len(generic_work_ids),
            "review_bureaucracy_detector": {
                "engineering_review_amplification_work_item_ids": generic_work_ids,
                "remedy": "DETERMINISTIC_REGRESSION_PLUS_DELTA_VERIFICATION",
            },
            "self_correction_deficit_detector": {
                "code": "DESIGN_SELF_CORRECTION_MISSING",
                "designs": design_deficits,
            },
            "implementation_overreview_detector": {
                "code": "IMPLEMENTATION_OVERREVIEW",
                "repeated_fidelity_check_ids_by_design": implementation_overreview,
            },
            "legacy_design_attack_extraction_ids": legacy_attacks,
            "correction_metrics": {
                "major_designs_created": len(designs),
                "major_designs_independently_attacked": len({str(item["data"].get("target_design_id")) for item in design_attacks if not (item["data"] or {}).get("legacy_extraction_key")}),
                "design_assumptions_attacked": len({str(item) for entry in coverage for item in entry["covered_assumption_ids"]}),
                "unique_material_challenges": len({str(item["id"]) for item in self.store.objects_list(project_id, "DesignChallenge", limit=None) if str((item["data"] or {}).get("severity")).upper() in {"HIGH", "CRITICAL"}}),
                "discriminating_experiments_generated": len(required_discriminators),
                "targeted_fidelity_checks": len(fidelity_checks),
                "deterministic_engineering_checks": len(regression_proposals),
                "interpretation_attacks": len(interpretation_attacks),
            },
            "historical_classification_persisted": bool(persisted),
            "persisted_review_classification_count": len(persisted),
            "confirmation_integrity": {
                "status": "UNCHANGED_BY_V366_AUDIT",
                "authorization_record_count": len(confirmation_authorizations),
                "v366_records_grant_confirmation_access": False,
            },
        }
