"""Durable v3.3 distributed discovery, deliberately separate from belief.

The service uses Research OS objects for immutable scientific-search records and
two small PostgreSQL coordination tables only for uniqueness/phase races.  It
does not execute experiments, update a Hypothesis, or treat agreement between
workers as evidence.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from .discovery import ScientificDistance, SearchRegime, classify_scientific_distance
from .errors import GPUError
from .research import ResearchStore

DDE_VERSION = "distributed-discovery-engine-v3.3"
ROUND_PHASES = {
    "CREATED", "STATE_FROZEN", "INDEPENDENT_GENERATION", "GENERATION_FROZEN",
    "CHARACTERIZATION", "DEAD_MEMORY_SCREEN", "QD_ARCHIVE", "LITERATURE_PASS",
    "SYNTHESIS", "COMPLETED", "INVALID", "STALE",
}
GENERATION_OPERATORS = {
    "LOCAL_CAUSAL_REPAIR", "CAUSAL_INVERSION", "REPRESENTATION_RESET",
    "LATENT_OBJECT_REDESIGN", "INFORMATION_PATH_REDESIGN", "GENERATIVE_PROCESS_CHANGE",
    "ONTOLOGY_CHALLENGE", "OBJECTIVE_REFORMULATION", "CROSS_DOMAIN_STRUCTURAL_TRANSFER",
    "STRONG_NULL_CONSTRUCTION", "ASSUMPTION_REMOVAL", "ASSUMPTION_INVERSION",
    "BOTTLENECK_FIRST_DESIGN", "BREAKTHROUGH_EXTRACTION", "FRONTIER_BACKWARD_DESIGN",
}
MEMBER_STATUSES = {"JOINED", "FROZEN", "ABSTAINED", "FAILED", "EXPIRED"}
DISCOVERY_KINDS = {
    "DiscoveryRound", "CandidateBatch", "DiscoveryCandidate", "MechanisticNiche",
    "CrossWorkerQDArchive", "DiscoveryCandidateOutcome", "DiscoveryCoverageAtDecision",
}
CONTROL_PLANE_KINDS = {
    # Brain-step materialization records describe selection/provenance.  They
    # are not new observations, hypotheses, claims, or negative evidence.
    "HypothesisPortfolio", "ResearchSituation", "CandidatePortfolio",
    "ResearchActionCandidate", "ResearchDecision", "ResearchDecisionOutcome",
    "ResearchStrategyPattern", "StrategyOutcome",
}


def _snapshot_relevant_kind(kind: str) -> bool:
    """Return whether an object can change the scientific discovery context.

    Engineering implementation bookkeeping can be frequent while a discovery
    round is generating.  It is operational provenance, not a scientific
    state transition, and must never stale an independent generation round.
    """
    return (
        kind not in DISCOVERY_KINDS
        and kind not in CONTROL_PLANE_KINDS
        and not kind.startswith("Engineering")
    )


class DistributedDiscoveryService:
    """PostgreSQL-durable coordinator for independent mechanistic search."""

    def __init__(self, store: ResearchStore, *, migrate: bool = True):
        self.store = store
        if migrate:
            self._migrate()

    def _migrate(self) -> None:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('gpu_lab_dde_v33_migration'))")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS discovery_round_memberships (
                    discovery_round_id UUID NOT NULL REFERENCES research_objects(id),
                    worker_id UUID NOT NULL, worker_session_id UUID NOT NULL,
                    candidate_batch_id UUID NOT NULL UNIQUE REFERENCES research_objects(id),
                    generation_operator TEXT NOT NULL, requested_distance TEXT NOT NULL,
                    status TEXT NOT NULL, independent_generation BOOLEAN NOT NULL DEFAULT TRUE,
                    joined_at TIMESTAMPTZ NOT NULL, frozen_at TIMESTAMPTZ,
                    failure_reason TEXT,
                    PRIMARY KEY(discovery_round_id, worker_session_id)
                );
                CREATE INDEX IF NOT EXISTS discovery_round_memberships_round_idx
                    ON discovery_round_memberships(discovery_round_id,status);
                CREATE TABLE IF NOT EXISTS discovery_round_syntheses (
                    discovery_round_id UUID PRIMARY KEY REFERENCES research_objects(id),
                    archive_id UUID NOT NULL UNIQUE REFERENCES research_objects(id),
                    created_at TIMESTAMPTZ NOT NULL
                );
            """)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _hash(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    @staticmethod
    def _json(value: Any) -> str:
        """Serialize DB-origin UUID/timestamp values without weakening data validation."""
        return json.dumps(value, default=str)

    @staticmethod
    def _record(row: dict | None) -> dict | None:
        if not row:
            return None
        return {key: str(value) if isinstance(value, uuid.UUID) else value for key, value in row.items()}

    def _scientific_snapshot(self, project_id: str, *, legacy_full_data: bool = False) -> dict[str, Any]:
        state = self.store.state_get(project_id)
        records = sorted([
            # A frozen discovery snapshot needs an immutable identity and
            # content hash, not a second copy of every historical document.
            # This keeps future rounds bounded even for long-lived projects.
            (
                {"id": str(item["id"]), "kind": item["kind"], "status": item["status"], "data": item["data"]}
                if legacy_full_data else
                {"id": str(item["id"]), "kind": item["kind"], "status": item["status"], "data_hash": self._hash(item["data"])}
            )
            for item in state["objects"] if _snapshot_relevant_kind(item["kind"])
        ], key=lambda item: item["id"])
        return {
            "research_state_version": state["state_freshness"]["research_state_version"],
            "world_model_version": state["state_freshness"].get("world_model_version"),
            "negative_result_snapshot_version": len(state["canonical_state"].get("negative_results", [])),
            "records": records,
            "active_hypothesis_ids": [item["id"] for item in records if item["kind"] == "Hypothesis" and item["status"] in {"ACTIVE", "SURVIVES_INITIAL_TEST"}],
            "negative_result_ids": [item["id"] for item in records if item["kind"] == "NegativeResult"],
            "frontier_gap_ids": [item["id"] for item in records if item["kind"] in {"FrontierGap", "BenchmarkGap"}],
            "stagnation_ids": [item["id"] for item in records if item["kind"] == "StagnationState"],
            "architecture_lineage_ids": [item["id"] for item in records if item["kind"] == "ArchitectureLineage"],
            "breakthrough_signal_ids": [item["id"] for item in records if item["kind"] == "BreakthroughSignal"],
        }

    @staticmethod
    def reservations(regime: str) -> dict[str, int]:
        regime = str(regime).upper()
        if regime not in {item.value for item in SearchRegime}:
            raise GPUError("DISCOVERY_SEARCH_REGIME_INVALID", regime)
        policy = {
            "EXPLOIT": {"NEAR": 2, "MID": 1, "FAR": 1, "ORTHOGONAL": 0},
            "MECHANISM_SEARCH": {"NEAR": 1, "MID": 1, "FAR": 1, "ORTHOGONAL": 1},
            "DIVERGENT_SEARCH": {"NEAR": 1, "MID": 1, "FAR": 2, "ORTHOGONAL": 1},
            "PARADIGM_RESET": {"NEAR": 1, "MID": 1, "FAR": 2, "ORTHOGONAL": 1},
        }
        return policy[regime]

    def create_round(
        self, project_id: str, agenda_item_id: str | None, search_regime: str,
        *, triggering_decision_id: str | None = None, generation_budget: dict[str, int] | None = None,
        policy_version: str | None = None, brain_policy_version: str | None = None,
    ) -> dict:
        snapshot = self._scientific_snapshot(project_id)
        budget = {"max_workers": 4, "max_candidates_per_batch": 3, "max_generation_waves": 2, "max_literature_calls": 1, "max_synthesis_rounds": 1, **(generation_budget or {})}
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in budget.values()):
            raise GPUError("DISCOVERY_BUDGET_INVALID", "Discovery budget values must be non-negative integers")
        if agenda_item_id:
            agenda = self.store.object_get(agenda_item_id)
            if agenda["kind"] != "AgendaItem" or str(agenda["project_id"]) != str(project_id):
                raise GPUError("DISCOVERY_AGENDA_ITEM_INVALID", agenda_item_id)
        baseline_signature: dict[str, Any] = {}
        baseline_signature_source: dict[str, Any] | None = None
        if triggering_decision_id:
            decision = self.store.object_get(triggering_decision_id)
            if decision["kind"] != "ResearchDecision" or str(decision["project_id"]) != str(project_id):
                raise GPUError("DISCOVERY_TRIGGERING_DECISION_INVALID", str(triggering_decision_id))
            selected = decision.get("data", {}).get("selected_action", {})
            try:
                baseline_signature = self._signature(selected)
            except GPUError as exc:
                if exc.error_type != "DISCOVERY_DIVERSITY_SIGNATURE_REQUIRED":
                    raise
            if baseline_signature:
                baseline_signature_source = {
                    "kind": "ResearchDecision",
                    "id": str(triggering_decision_id),
                    "selected_action_id": selected.get("id"),
                }
        data = {
            "implementation_version": DDE_VERSION, "agenda_item_id": agenda_item_id,
            "triggering_decision_id": triggering_decision_id, "search_regime": search_regime.upper(),
            "baseline_signature": baseline_signature,
            "baseline_signature_source": baseline_signature_source,
            "phase": "INDEPENDENT_GENERATION", "peer_visibility": "HIDDEN",
            "independent_generation": True, "required_distance_coverage": self.reservations(search_regime),
            "generation_budget": budget, "frozen_state": snapshot,
            "frozen_state_hash": self._hash(snapshot), "scientific_snapshot_hash": self._hash(snapshot["records"]),
            "research_state_version": snapshot["research_state_version"],
            "world_model_version": snapshot["world_model_version"],
            "agenda_version": str(agenda_item_id) if agenda_item_id else None,
            "frontier_gap_snapshot": snapshot["frontier_gap_ids"],
            "stagnation_snapshot": snapshot["stagnation_ids"],
            "architecture_lineage_snapshot": snapshot["architecture_lineage_ids"],
            "negative_result_snapshot_version": snapshot["negative_result_snapshot_version"],
            "policy_version": policy_version, "brain_policy_version": brain_policy_version,
            "literature_status": "NOT_REQUESTED", "generation_wave": 1,
            "started_at": self._now().isoformat(),
        }
        result = self.store.object_create(project_id, "DiscoveryRound", data, "DISCOVERY_ROUND_CREATED", "ACTIVE")
        self._event(project_id, "DISCOVERY_STATE_FROZEN", result["id"], {"frozen_state_hash": data["frozen_state_hash"]})
        return result

    def stale_check(self, round_id: str, mark_stale: bool = False) -> dict:
        """Detect external scientific-state changes without mixing generations across state versions."""
        round_ = self._round(round_id)
        frozen_records = round_["data"].get("frozen_state", {}).get("records", [])
        # Older rounds stored engineering records before the snapshot-scope
        # correction. Filter both sides so the corrected semantics are also
        # safe for an already-active legacy round.
        scientific_frozen_records = sorted([
            item for item in frozen_records if _snapshot_relevant_kind(str(item.get("kind", "")))
        ], key=lambda item: str(item["id"]))
        current = self._scientific_snapshot(
            str(round_["project_id"]),
            legacy_full_data=bool(frozen_records and "data" in frozen_records[0]),
        )
        current_hash = self._hash(current["records"])
        frozen_hash = self._hash(scientific_frozen_records)
        stale = current_hash != frozen_hash
        frozen_by_id = {str(item["id"]): item for item in scientific_frozen_records}
        current_by_id = {str(item["id"]): item for item in current["records"]}
        changed_ids = sorted(
            object_id for object_id in frozen_by_id.keys() | current_by_id.keys()
            if frozen_by_id.get(object_id) != current_by_id.get(object_id)
        )
        result = {
            "discovery_round_id": str(round_id), "stale": stale,
            "frozen_scientific_snapshot_hash": frozen_hash,
            "current_scientific_snapshot_hash": current_hash,
            "changed_scientific_record_ids": changed_ids[:50],
            "changed_scientific_record_count": len(changed_ids),
        }
        if stale and mark_stale:
            now = self._now()
            with self.store._connect() as conn, conn.cursor() as cur:
                # Serialise every generation-state transition on the parent.
                # Candidate submission takes the same lock before creating a
                # proposal, so a stale transition cannot leave a JOINED batch
                # apparently writable under a non-generating parent.
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"dde-round:{round_id}",))
                cur.execute("SELECT * FROM research_objects WHERE id=%s AND kind='DiscoveryRound' FOR UPDATE", (round_id,))
                locked_round = self._record(cur.fetchone())
                if not locked_round:
                    raise GPUError("DISCOVERY_ROUND_NOT_FOUND", str(round_id))
                # Also reconcile legacy stale rounds created before open child
                # batches were expired atomically.
                if locked_round["status"] in {"ACTIVE", "STALE"}:
                    cur.execute(
                        "SELECT * FROM discovery_round_memberships WHERE discovery_round_id=%s AND status='JOINED' FOR UPDATE",
                        (round_id,),
                    )
                    open_members = cur.fetchall()
                    expired_batch_ids: list[str] = []
                    for member in open_members:
                        batch_id = str(member["candidate_batch_id"])
                        cur.execute("SELECT data FROM research_objects WHERE id=%s AND kind='CandidateBatch' FOR UPDATE", (batch_id,))
                        batch = cur.fetchone()
                        if batch:
                            batch_data = {
                                **batch["data"],
                                "status": "EXPIRED",
                                "writeable": False,
                                "expired_at": now.isoformat(),
                                "expiration_reason": "ROUND_STALE",
                            }
                            cur.execute("UPDATE research_objects SET status='EXPIRED',data=%s WHERE id=%s", (self._json(batch_data), batch_id))
                            expired_batch_ids.append(batch_id)
                        cur.execute(
                            "UPDATE discovery_round_memberships SET status='EXPIRED',frozen_at=%s,failure_reason='ROUND_STALE' "
                            "WHERE discovery_round_id=%s AND candidate_batch_id=%s",
                            (now, round_id, batch_id),
                        )
                    round_data = {
                        **locked_round["data"],
                        "phase": "STALE",
                        "stale_detected_at": now.isoformat(),
                        "recovery_action": "CREATE_FRESH_ROUND",
                        **result,
                    }
                    cur.execute("UPDATE research_objects SET status='STALE',data=%s WHERE id=%s", (self._json(round_data), round_id))
                    for batch_id in expired_batch_ids:
                        self.store._event(cur, str(locked_round["project_id"]), "DISCOVERY_BATCH_EXPIRED", batch_id, {"round_id": str(round_id), "reason": "ROUND_STALE"})
                    self.store._event(
                        cur,
                        str(locked_round["project_id"]),
                        "DISCOVERY_ROUND_STALE" if locked_round["status"] == "ACTIVE" else "DISCOVERY_STALE_ROUND_RECONCILED",
                        str(round_id),
                        {**result, "expired_batch_ids": expired_batch_ids},
                    )
            result["round"] = self._round(str(round_id))
        return result

    def shadow_preview(self, project_id: str, candidates: list[dict[str, Any]], search_regime: str) -> dict:
        """Read-only v3.3 characterization of candidate inputs from a frozen snapshot."""
        snapshot = self._scientific_snapshot(project_id)
        reservations = self.reservations(search_regime)
        characterized = []
        baseline_signature: dict[str, Any] | None = None
        for index, raw in enumerate(candidates):
            try:
                signature = self._signature(raw)
            except GPUError as exc:
                if exc.error_type != "DISCOVERY_DIVERSITY_SIGNATURE_REQUIRED":
                    raise
                signature = {}
            if signature and baseline_signature is None:
                # Match Brain._discovery_portfolio semantics: the first serious
                # candidate is the read-only comparison baseline.  Comparing to
                # an empty signature makes every explicit causal_object look
                # ORTHOGONAL and collapses NEAR/MID/FAR coverage.
                baseline_signature = dict(signature)
            distance = classify_scientific_distance(
                {"payload": {"scientific_dimensions": signature}},
                {"payload": {"scientific_dimensions": baseline_signature or {}}},
            ) if signature else {"scientific_distance": "UNCHARACTERIZED", "reason": "Existing candidate lacks a structured DiversitySignature"}
            characterized.append({
                "index": index, "title": raw.get("title") or raw.get("action_type", f"candidate-{index}"),
                "mechanistic_niche": self._niche(signature, raw.get("mechanistic_niche")),
                "scientific_distance": distance["scientific_distance"],
                "diversity_signature": signature,
                "generation_operator": sorted(GENERATION_OPERATORS)[index % len(GENERATION_OPERATORS)],
                "characterization_status": "CHARACTERIZED" if signature else "NICHE_UNRESOLVED",
                "characterization_reason": distance["reason"],
            })
        coverage = Counter(item["scientific_distance"] for item in characterized)
        niches = Counter(item["mechanistic_niche"] for item in characterized)
        return {
            "read_only": True, "implementation_version": DDE_VERSION,
            "frozen_scientific_snapshot_hash": self._hash(snapshot["records"]),
            "search_regime": search_regime.upper(), "required_distance_coverage": reservations,
            "baseline_signature": baseline_signature or {},
            "candidates": characterized, "effective_niche_count": len(niches),
            "niche_distribution": dict(niches), "scientific_distance_distribution": dict(coverage),
            "unfilled_distance_slots": [key for key, amount in reservations.items() if amount and not coverage[key]],
            "uncharacterized_candidate_count": sum(item["characterization_status"] != "CHARACTERIZED" for item in characterized),
            "literature_status": "NOT_CALLED_READ_ONLY", "executed_experiments": 0,
        }

    def _event(self, project_id: str, event_type: str, subject_id: str | None, payload: dict[str, Any]) -> None:
        with self.store._connect() as conn, conn.cursor() as cur:
            self.store._event(
                cur,
                str(project_id),
                event_type,
                str(subject_id) if subject_id else None,
                json.loads(self._json(payload)),
            )

    def _round(self, round_id: str) -> dict:
        round_ = self.store.object_get(round_id)
        if round_["kind"] != "DiscoveryRound":
            raise GPUError("DISCOVERY_ROUND_NOT_FOUND", round_id)
        return round_

    def _assert_independent_phase(self, round_: dict, batch_effective_status: str | None = None) -> None:
        if round_["status"] != "ACTIVE" or round_["data"].get("phase") != "INDEPENDENT_GENERATION":
            raise GPUError(
                "DISCOVERY_ROUND_NOT_GENERATING",
                "Discovery round is not accepting candidate submissions.",
                details={
                    "actual_round_phase": round_["data"].get("phase"),
                    "stale": round_["status"] == "STALE" or round_["data"].get("phase") == "STALE",
                    "batch_effective_status": batch_effective_status,
                    "recovery_action": round_["data"].get("recovery_action"),
                },
            )

    def join_round(
        self, round_id: str, worker_id: str, session_id: str, generation_operator: str,
        requested_distance: str,
    ) -> dict:
        round_ = self._round(round_id)
        self._assert_independent_phase(round_)
        operator, distance = generation_operator.upper(), requested_distance.upper()
        if operator not in GENERATION_OPERATORS:
            raise GPUError("DISCOVERY_GENERATION_OPERATOR_INVALID", operator)
        if distance not in {item.value for item in ScientificDistance}:
            raise GPUError("DISCOVERY_DISTANCE_INVALID", distance)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"dde-member:{round_id}:{session_id}",))
            cur.execute("SELECT * FROM discovery_round_memberships WHERE discovery_round_id=%s AND worker_session_id=%s FOR UPDATE", (round_id, session_id))
            existing = cur.fetchone()
            if existing:
                return self.store.object_get(str(existing["candidate_batch_id"]))
            cur.execute("SELECT count(*) AS count FROM discovery_round_memberships WHERE discovery_round_id=%s", (round_id,))
            member_count = int(cur.fetchone()["count"])
            max_workers = round_["data"]["generation_budget"]["max_workers"]
            if member_count >= max_workers:
                raise GPUError("DISCOVERY_WORKER_BUDGET_EXHAUSTED", round_id)
            cur.execute(
                "SELECT requested_distance,count(*) AS count FROM discovery_round_memberships "
                "WHERE discovery_round_id=%s GROUP BY requested_distance",
                (round_id,),
            )
            assignments = {row["requested_distance"]: int(row["count"]) for row in cur.fetchall()}
            required = {key for key, amount in round_["data"]["required_distance_coverage"].items() if amount}
            missing_before = {key for key in required if not assignments.get(key)}
            remaining_after_join = max_workers - member_count - 1
            # A repeated local slot cannot consume the final opportunity that
            # is needed to attempt an as-yet-unrepresented scientific radius.
            if distance not in missing_before and len(missing_before) > remaining_after_join:
                raise GPUError("DISCOVERY_DISTANCE_SLOT_RESERVED", ",".join(sorted(missing_before)))
            cur.execute("SELECT id FROM research_worker_sessions WHERE id=%s AND worker_id=%s AND current_project_id=%s AND status NOT IN ('DISCONNECTED','EXPIRED')", (session_id, worker_id, round_["project_id"]))
            if not cur.fetchone():
                raise GPUError("DISCOVERY_WORKER_SESSION_INVALID", session_id)
            batch_id = str(uuid.uuid4())
            batch = {
                "discovery_round_id": round_id, "worker_id": worker_id, "worker_session_id": session_id,
                "generation_operator": operator, "requested_distance": distance, "candidate_ids": [],
                "status": "JOINED", "immutable_after_freeze": True, "created_at": now.isoformat(),
            }
            cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'CandidateBatch','ACTIVE',%s,%s)", (batch_id, round_["project_id"], self._json(batch), now))
            self.store._event(cur, round_["project_id"], "DISCOVERY_BATCH_JOINED", batch_id, json.loads(self._json(batch)))
            cur.execute("INSERT INTO discovery_round_memberships(discovery_round_id,worker_id,worker_session_id,candidate_batch_id,generation_operator,requested_distance,status,joined_at) VALUES(%s,%s,%s,%s,%s,%s,'JOINED',%s)", (round_id, worker_id, session_id, batch_id, operator, distance, now))
        return self.store.object_get(batch_id)

    def recommended_assignments(self, round_id: str, worker_slots: int | None = None) -> list[dict[str, str]]:
        """Suggest distinct search transformations without assigning worker personas.

        The caller still binds each recommendation to a real worker/session via
        ``join_round``.  Recommendations are derived only from the frozen round
        regime and currently unfilled reservation slots.
        """
        round_ = self._round(round_id)
        self._assert_independent_phase(round_)
        members = self._members(round_id)
        used_operators = {member["generation_operator"] for member in members}
        counts = Counter(member["requested_distance"] for member in members)
        required_coverage = round_["data"]["required_distance_coverage"]
        # Cover each required scientific radius once before assigning a second
        # attempt to any radius.  Otherwise DIVERGENT_SEARCH's second FAR
        # reservation can consume the final worker slot and silently omit the
        # required ORTHOGONAL attempt.
        missing = [
            distance for distance, required in required_coverage.items()
            if required and not counts[distance]
        ]
        missing.extend(
            distance for distance, required in required_coverage.items()
            for _ in range(max(0, required - max(1, counts[distance])))
        )
        regime_operators = {
            "EXPLOIT": ["LOCAL_CAUSAL_REPAIR", "CAUSAL_INVERSION", "REPRESENTATION_RESET"],
            "MECHANISM_SEARCH": ["LOCAL_CAUSAL_REPAIR", "CAUSAL_INVERSION", "INFORMATION_PATH_REDESIGN", "STRONG_NULL_CONSTRUCTION"],
            "DIVERGENT_SEARCH": ["REPRESENTATION_RESET", "GENERATIVE_PROCESS_CHANGE", "STRONG_NULL_CONSTRUCTION", "ONTOLOGY_CHALLENGE", "CROSS_DOMAIN_STRUCTURAL_TRANSFER"],
            "PARADIGM_RESET": ["REPRESENTATION_RESET", "ONTOLOGY_CHALLENGE", "OBJECTIVE_REFORMULATION", "STRONG_NULL_CONSTRUCTION", "GENERATIVE_PROCESS_CHANGE"],
        }[round_["data"]["search_regime"]]
        available = [operator for operator in regime_operators if operator not in used_operators]
        available.extend(operator for operator in sorted(GENERATION_OPERATORS) if operator not in used_operators and operator not in available)
        remaining = round_["data"]["generation_budget"]["max_workers"] - len(members)
        count = min(remaining, worker_slots if worker_slots is not None else remaining, len(missing), len(available))
        return [
            {"generation_operator": available[index], "requested_distance": missing[index], "reason": "Missing reserved search coverage"}
            for index in range(count)
        ]

    @staticmethod
    def _signature(candidate: dict[str, Any]) -> dict[str, Any]:
        payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
        value = (
            candidate.get("diversity_signature")
            or candidate.get("scientific_dimensions")
            or payload.get("scientific_dimensions")
            or {}
        )
        if not isinstance(value, dict):
            raise GPUError("DISCOVERY_DIVERSITY_SIGNATURE_INVALID", "Signature must be structured")
        signature = {str(key): value[key] for key in sorted(value) if value[key] not in (None, "", [], {})}
        if not signature:
            raise GPUError("DISCOVERY_DIVERSITY_SIGNATURE_REQUIRED", "A serious candidate requires a structured signature")
        return signature

    @staticmethod
    def _niche(signature: dict[str, Any], supplied: str | None) -> str:
        # A model-provided label is advisory only.  The identity used for QD is
        # derived from scientific structure so cosmetic labels cannot fake
        # niche diversity.
        ordered = ("causal_object", "representation", "generative_process", "objective_formulation", "architecture_family", "information_path")
        for key in ordered:
            if signature.get(key):
                return f"{key.upper()}::{str(signature[key]).upper()}"
        return "NICHE_UNRESOLVED"

    @staticmethod
    def _validate_candidate(candidate: dict[str, Any]) -> None:
        if not isinstance(candidate.get("mechanism"), str) or not candidate["mechanism"].strip():
            raise GPUError("DISCOVERY_CANDIDATE_MECHANISM_REQUIRED", "candidate.mechanism must be a non-empty string")
        predictions = candidate.get("predictions")
        if not isinstance(predictions, list) or not any(isinstance(item, str) and item.strip() for item in predictions):
            raise GPUError(
                "DISCOVERY_CANDIDATE_PREDICTION_REQUIRED",
                "candidate.predictions must be a non-empty array containing at least one non-empty string; "
                "candidate.discriminating_prediction is not a supported field",
            )
        if not isinstance(candidate.get("falsifier"), str) or not candidate["falsifier"].strip():
            raise GPUError("DISCOVERY_CANDIDATE_FALSIFIER_REQUIRED", "candidate.falsifier must be a non-empty string")

    def submit_candidate(self, round_id: str, batch_id: str, worker_id: str, session_id: str, candidate: dict[str, Any]) -> dict:
        self._validate_candidate(candidate)
        signature = self._signature(candidate)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"dde-round:{round_id}",))
            cur.execute("SELECT * FROM research_objects WHERE id=%s AND kind='DiscoveryRound' FOR UPDATE", (round_id,))
            round_ = self._record(cur.fetchone())
            if not round_:
                raise GPUError("DISCOVERY_ROUND_NOT_FOUND", str(round_id))
            cur.execute("SELECT * FROM discovery_round_memberships WHERE discovery_round_id=%s AND candidate_batch_id=%s AND worker_id=%s AND worker_session_id=%s FOR UPDATE", (round_id, batch_id, worker_id, session_id))
            member = cur.fetchone()
            self._assert_independent_phase(round_, member["status"] if member else None)
            if not member or member["status"] != "JOINED":
                raise GPUError("DISCOVERY_BATCH_NOT_WRITABLE", batch_id)
            cur.execute("SELECT data FROM research_objects WHERE id=%s AND kind='CandidateBatch' FOR UPDATE", (batch_id,))
            batch = cur.fetchone()
            if not batch:
                raise GPUError("DISCOVERY_BATCH_NOT_FOUND", batch_id)
            candidate_ids = batch["data"].get("candidate_ids", [])
            if len(candidate_ids) >= round_["data"]["generation_budget"]["max_candidates_per_batch"]:
                raise GPUError("DISCOVERY_BATCH_CANDIDATE_BUDGET_EXHAUSTED", batch_id)
            distance = classify_scientific_distance({"payload": {"scientific_dimensions": signature}}, {"payload": {"scientific_dimensions": round_["data"].get("baseline_signature", {})}})
            requested = member["requested_distance"]
            data = {
                "implementation_version": DDE_VERSION, "discovery_round_id": round_id, "candidate_batch_id": batch_id,
                "worker_id": worker_id, "worker_session_id": session_id, "generation_operator": member["generation_operator"],
                "requested_distance": requested, "title": str(candidate.get("title", "Untitled discovery candidate"))[:500],
                "mechanism": candidate.get("mechanism"), "predictions": candidate.get("predictions", []),
                "falsifier": candidate.get("falsifier"), "diversity_signature": signature,
                "mechanistic_niche": self._niche(signature, candidate.get("mechanistic_niche")),
                "requested_mechanistic_niche": candidate.get("mechanistic_niche"),
                "scientific_distance": distance["scientific_distance"], "distance_reason": distance["reason"],
                "changed_scientific_dimensions": distance["changed_scientific_dimensions"],
                "architecture_lineage": candidate.get("architecture_lineage") or signature.get("architecture_family"),
                "parent_candidate_ids": list(candidate.get("parent_candidate_ids") or []),
                "genealogy_relation": candidate.get("genealogy_relation", "INDEPENDENT_GENERATION"),
                "quality_components": dict(candidate.get("quality_components") or {}),
                "expected_failure_modes": list(candidate.get("expected_failure_modes") or []),
                "novelty_status": "NOVELTY_UNVERIFIED", "dead_memory_status": "NOT_SCREENED",
                "proposal_state_hash": round_["data"]["frozen_state_hash"], "proposed_at": now.isoformat(),
                "independent_generation": bool(member["independent_generation"]), "selected_for_portfolio": False,
                "selected_for_execution": False,
            }
            candidate_id = str(uuid.uuid4())
            cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'DiscoveryCandidate','PROPOSED',%s,%s)", (candidate_id, round_["project_id"], self._json(data), now))
            self.store._event(cur, round_["project_id"], "DISCOVERY_CANDIDATE_PROPOSED", candidate_id, json.loads(self._json({"round_id": round_id, "batch_id": batch_id, "mechanistic_niche": data["mechanistic_niche"], "scientific_distance": data["scientific_distance"]})))
            batch_data = {**batch["data"], "candidate_ids": [*candidate_ids, candidate_id]}
            cur.execute("UPDATE research_objects SET data=%s WHERE id=%s", (self._json(batch_data), batch_id))
        return self.store.object_get(candidate_id)

    def _batch(self, batch_id: str) -> dict:
        batch = self.store.object_get(batch_id)
        if batch["kind"] != "CandidateBatch":
            raise GPUError("DISCOVERY_BATCH_NOT_FOUND", batch_id)
        return batch

    def batch_freeze(self, round_id: str, batch_id: str, worker_id: str, session_id: str, abstention_reason: str | None = None) -> dict:
        round_ = self._round(round_id)
        self._assert_independent_phase(round_)
        now = self._now()
        synthesis_ready = False
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM discovery_round_memberships WHERE discovery_round_id=%s AND candidate_batch_id=%s AND worker_id=%s AND worker_session_id=%s FOR UPDATE", (round_id, batch_id, worker_id, session_id))
            member = cur.fetchone()
            if not member:
                raise GPUError("DISCOVERY_BATCH_NOT_OWNED", batch_id)
            if member["status"] in {"FROZEN", "ABSTAINED"}:
                return self.store.object_get(batch_id)
            if member["status"] != "JOINED":
                raise GPUError("DISCOVERY_BATCH_NOT_FREEZABLE", member["status"])
            cur.execute("SELECT data FROM research_objects WHERE id=%s AND kind='CandidateBatch' FOR UPDATE", (batch_id,))
            batch = cur.fetchone()
            candidate_ids = batch["data"].get("candidate_ids", []) if batch else []
            if not candidate_ids and not (abstention_reason or "").strip():
                raise GPUError("DISCOVERY_ABSTENTION_REASON_REQUIRED", batch_id)
            status = "FROZEN" if candidate_ids else "ABSTAINED"
            batch_data = {**batch["data"], "status": status, "frozen_at": now.isoformat(), "abstention_reason": abstention_reason}
            cur.execute("UPDATE research_objects SET status=%s,data=%s WHERE id=%s", (status, self._json(batch_data), batch_id))
            cur.execute("UPDATE discovery_round_memberships SET status=%s,frozen_at=%s,failure_reason=%s WHERE discovery_round_id=%s AND candidate_batch_id=%s", (status, now, abstention_reason, round_id, batch_id))
            self.store._event(cur, round_["project_id"], "CANDIDATE_BATCH_FROZEN", batch_id, json.loads(self._json({"round_id": round_id, "status": status, "candidate_count": len(candidate_ids), "abstention_reason": abstention_reason})))
            if not candidate_ids:
                self.store._event(cur, round_["project_id"], "DISCOVERY_DISTANCE_SLOT_UNFILLED", round_id, {
                    "requested_distance": member["requested_distance"], "generation_operator": member["generation_operator"],
                    "reason": abstention_reason,
                })
            cur.execute("SELECT count(*) FILTER (WHERE status='JOINED') AS pending FROM discovery_round_memberships WHERE discovery_round_id=%s", (round_id,))
            if int(cur.fetchone()["pending"]) == 0:
                new_data = {**round_["data"], "phase": "GENERATION_FROZEN", "peer_visibility": "VISIBLE_FOR_SYNTHESIS", "generation_frozen_at": now.isoformat()}
                cur.execute("UPDATE research_objects SET data=%s WHERE id=%s", (self._json(new_data), round_id))
                self.store._event(cur, round_["project_id"], "DISCOVERY_GENERATION_FROZEN", round_id, {"peer_visibility": "VISIBLE_FOR_SYNTHESIS"})
                synthesis_ready = True
        if synthesis_ready:
            self.synthesis_work_item(round_id, worker_id, session_id)
        return self.store.object_get(batch_id)

    def _members(self, round_id: str) -> list[dict]:
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM discovery_round_memberships WHERE discovery_round_id=%s ORDER BY joined_at", (round_id,))
            return [self._record(row) or {} for row in cur.fetchall()]

    def round_get(self, round_id: str, requester_session_id: str | None = None) -> dict:
        round_ = self._round(round_id)
        result = {**round_, "data": dict(round_["data"]), "members": self._members(round_id)}
        if round_["data"].get("peer_visibility") == "HIDDEN":
            for member in result["members"]:
                member.pop("candidate_batch_id", None)
            result["peer_candidates_visible"] = False
        else:
            result["peer_candidates_visible"] = True
        return result

    def batch_get(self, round_id: str, batch_id: str, requester_session_id: str) -> dict:
        round_ = self._round(round_id)
        batch = self._batch(batch_id)
        if str(batch["data"].get("discovery_round_id")) != str(round_id):
            raise GPUError("DISCOVERY_BATCH_ROUND_MISMATCH", batch_id)
        if round_["data"].get("peer_visibility") == "HIDDEN" and str(batch["data"].get("worker_session_id")) != str(requester_session_id):
            requester = next(
                (member for member in self._members(round_id) if member["worker_session_id"] == str(requester_session_id)),
                None,
            )
            if not requester or requester["independent_generation"]:
                raise GPUError("DISCOVERY_PEER_ISOLATION_ACTIVE", "Peer candidate batches are hidden until generation freezes")
        candidate_ids = batch["data"].get("candidate_ids", [])
        member = next((item for item in self._members(round_id) if item["candidate_batch_id"] == str(batch_id)), None)
        effective_status = member["status"] if member else batch["status"]
        writeable = (
            round_["status"] == "ACTIVE"
            and round_["data"].get("phase") == "INDEPENDENT_GENERATION"
            and effective_status == "JOINED"
            and batch["status"] == "ACTIVE"
        )
        return {
            **batch,
            "candidates": [self.store.object_get(candidate_id) for candidate_id in candidate_ids],
            "effective_status": effective_status,
            "writeable": writeable,
            "recovery_action": round_["data"].get("recovery_action"),
        }

    def peer_isolation_override(
        self, round_id: str, batch_id: str, worker_id: str, session_id: str, rationale: str
    ) -> dict:
        """Record an explicit human-directed loss of independence for one batch."""
        if not rationale.strip():
            raise GPUError("DISCOVERY_PEER_OVERRIDE_RATIONALE_REQUIRED", batch_id)
        round_ = self._round(round_id)
        self._assert_independent_phase(round_)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM discovery_round_memberships WHERE discovery_round_id=%s "
                "AND candidate_batch_id=%s AND worker_id=%s AND worker_session_id=%s FOR UPDATE",
                (round_id, batch_id, worker_id, session_id),
            )
            if not cur.fetchone():
                raise GPUError("DISCOVERY_BATCH_NOT_OWNED", batch_id)
            cur.execute(
                "UPDATE discovery_round_memberships SET independent_generation=FALSE "
                "WHERE discovery_round_id=%s AND candidate_batch_id=%s",
                (round_id, batch_id),
            )
            data = {
                **round_["data"], "independent_generation": False,
                "peer_isolation_override": {"batch_id": str(batch_id), "rationale": rationale[:4000], "at": now.isoformat()},
            }
            cur.execute("UPDATE research_objects SET data=%s WHERE id=%s", (self._json(data), round_id))
            self.store._event(cur, round_["project_id"], "PEER_ISOLATION_OVERRIDE", round_id, {"batch_id": str(batch_id), "rationale": rationale[:4000]})
        return self._round(round_id)

    def lab_work_items(self, round_id: str) -> list[dict]:
        """Materialize one bounded, non-executing generation WorkItem per joined worker."""
        # Local import avoids a module cycle and keeps Lab as operational ownership.
        from .lab import LabController

        round_ = self._round(round_id)
        controller = LabController(self.store)
        items: list[dict] = []
        for member in self._members(round_id):
            item = controller.create_work(
                str(round_["project_id"]), "DISCOVERY_GENERATION",
                f"Independent discovery: {member['generation_operator']}",
                "Generate a bounded candidate batch from frozen state. Current-round peer ideas are unavailable.",
                "DISCOVERY_GENERATOR", member["worker_id"], related_refs={
                    "discovery_round_id": str(round_id), "candidate_batch_id": member["candidate_batch_id"],
                    "generation_operator": member["generation_operator"], "requested_distance": member["requested_distance"],
                    "peer_visibility": "HIDDEN",
                }, equivalence_key=f"dde-v33:{round_id}:{member['worker_session_id']}",
                created_session_id=member["worker_session_id"], subject_id=str(round_id),
                recovery_policy={"reassign_same_work_item": True, "fallback_live_work_item": False},
            )
            items.append(item)
        return items

    def synthesis_work_item(self, round_id: str, worker_id: str, session_id: str) -> dict:
        """Materialize one READY operational synthesis task only after isolation ends."""
        from .lab import LabController

        round_ = self._round(round_id)
        if round_["data"].get("phase") != "GENERATION_FROZEN":
            raise GPUError("DISCOVERY_SYNTHESIS_NOT_READY", round_id)
        controller = LabController(self.store)
        key = f"dde-v33-synthesis:{round_id}"
        existing = [
            item for item in controller.work_list(str(round_["project_id"]), limit=500)
            if item.get("equivalence_key") == key
        ]
        if existing:
            return existing[0]
        return controller.create_work(
            str(round_["project_id"]), "DISCOVERY_SYNTHESIS", "Synthesize independent discovery batches",
            "Compare frozen batches through structured niche, distance, dead-memory, and QD checks.",
            "DISCOVERY_SYNTHESIS", worker_id,
            related_refs={"discovery_round_id": str(round_id), "peer_visibility": "VISIBLE_FOR_SYNTHESIS"},
            equivalence_key=key, created_session_id=session_id, subject_id=str(round_id),
            recovery_policy={"reassign_same_work_item": True, "fallback_live_work_item": False},
        )

    @staticmethod
    def _signature_key(candidate: dict) -> str:
        return json.dumps(candidate["data"].get("diversity_signature", {}), sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _scientific_equivalence_key(candidate: dict) -> str:
        """Ignore hyperparameter-only dimensions when de-duplicating a niche."""
        cosmetic = {"learning_rate", "width", "residual_cap", "loss_weight", "seed", "batch_size", "epochs"}
        signature = {
            key: value for key, value in candidate["data"].get("diversity_signature", {}).items()
            if key not in cosmetic
        }
        return json.dumps(signature, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _quality_key(candidate: dict) -> tuple:
        components = candidate["data"].get("quality_components", {})
        return tuple(float(components.get(key, 0) or 0) for key in ("discriminating_value", "mechanistic_information_potential", "option_value", "frontier_closure_potential"))

    def _dead_screen(self, project_id: str, candidate: dict, negatives: list[dict]) -> dict:
        signature = candidate["data"]["diversity_signature"]
        for negative in negatives:
            dead_signature = negative["data"].get("diversity_signature") or negative["data"].get("scientific_dimensions") or {}
            if dead_signature and all(signature.get(key) == value for key, value in dead_signature.items()):
                return {"status": "DEAD_EQUIVALENT", "negative_result_id": str(negative["id"])}
        return {"status": "CLEAR"}

    def synthesize(self, round_id: str, literature_available: bool = False) -> dict:
        staleness = self.stale_check(round_id, mark_stale=True)
        if staleness["stale"]:
            raise GPUError(
                "DISCOVERY_ROUND_STALE",
                "Scientific state changed after independent generation; preserve this round for history and start a fresh one.",
            )
        round_ = self._round(round_id)
        if round_["data"].get("phase") not in {"GENERATION_FROZEN", "CHARACTERIZATION", "DEAD_MEMORY_SCREEN", "QD_ARCHIVE", "LITERATURE_PASS", "SYNTHESIS", "COMPLETED"}:
            raise GPUError("DISCOVERY_SYNTHESIS_BEFORE_FREEZE", round_id)
        now = self._now()
        with self.store._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"dde-synthesis:{round_id}",))
            cur.execute("SELECT * FROM discovery_round_syntheses WHERE discovery_round_id=%s FOR UPDATE", (round_id,))
            existing = cur.fetchone()
            if existing:
                return self.archive_get(str(existing["archive_id"]))
            cur.execute("SELECT count(*) FILTER (WHERE status='JOINED') AS pending FROM discovery_round_memberships WHERE discovery_round_id=%s", (round_id,))
            if int(cur.fetchone()["pending"]):
                raise GPUError("DISCOVERY_SYNTHESIS_BEFORE_FREEZE", round_id)
            batches = self.store.objects_list(round_["project_id"], "CandidateBatch", limit=None, data_filters={"discovery_round_id": round_id})
            candidates = [self.store.object_get(candidate_id) for batch in batches for candidate_id in batch["data"].get("candidate_ids", [])]
            negatives = self.store.objects_list(round_["project_id"], "NegativeResult", limit=None)
            screened = [(candidate, self._dead_screen(round_["project_id"], candidate, negatives)) for candidate in candidates]
            live = [candidate for candidate, screen in screened if screen["status"] == "CLEAR"]
            grouped: dict[str, list[dict]] = defaultdict(list)
            for candidate in live:
                grouped[str(candidate["data"]["mechanistic_niche"])].append(candidate)
            niche_ids: dict[str, str] = {}
            for niche, group in grouped.items():
                niche_id = str(uuid.uuid4())
                niche_ids[niche] = niche_id
                niche_data = {
                    "implementation_version": DDE_VERSION, "discovery_round_id": str(round_id),
                    "name": niche, "candidate_ids": [str(candidate["id"]) for candidate in group],
                    "representative_signature": group[0]["data"]["diversity_signature"],
                    "assignment_policy": "STRUCTURED_SIGNATURE_NOT_EMBEDDING_CLUSTER", "created_at": now.isoformat(),
                }
                cur.execute(
                    "INSERT INTO research_objects(id,project_id,kind,status,data,created_at) "
                    "VALUES(%s,%s,'MechanisticNiche','ACTIVE',%s,%s)",
                    (niche_id, round_["project_id"], self._json(niche_data), now),
                )
                self.store._event(cur, round_["project_id"], "NICHE_ASSIGNED", niche_id, {"round_id": str(round_id), "name": niche, "candidate_count": len(group)})
            survivors: list[dict] = []
            duplicate_ids: list[str] = []
            for group in grouped.values():
                by_signature: dict[str, list[dict]] = defaultdict(list)
                for candidate in group:
                    by_signature[self._scientific_equivalence_key(candidate)].append(candidate)
                for equivalent in by_signature.values():
                    equivalent.sort(key=self._quality_key, reverse=True)
                    survivors.append(equivalent[0])
                    duplicate_ids.extend(str(item["id"]) for item in equivalent[1:])
            survivors.sort(key=self._quality_key, reverse=True)
            distances = Counter(item["data"]["scientific_distance"] for item in candidates)
            niches = Counter(item["data"]["mechanistic_niche"] for item in live)
            required = round_["data"]["required_distance_coverage"]
            attempted = {distance: int(distances.get(distance, 0)) for distance in required}
            unfilled = [distance for distance, required_count in required.items() if required_count and not attempted[distance]]
            health_flags: list[str] = []
            if not niches:
                health_flags.append("LOW_NICHE_COVERAGE")
            if unfilled:
                health_flags.append("DISTANCE_COVERAGE_INCOMPLETE")
            if len({item["data"].get("architecture_lineage") for item in live}) <= 1 and len(live) > 1:
                health_flags.append("SAME_LINEAGE_COLLAPSE")
            if len(survivors) < len(live) and len({self._scientific_equivalence_key(item) for item in live}) < len(live):
                health_flags.append("GENERATOR_CORRELATION_HIGH")
            if not literature_available:
                health_flags.append("LITERATURE_UNVERIFIED")
            if not live and candidates:
                health_flags.append("DEAD_MEMORY_SATURATION")
            health = "HEALTHY" if not health_flags else health_flags[0]
            coverage = {
                "generated_candidate_count": len(candidates), "candidate_batch_count": len(batches),
                "independent_generator_count": len({item["data"]["worker_session_id"] for item in candidates if item["data"].get("independent_generation")}),
                "generation_operator_count": len({item["data"]["generation_operator"] for item in candidates}),
                "peer_isolation_used": round_["data"].get("independent_generation", True),
                "effective_niche_count": len(niches), "niche_distribution": dict(niches),
                "scientific_distance_distribution": dict(distances), "required_distance_slots": required,
                "filled_distance_slots": attempted, "unfilled_distance_slots": unfilled,
                "architecture_lineage_distribution": dict(Counter(str(item["data"].get("architecture_lineage") or "UNSPECIFIED") for item in candidates)),
                "representation_family_distribution": dict(Counter(str(item["data"]["diversity_signature"].get("representation") or "UNSPECIFIED") for item in candidates)),
                "literature_status": "AVAILABLE" if literature_available else "UNAVAILABLE_NOVELTY_UNVERIFIED",
                "novelty_verified_count": 0, "dead_equivalent_count": len(candidates) - len(live),
                "archive_survivor_count": len(survivors), "portfolio_health": health,
                "portfolio_health_flags": health_flags,
                "causal_object_distribution": dict(Counter(str(item["data"]["diversity_signature"].get("causal_object") or "UNSPECIFIED") for item in candidates)),
            }
            archive_id = str(uuid.uuid4())
            archive = {"implementation_version": DDE_VERSION, "discovery_round_id": str(round_id), "mechanistic_niche_ids": niche_ids, "survivor_candidate_ids": [str(item["id"]) for item in survivors], "duplicate_candidate_ids": duplicate_ids, "dead_memory_screen": [{"candidate_id": str(candidate["id"]), **screen} for candidate, screen in screened], "coverage": coverage, "selected_candidate_id": str(survivors[0]["id"]) if survivors else None, "runner_up_candidate_id": str(survivors[1]["id"]) if len(survivors) > 1 else None, "created_at": now.isoformat(), "not_evidence": True}
            cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'CrossWorkerQDArchive','ARCHIVED',%s,%s)", (archive_id, round_["project_id"], self._json(archive), now))
            self.store._event(cur, round_["project_id"], "QD_ARCHIVE_CREATED", archive_id, json.loads(self._json({"round_id": round_id, "survivor_count": len(survivors), "portfolio_health": health})))
            coverage_id = str(uuid.uuid4())
            cur.execute("INSERT INTO research_objects(id,project_id,kind,status,data,created_at) VALUES(%s,%s,'DiscoveryCoverageAtDecision','COMPLETED',%s,%s)", (coverage_id, round_["project_id"], self._json({**coverage, "discovery_round_id": round_id, "archive_id": archive_id, "created_at": now.isoformat()}), now))
            self.store._event(cur, round_["project_id"], "DISCOVERY_SYNTHESIS_COMPLETED", round_id, json.loads(self._json({"archive_id": archive_id, "coverage_id": coverage_id, "portfolio_health": health})))
            cur.execute("INSERT INTO discovery_round_syntheses(discovery_round_id,archive_id,created_at) VALUES(%s,%s,%s)", (round_id, archive_id, now))
            data = {**round_["data"], "phase": "COMPLETED", "completed_at": now.isoformat(), "literature_status": coverage["literature_status"], "archive_id": archive_id, "coverage_id": coverage_id}
            cur.execute("UPDATE research_objects SET status='COMPLETED',data=%s WHERE id=%s", (self._json(data), round_id))
        return self.archive_get(archive_id)

    def archive_get(self, archive_id: str) -> dict:
        archive = self.store.object_get(archive_id)
        if archive["kind"] != "CrossWorkerQDArchive":
            raise GPUError("DISCOVERY_ARCHIVE_NOT_FOUND", archive_id)
        return {**archive, "survivors": [self.store.object_get(candidate_id) for candidate_id in archive["data"].get("survivor_candidate_ids", [])]}

    def outcome_get(self, candidate_id: str) -> dict:
        candidate_id = str(candidate_id)
        candidate = self.store.object_get(candidate_id)
        if candidate["kind"] != "DiscoveryCandidate":
            raise GPUError("DISCOVERY_CANDIDATE_NOT_FOUND", candidate_id)
        outcomes = self.store.objects_list(candidate["project_id"], "DiscoveryCandidateOutcome", limit=None, data_filters={"candidate_id": candidate_id})
        return outcomes[0] if outcomes else {"candidate_id": candidate_id, "resolution_status": "UNKNOWN", "reason": "UNEXECUTED_OR_UNASSESSED_CANDIDATE"}

    def outcome_record(self, candidate_id: str, outcome: dict[str, Any]) -> dict:
        candidate_id = str(candidate_id)
        candidate = self.store.object_get(candidate_id)
        if candidate["kind"] != "DiscoveryCandidate":
            raise GPUError("DISCOVERY_CANDIDATE_NOT_FOUND", candidate_id)
        resolution = str(outcome.get("resolution_status", "UNKNOWN")).upper()
        allowed = {"UNKNOWN", "TESTED", "SUPPORTED", "WEAKENED", "REFUTED", "IMPLEMENTATION_FAILURE", "EXPERIMENT_DESIGN_FAILURE", "TRANSFER_FAILURE"}
        if resolution not in allowed:
            raise GPUError("DISCOVERY_OUTCOME_STATUS_INVALID", resolution)
        data = {"candidate_id": candidate_id, "discovery_round_id": candidate["data"]["discovery_round_id"], "resolution_status": resolution, "first_tested_at": outcome.get("first_tested_at"), "first_decisive_evidence_at": outcome.get("first_decisive_evidence_at"), "current_hypothesis_status": outcome.get("current_hypothesis_status"), "evidence_family_count": int(outcome.get("evidence_family_count", 0) or 0), "independent_origin_count": int(outcome.get("independent_origin_count", 0) or 0), "caused_world_model_change": bool(outcome.get("caused_world_model_change", False)), "caused_agenda_change": bool(outcome.get("caused_agenda_change", False)), "spawned_descendant_count": int(outcome.get("spawned_descendant_count", 0) or 0), "spawned_new_niche": bool(outcome.get("spawned_new_niche", False)), "produced_breakthrough_signal": bool(outcome.get("produced_breakthrough_signal", False)), "frontier_gap_change": outcome.get("frontier_gap_change"), "architecture_lineage_impact": outcome.get("architecture_lineage_impact"), "generalized": bool(outcome.get("generalized", False)), "later_refuted": bool(outcome.get("later_refuted", False)), "revisit_count": int(outcome.get("revisit_count", 0) or 0), "retrospective_discovery_value_components": dict(outcome.get("retrospective_discovery_value_components") or {}), "assessed_at": self._now().isoformat()}
        result = self.store.object_create(str(candidate["project_id"]), "DiscoveryCandidateOutcome", data, "DISCOVERY_CANDIDATE_OUTCOME_UPDATED", "COMPLETED")
        return result
