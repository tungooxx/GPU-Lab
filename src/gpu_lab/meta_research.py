import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

from .branches import INSPECTABLE_RUN_STATUSES, RECOVERABLE_RUN_STATUSES
from .research import ResearchStore, classify_decision_data, strategy_learning_eligibility

_INFORMATION_VALUE = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


class MetaResearchService:
    """Measure and reflect on research process without changing scientific truth."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def progress(self, project_id: str) -> dict[str, Any]:
        _objects, by_kind = self._snapshot(project_id)
        return self._progress(project_id, by_kind)

    def _progress(
        self, project_id: str, by_kind: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        decisions = by_kind["ResearchDecision"]
        runs = by_kind["ExperimentRun"]
        outcomes_by_decision = {
            str(item["data"].get("decision_id")): item
            for item in by_kind["ResearchDecisionOutcome"]
        }
        eligibility = {
            str(item["id"]): strategy_learning_eligibility(
                item, outcomes_by_decision.get(str(item["id"]))
            )
            for item in decisions
        }
        scientific_decisions = [
            item for item in decisions if classify_decision_data(item["data"])["decision_role"] == "SCIENTIFIC_ACTION"
        ]
        eligible_decisions = [item for item in decisions if eligibility[str(item["id"])] ["eligible"]]
        information_points = sum(
            _INFORMATION_VALUE.get(item["data"].get("actual_information_gain"), 0)
            for item in scientific_decisions
        )
        gpu_hours = sum(self._gpu_hours(item) for item in runs)
        metrics = {
            "uncertainties_resolved": self._status(by_kind["AgendaItem"], "RESOLVED"),
            "claims_strengthened": self._status(by_kind["Claim"], "SUPPORTED"),
            "claims_falsified": self._status(by_kind["Claim"], "REFUTED"),
            "hypotheses_eliminated": self._status(by_kind["Hypothesis"], "REFUTED"),
            "contradictions_resolved": self._status(by_kind["Contradiction"], "RESOLVED"),
            "anomalies_explained": self._status(by_kind["Anomaly"], "RESOLVED"),
            "reproductions_completed": self._status(by_kind["Reproduction"], "REPRODUCED"),
            "novel_mechanisms_surviving": sum(
                item["status"] in {"SURVIVES_INITIAL_TEST", "SUPPORTED"}
                for item in by_kind["Hypothesis"]
            ),
            "inspected_experiments": self._status(runs, "RESULT_INSPECTED"),
            "negative_results_preserved": len(by_kind["NegativeResult"]),
            "duplicate_ideas_avoided": sum(
                bool(item["data"].get("similar_dead_hypothesis_ids"))
                for item in by_kind["Hypothesis"]
            ),
            "experiments_saved_by_negative_memory": sum(
                bool(item["data"].get("dead_ideas_retrieved")) for item in decisions
            ),
            "comparative_lessons": len(by_kind["ComparativeLesson"]),
            "null_models_created": len(by_kind["NullModel"]),
            "null_models_eliminated": self._status(by_kind["NullModel"], "REFUTED"),
            "decision_outcomes_assessed": len(by_kind["ResearchDecisionOutcome"]),
            "high_value_decisions": sum(
                item["data"].get("label") == "HIGH_VALUE"
                for item in by_kind["ResearchDecisionOutcome"]
            ),
            "zero_information_decisions": sum(
                item["data"].get("label") == "ZERO_INFORMATION"
                for item in by_kind["ResearchDecisionOutcome"]
            ),
            "strategy_patterns": len(by_kind["ResearchStrategyPattern"]),
            "strategy_negative_transfer_records": sum(
                len(item["data"].get("counterexamples", []))
                for item in by_kind["ResearchStrategyPattern"]
            ),
            "decisions_recorded": len(decisions),
            "scientific_decisions_total": len(scientific_decisions),
            "scientific_cycles_closed": len(eligible_decisions),
            "scientific_cycles_open": max(len(scientific_decisions) - len(eligible_decisions), 0),
            "strategy_eligible_cycles": len(eligible_decisions),
            "administrative_decisions": sum(
                classify_decision_data(item["data"])["decision_role"] == "ADMINISTRATIVE_RECOVERY"
                for item in decisions
            ),
            "system_verification_decisions": sum(
                classify_decision_data(item["data"])["decision_role"] in {"SYSTEM_VERIFICATION", "PROVIDER_CONTRACT_TEST"}
                for item in decisions
            ),
            "decisions_with_hindsight": sum(
                bool(item["data"].get("hindsight_assessment")) for item in scientific_decisions
            ),
            "gpu_hours_recorded": round(gpu_hours, 6),
            "information_gain_points": information_points,
            "information_gained_per_gpu_hour": (
                round(information_points / gpu_hours, 6) if gpu_hours > 0 else None
            ),
        }
        return {
            "project_id": project_id,
            "metrics": metrics,
            "warning": "Counts and information labels are operational heuristics, not calibrated scientific probabilities.",
        }

    def meta_review(self, project_id: str) -> dict[str, Any]:
        objects, by_kind = self._snapshot(project_id)
        progress = self._progress(project_id, by_kind)
        runs = by_kind["ExperimentRun"]
        unfinished = [
            str(item["id"])
            for item in runs
            if item["status"] in RECOVERABLE_RUN_STATUSES
        ]
        uninspected = [
            str(item["id"])
            for item in runs
            if item["status"] in INSPECTABLE_RUN_STATUSES
        ]
        incomplete_reproductions = [
            str(item["id"])
            for item in by_kind["Reproduction"]
            if item["status"] != "REPRODUCED"
        ]
        failed_assumptions = Counter(
            str(item["data"].get("failed_assumption", "")).strip()
            for item in by_kind["NegativeResult"]
            if str(item["data"].get("failed_assumption", "")).strip()
        )
        action_patterns = Counter(
            str(item["data"].get("selected_action", {}).get("action_type", "UNKNOWN"))
            for item in by_kind["ResearchDecision"]
            if classify_decision_data(item["data"])["decision_role"] == "SCIENTIFIC_ACTION"
        )
        outcomes = by_kind["ResearchDecisionOutcome"]
        repeated_low_value = Counter(
            str(item["data"].get("action_type", "UNKNOWN"))
            for item in outcomes
            if item["data"].get("label")
            in {"LOW_VALUE", "ZERO_INFORMATION", "REDUNDANT", "PREMATURE"}
        )
        branches_without_comparison = []
        for branch in by_kind["ExperimentBranch"]:
            branch_id = str(branch["id"])
            inspected_nodes = sum(
                item["status"] == "RESULT_INSPECTED"
                and str(item["data"].get("branch_id")) == branch_id
                for item in by_kind["ExperimentNode"]
            )
            compared = any(
                str(item["data"].get("branch_id")) == branch_id
                for item in by_kind["ComparativeLesson"]
            )
            if inspected_nodes >= 2 and not compared:
                branches_without_comparison.append(branch_id)
        recommendations = []
        if uninspected:
            recommendations.append("Inspect available experiment results before starting new work.")
        if unfinished:
            recommendations.append("Recover unfinished experiment runs before submitting replacements.")
        if incomplete_reproductions:
            recommendations.append("Complete baseline reproduction before internal causal intervention.")
        if branches_without_comparison:
            recommendations.append("Create ComparativeLessons for completed branch alternatives.")
        repeated_dead = [key for key, count in failed_assumptions.items() if count >= 2]
        if repeated_dead:
            recommendations.append("Avoid descendants that inherit repeatedly failed assumptions.")
        repeated_low_action_types = [
            action for action, count in repeated_low_value.items() if count >= 2
        ]
        if repeated_low_action_types:
            recommendations.append(
                "Diminishing returns detected for: "
                + ", ".join(sorted(repeated_low_action_types))
                + ". Prefer a more discriminating action family when gates permit."
            )
        metrics = progress["metrics"]
        inspected = int(metrics["inspected_experiments"])
        decisions = int(metrics["scientific_decisions_total"])
        hindsight = int(metrics["decisions_with_hindsight"])
        campaign_reasons = []
        if inspected < 5:
            campaign_reasons.append("fewer than five inspected experiments")
        if decisions == 0:
            campaign_reasons.append("no research decisions recorded")
        elif hindsight / decisions < 0.8:
            campaign_reasons.append("scientific decision hindsight coverage below 80%")
        if metrics["scientific_cycles_closed"] == 0 and decisions:
            campaign_reasons.append("no closed scientific decision cycles")
        if metrics["information_gained_per_gpu_hour"] is None:
            campaign_reasons.append("information-per-GPU-hour is not measurable")
        campaign_readiness = "DO_NOT_BUILD_YET" if campaign_reasons else "EVALUATE_BOUNDED_PILOT"
        review_payload_fingerprint = hashlib.sha256(
            json.dumps(
                sorted(
                    (
                        str(item["id"]),
                        item["kind"],
                        item["status"],
                        item["data"],
                    )
                    for item in objects
                    if item["kind"] != "MetaLesson"
                ),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        basis = {
            "object_versions": sorted(
                (str(item["id"]), item["kind"], item["status"])
                for item in objects
                if item["kind"] != "MetaLesson"
            ),
            "review_payload_fingerprint": review_payload_fingerprint,
            "metrics": metrics,
        }
        lesson = self.store.meta_lesson_create(
            project_id,
            {
                "basis": basis,
                "metrics": metrics,
                "unfinished_run_ids": unfinished,
                "uninspected_run_ids": uninspected,
                "incomplete_reproduction_ids": incomplete_reproductions,
                "repeated_failed_assumptions": repeated_dead,
                "experiment_action_patterns": dict(action_patterns),
                "repeated_low_value_action_types": sorted(repeated_low_action_types),
                "strategy_pattern_ids": [
                    str(item["id"]) for item in by_kind["ResearchStrategyPattern"]
                ],
                "branches_without_comparison": branches_without_comparison,
                "recommendations": recommendations,
                "campaign_readiness": campaign_readiness,
                "campaign_readiness_reasons": campaign_reasons,
                "warning": "Meta-review improves research policy; it does not alter scientific claims.",
            },
        )
        return {**lesson, "progress": progress}

    def list_lessons(self, project_id: str) -> list[dict[str, Any]]:
        return self.store.objects_list(project_id, "MetaLesson", limit=None)

    def _snapshot(
        self, project_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        objects = self.store.state_get(project_id)["objects"]
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in objects:
            by_kind[item["kind"]].append(item)
        return objects, by_kind

    @staticmethod
    def _status(items: list[dict[str, Any]], status: str) -> int:
        return sum(item["status"] == status for item in items)

    @staticmethod
    def _gpu_hours(run: dict[str, Any]) -> float:
        value = run["data"].get("actual_gpu_hours")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value)
        return 0.0
