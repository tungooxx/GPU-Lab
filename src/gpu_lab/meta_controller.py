"""Bounded controller for durable meta-research campaigns."""

from __future__ import annotations

from typing import Any

from .errors import GPUError


class MetaResearchController:
    """Observe research behavior and launch at most one evidence-gated campaign."""

    def __init__(self, store, policy_lab, *, mode: str = "AUTO_PROJECT", candidate_budget: int = 3, benchmark_budget: int = 6, literature_budget: int = 1):
        self.store, self.policy_lab = store, policy_lab
        self.defaults = {
            "mode": mode,
            "candidate_budget": candidate_budget,
            "benchmark_budget": benchmark_budget,
            "literature_budget": literature_budget,
            "max_revision_rounds": 3,
            "token_budget": 12_000,
            "llm_call_budget": 12,
            "engineering_budget": 0,
            "wall_clock_iteration_budget": 3,
            "gpu_budget": 0.0,
        }

    def _objects(self, project_id: str, kind: str) -> list[dict[str, Any]]:
        return self.store.objects_list(project_id, kind, limit=None)

    def _find_by_fingerprint(self, project_id: str, kind: str, fingerprint: str) -> dict[str, Any] | None:
        return next(
            (item for item in self._objects(project_id, kind) if item["data"].get("fingerprint") == fingerprint),
            None,
        )

    def _candidate_sources(self, project_id: str, opportunity: dict[str, Any]) -> dict[str, list[str]]:
        """Return durable, reviewable inputs for policy invention.

        Sources are evidence and constraints, not executable instructions.  In
        particular rejected policy results are retained to avoid repeating a
        failed mechanism casually.
        """
        return {
            "ImprovementOpportunity": [str(opportunity["id"])],
            "MetaWorldModel": [str(item["id"]) for item in self._objects(project_id, "MetaWorldModel")[-5:]],
            "MetaLesson": [str(item["id"]) for item in self._objects(project_id, "MetaLesson")[-5:]],
            "ResearchStrategyPattern": [str(item["id"]) for item in self._objects(project_id, "ResearchStrategyPattern")[-5:]],
            "PolicyNegativeResult": [str(item["id"]) for item in self._objects(project_id, "PolicyNegativeResult")[-10:]],
        }

    def _ensure_meta_records(self, project_id: str, opportunity: dict[str, Any]) -> None:
        """Turn an observed weakness into explicit, non-causal meta-science records.

        The records deliberately retain counterexamples and unresolved causal links:
        repeated outcomes are a reason to investigate, never proof that a policy
        component caused them.
        """
        data = opportunity["data"]
        fingerprint = str(data["fingerprint"])
        if not self._find_by_fingerprint(project_id, "MetaWorldModel", fingerprint):
            self.store.object_create(
                project_id,
                "MetaWorldModel",
                {
                    "fingerprint": fingerprint,
                    "scope": data.get("scope", "PROJECT"),
                    "relationships": [{
                        "from": "research_decision_behavior",
                        "to": data["target_component"],
                        "observation": data["observed_failure"],
                        "causal_status": "UNRESOLVED",
                    }],
                    "evidence": data.get("supporting_evidence", []),
                    "confidence": data.get("confidence", 0.0),
                    "counterexamples": [],
                    "unresolved_relationships": [
                        "Observed association is not sufficient to assign component-level causality."
                    ],
                    "provider_sensitivity": "UNVERIFIED",
                    "domain_sensitivity": "UNVERIFIED",
                },
                "META_WORLD_MODEL_UPDATED",
                "ACTIVE",
            )
        if not self._find_by_fingerprint(project_id, "MetaResearchAgenda", fingerprint):
            self.store.object_create(
                project_id,
                "MetaResearchAgenda",
                {
                    "fingerprint": fingerprint,
                    "question": f"Why does {data['target_component']} exhibit: {data['observed_failure']}?",
                    "opportunity_id": str(opportunity["id"]),
                    "priority": data["expected_value_of_improvement"],
                    "scope": data.get("scope", "PROJECT"),
                    "status_reason": "Evidence-backed question awaiting bounded diagnosis.",
                    "required_evaluation": data.get("required_evaluation", "STANDARD_POLICY_BENCHMARK"),
                },
                "META_RESEARCH_AGENDA_CREATED",
                "OPEN",
            )
        if not self._find_by_fingerprint(project_id, "BenchmarkGap", fingerprint):
            # A failure that generated this opportunity is contaminated for the
            # current candidate.  It can seed a later benchmark episode only.
            self.store.object_create(
                project_id,
                "BenchmarkGap",
                {
                    "fingerprint": fingerprint,
                    "observed_failure": data["observed_failure"],
                    "supporting_evidence": data.get("supporting_evidence", []),
                    "scope": data.get("scope", "PROJECT"),
                    "candidate_evaluation_eligibility": "FUTURE_BENCHMARK_ONLY",
                    "excluded_from_improvement_opportunity_id": str(opportunity["id"]),
                    "reason": "Gap discovered from the same evidence that generated the policy candidate.",
                },
                "BENCHMARK_GAP_DISCOVERED",
                "CANDIDATE",
            )

    def config_get(self, project_id: str) -> dict[str, Any]:
        configs = self._objects(project_id, "PolicyAutonomyConfig")
        if configs:
            return configs[0]
        return self.store.object_create(project_id, "PolicyAutonomyConfig", {**self.defaults, "paused": False, "pinned_policy_id": None, "provenance": "v3-default"}, "POLICY_AUTONOMY_CONFIG_CREATED", "ACTIVE")

    def config_update(self, project_id: str, update: dict[str, Any]) -> dict[str, Any]:
        config = self.config_get(project_id)
        allowed = {
            "mode", "paused", "pinned_policy_id", "candidate_budget", "benchmark_budget",
            "literature_budget", "max_revision_rounds", "token_budget", "llm_call_budget",
            "engineering_budget", "wall_clock_iteration_budget", "gpu_budget",
        }
        if not isinstance(update, dict) or set(update) - allowed:
            raise GPUError("POLICY_AUTONOMY_CONFIG_INVALID", "unsupported autonomy setting")
        mode = str(update.get("mode", config["data"]["mode"])).upper()
        if mode not in {"ADVISORY", "AUTO_PROJECT", "AUTO_DOMAIN", "AUTO_GLOBAL"}:
            raise GPUError("POLICY_AUTONOMY_MODE_INVALID", mode)
        non_negative = {
            "candidate_budget", "benchmark_budget", "literature_budget", "max_revision_rounds",
            "token_budget", "llm_call_budget", "engineering_budget", "wall_clock_iteration_budget", "gpu_budget",
        }
        if any(not isinstance(update[key], (int, float)) or isinstance(update[key], bool) or update[key] < 0 for key in non_negative & set(update)):
            raise GPUError("POLICY_AUTONOMY_CONFIG_INVALID", "budgets must be non-negative numbers")
        return self.store.object_update(str(config["id"]), update, "ACTIVE", "POLICY_AUTONOMY_CONFIG_UPDATED")

    def detect_opportunities(self, project_id: str) -> list[dict[str, Any]]:
        outcomes = self._objects(project_id, "ResearchDecisionOutcome")
        grouped: dict[str, list[dict[str, Any]]] = {}
        for outcome in outcomes:
            label = outcome["data"].get("label")
            if label in {"LOW_VALUE", "ZERO_INFORMATION", "PREMATURE", "INVALID"}:
                grouped.setdefault(str(outcome["data"].get("action_type", "UNKNOWN")), []).append(outcome)
        existing = {item["data"].get("fingerprint") for item in self._objects(project_id, "ImprovementOpportunity")}
        opportunities = []
        for action, records in grouped.items():
            catastrophic = any(record["data"].get("label") == "INVALID" for record in records)
            if len(records) < 2 and not catastrophic:
                continue
            fingerprint = f"decision-outcome:{action}:{','.join(sorted(str(r['id']) for r in records))}"
            if fingerprint in existing:
                continue
            frequency, severity = len(records), sum(record["data"].get("label") in {"PREMATURE", "INVALID"} for record in records)
            expected_value = round((frequency + severity * 2) * 0.2, 3)
            opportunity = self.store.object_create(project_id, "ImprovementOpportunity", {"source": "DECISION_OUTCOME", "target_component": "experiment_selection", "observed_failure": f"Repeated {action} decisions have low or invalid information value.", "supporting_evidence": [str(r["id"]) for r in records], "frequency": frequency, "severity": severity, "scientific_cost": frequency, "compute_cost": 0.0, "confidence": min(0.95, 0.3 + frequency * 0.15), "estimated_fixability": 0.5, "expected_value_of_improvement": expected_value, "scope": "PROJECT", "fingerprint": fingerprint}, "IMPROVEMENT_OPPORTUNITY_CREATED", "CANDIDATE")
            self._ensure_meta_records(project_id, opportunity)
            opportunities.append(opportunity)
        return opportunities

    def run_once(self, project_id: str) -> dict[str, Any]:
        config = self.config_get(project_id)
        self.detect_opportunities(project_id)
        # Event-driven sources (for example a provider/model change) are
        # already durable opportunities.  Scheduler passes must consider them
        # alongside newly detected outcome patterns.
        opportunities = [
            item for item in self._objects(project_id, "ImprovementOpportunity")
            if item["status"] in {"CANDIDATE", "ACTIVE"}
        ]
        if config["data"].get("paused"):
            return {"decision": "PAUSED", "opportunities": opportunities}
        if not config["data"].get("candidate_budget") or not config["data"].get("benchmark_budget"):
            return {"decision": "BUDGET_EXHAUSTED", "opportunities": opportunities}
        viable = [item for item in opportunities if item["data"]["expected_value_of_improvement"] >= 0.4]
        if not viable:
            return {"decision": "NO_CAMPAIGN", "opportunities": opportunities}
        opportunity = max(viable, key=lambda item: item["data"]["expected_value_of_improvement"])
        if opportunity["data"].get("required_evaluation") == "COMPACT_COMPATIBILITY_BENCHMARK":
            compatibility = self.policy_lab.evaluate_provider_compatibility(
                project_id,
                str(opportunity["data"].get("provider", "GENERIC")),
                str(opportunity["data"].get("model", "unknown")),
            )
            self.store.object_update(
                str(opportunity["id"]),
                {"compatibility_experiment_id": str(compatibility["id"])},
                "COMPLETED",
                "POLICY_COMPATIBILITY_EVALUATED",
            )
            return {"decision": "COMPATIBILITY_EVALUATED", "opportunities": opportunities, "compatibility": compatibility}
        budget = {key: config["data"][key] for key in self.defaults if key != "mode"}
        source_context = self._candidate_sources(project_id, opportunity)
        result = self.policy_lab.improve(
            project_id,
            failure=opportunity["data"]["observed_failure"],
            component=opportunity["data"]["target_component"],
            candidate_budget=int(config["data"]["candidate_budget"]),
            max_revisions=int(config["data"]["max_revision_rounds"]),
            source_context=source_context,
        )
        run_id = str(result["improvement_run"]["id"])
        self.store.object_update(run_id, {"meta_campaign": {"trigger": str(opportunity["id"]), "target_component": opportunity["data"]["target_component"], "scope": opportunity["data"]["scope"], "budget": budget, "candidate_sources": source_context, "stop_conditions": ["candidate budget exhausted", "benchmark budget exhausted", "hard epistemic regression", "revision limit reached"]}}, "COMPLETED", "META_RESEARCH_BUDGET_RECORDED")
        self.store.object_update(str(opportunity["id"]), {"improvement_run_id": run_id}, "COMPLETED", "META_RESEARCH_STARTED")
        promoted = None
        best_patch = result["improvement_run"]["data"].get("best_supported_patch_id")
        if config["data"]["mode"] == "AUTO_PROJECT" and result["recommendation"] == "PROMOTE" and best_patch:
            promoted = self.policy_lab.promote(project_id, best_patch)
        return {"decision": "CAMPAIGN_STARTED", "opportunities": opportunities, "improvement": result, "promoted_policy": promoted}

    def monitor_promotions(self, project_id: str) -> list[dict[str, Any]]:
        """Detect severe, repeated post-promotion failures and rollback scoped policy."""
        config = self.config_get(project_id)
        regressions = []
        for policy in self._objects(project_id, "ResearchPolicy"):
            if policy["status"] != "PRODUCTION":
                continue
            hindsight = policy["data"].get("post_promotion_hindsight", [])
            failures = [item for item in hindsight if item.get("unexpected_failure") or (isinstance(item.get("observed_improvement"), (int, float)) and item["observed_improvement"] < 0)]
            if len(failures) < 2:
                continue
            regression = self.store.object_create(project_id, "PolicyRegression", {"policy_id": str(policy["id"]), "expected_behavior": "non-regressing scoped research behavior", "observed_behavior": "repeated negative post-promotion hindsight", "supporting_decisions": [], "severity": "HIGH", "confidence": min(0.95, 0.5 + len(failures) * 0.1), "affected_scope": "PROJECT", "rollback_decision": "PENDING", "revisit_condition": "new causal diagnosis required"}, "POLICY_REGRESSION_DETECTED", "CANDIDATE")
            parent = policy["data"].get("parent_policy_id")
            if config["data"]["mode"] == "AUTO_PROJECT" and parent and not config["data"].get("pinned_policy_id"):
                restored = self.policy_lab.rollback(project_id, str(parent))
                self.store.object_update(str(regression["id"]), {"rollback_decision": "ROLLED_BACK", "restored_policy_id": str(restored["id"])}, "COMPLETED", "POLICY_ROLLED_BACK")
            regressions.append(regression)
        return regressions

    def state_get(self, project_id: str) -> dict[str, Any]:
        """Compact durable view; never substitutes meta records for scientific truth."""
        policies = self._objects(project_id, "ResearchPolicy")
        production = next((item for item in policies if item["status"] == "PRODUCTION"), None)
        return {
            "production_policy_id": str(production["id"]) if production else None,
            "autonomy": self.config_get(project_id),
            "active_opportunities": [item for item in self._objects(project_id, "ImprovementOpportunity") if item["status"] in {"CANDIDATE", "ACTIVE"}],
            "active_agenda": [item for item in self._objects(project_id, "MetaResearchAgenda") if item["status"] in {"OPEN", "ACTIVE"}],
            "meta_world_models": self._objects(project_id, "MetaWorldModel")[-10:],
            "active_runs": [item for item in self._objects(project_id, "ImprovementRun") if item["status"] not in {"COMPLETED", "REJECTED"}],
            "recent_regressions": self._objects(project_id, "PolicyRegression")[-10:],
            "benchmark_health": {"policy_experiments": len(self._objects(project_id, "PolicyExperiment")), "benchmark_gaps": len(self._objects(project_id, "BenchmarkGap"))},
        }

    def model_change_detect(self, project_id: str, provider: str, model: str) -> dict[str, Any] | None:
        """Record provider/model drift as a bounded compatibility-evaluation opportunity."""
        fingerprint = f"model-change:{provider}:{model}"
        if self._find_by_fingerprint(project_id, "ImprovementOpportunity", fingerprint):
            return None
        opportunity = self.store.object_create(project_id, "ImprovementOpportunity", {"source": "MODEL_CHANGE", "target_component": "provider_adapter", "provider": provider, "model": model, "observed_failure": f"Provider/model changed to {provider}:{model}; policy compatibility is unverified.", "supporting_evidence": [], "frequency": 1, "severity": 1, "scientific_cost": 0.0, "compute_cost": 0.0, "confidence": 1.0, "estimated_fixability": 0.5, "expected_value_of_improvement": 0.4, "scope": "PROJECT", "fingerprint": fingerprint, "required_evaluation": "COMPACT_COMPATIBILITY_BENCHMARK"}, "IMPROVEMENT_OPPORTUNITY_CREATED", "CANDIDATE")
        self._ensure_meta_records(project_id, opportunity)
        return opportunity
