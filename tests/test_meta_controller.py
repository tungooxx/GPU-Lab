from __future__ import annotations

import uuid
from pathlib import Path

from gpu_lab.brain_bench import ResearchBrainBench
from gpu_lab.meta_controller import MetaResearchController
from gpu_lab.policy_lab import PolicyLabService


class Store:
    def __init__(self):
        self.items: list[dict] = []

    def object_create(self, project_id, kind, data, event, status="ACTIVE"):
        item = {"id": str(uuid.uuid4()), "project_id": project_id, "kind": kind, "data": data, "status": status}
        self.items.append(item)
        return item

    def objects_list(self, project_id, kind, limit=None):
        values = [item for item in self.items if item["project_id"] == project_id and item["kind"] == kind]
        return values if limit is None else values[:limit]

    def object_get(self, object_id):
        return next(item for item in self.items if item["id"] == object_id)

    def object_update(self, object_id, update, status, event):
        item = next(item for item in self.items if item["id"] == object_id)
        item["data"] = {**item["data"], **update}
        item["status"] = status
        return item


class PolicyLab:
    def improve(self, project_id, **kwargs):
        return {"recommendation": "REJECT_OR_REVISE", "improvement_run": {"id": "run", "data": {"best_supported_patch_id": None}}}

    def rollback(self, project_id, policy_id):
        return {"id": policy_id, "status": "PRODUCTION"}

    def evaluate_provider_compatibility(self, project_id, provider, model):
        return {"id": "compatibility", "data": {"provider": provider, "model": model}}

    def provider_adapter_candidate(self, project_id, provider, model, compatibility_experiment_id):
        return {"id": "adapter-candidate", "data": {"provider": provider, "model": model}}


class CampaignPolicyLab(PolicyLab):
    def __init__(self, store):
        self.store = store
        self.kwargs = None

    def improve(self, project_id, **kwargs):
        self.kwargs = kwargs
        run = self.store.object_create(project_id, "ImprovementRun", {"best_supported_patch_id": None}, "FIXTURE", "COMPLETED")
        return {"recommendation": "REJECT_OR_REVISE", "improvement_run": run}


def test_recurring_bad_outcomes_create_one_durable_opportunity():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    found = controller.detect_opportunities("project")

    assert len(found) == 1
    assert found[0]["data"]["expected_value_of_improvement"] >= 0.4
    model = store.objects_list("project", "MetaWorldModel")[0]
    agenda = store.objects_list("project", "MetaResearchAgenda")[0]
    benchmark_gap = store.objects_list("project", "BenchmarkGap")[0]
    assert model["data"]["relationships"][0]["causal_status"] == "UNRESOLVED"
    assert agenda["data"]["opportunity_id"] == found[0]["id"]
    assert benchmark_gap["data"]["candidate_evaluation_eligibility"] == "FUTURE_BENCHMARK_ONLY"
    assert controller.detect_opportunities("project") == []


def test_campaign_persists_competing_noncausal_component_diagnosis_before_patch_generation():
    store = Store()
    lab = CampaignPolicyLab(store)
    controller = MetaResearchController(store, lab)
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "INVALID", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    result = controller.run_once("project")

    diagnoses = [item for item in store.objects_list("project", "MetaWorldModel") if item["data"].get("diagnostic_hypotheses")]
    assert result["decision"] == "CAMPAIGN_STARTED"
    assert len(diagnoses) == 1
    assert {item["component"] for item in diagnoses[0]["data"]["diagnostic_hypotheses"]} == {"candidate_generation", "ranking", "critic"}
    assert all(item["causal_status"] == "HYPOTHESIS_NOT_ESTABLISHED" for item in diagnoses[0]["data"]["relationships"])
    assert lab.kwargs["diagnostic_hypotheses"] == diagnoses[0]["data"]["diagnostic_hypotheses"]


def test_paused_autonomy_never_starts_campaign():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    controller.config_update("project", {"paused": True})
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "INVALID", "action_type": "TRAINING_RUN"}, "FIXTURE", "RESULT_INSPECTED")

    assert controller.run_once("project")["decision"] == "PAUSED"


def test_minor_meta_research_defers_to_a_critical_unresolved_domain_agenda_item():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    store.object_create("project", "AgendaItem", {"question": "Resolve critical mechanism", "importance": 1.0, "uncertainty": 0.9}, "FIXTURE", "OPEN")
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    result = controller.run_once("project")

    assert result["decision"] == "DEFER_TO_DOMAIN_SCIENCE"
    assert result["scheduling"]["highest_domain_science"]["question"] == "Resolve critical mechanism"
    agenda = store.objects_list("project", "MetaResearchAgenda")[0]
    assert agenda["data"]["scheduling_decision"]["decision"] == "DEFER_TO_DOMAIN_SCIENCE"


def test_severe_meta_failure_is_not_deferred_by_domain_science():
    store = Store()
    lab = CampaignPolicyLab(store)
    controller = MetaResearchController(store, lab)
    store.object_create("project", "AgendaItem", {"question": "Resolve critical mechanism", "importance": 1.0, "uncertainty": 0.9}, "FIXTURE", "OPEN")
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "INVALID", "action_type": "TRAINING_RUN"}, "FIXTURE", "RESULT_INSPECTED")

    result = controller.run_once("project")

    assert result["decision"] == "CAMPAIGN_STARTED"
    assert result["scheduling"]["severe_meta_failure"] is True


def test_campaign_records_bounded_budget_and_passes_limits_to_policy_lab():
    store = Store()
    lab = CampaignPolicyLab(store)
    controller = MetaResearchController(store, lab)
    controller.config_update("project", {"candidate_budget": 1, "max_revision_rounds": 0, "token_budget": 100})
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    result = controller.run_once("project")

    assert result["decision"] == "CAMPAIGN_STARTED"
    assert lab.kwargs["candidate_budget"] == 1
    run = store.objects_list("project", "ImprovementRun")[0]
    assert run["data"]["meta_campaign"]["budget"]["token_budget"] == 100


def test_auto_project_rolls_back_only_after_repeated_negative_hindsight():
    store = Store()
    controller = MetaResearchController(store, PolicyLab(), mode="AUTO_PROJECT")
    store.object_create("project", "ResearchPolicy", {"parent_policy_id": "parent", "post_promotion_hindsight": [{"observed_improvement": -0.1}, {"unexpected_failure": "scope regression"}]}, "FIXTURE", "PRODUCTION")

    regressions = controller.monitor_promotions("project")

    assert len(regressions) == 1
    assert regressions[0]["data"]["rollback_decision"] == "ROLLED_BACK"
    assert store.objects_list("project", "PolicyNegativeResult")
    assert store.objects_list("project", "MetaWorldModel")
    assert controller.monitor_promotions("project") == []


def test_policy_pin_prevents_automatic_rollback():
    store = Store()
    policy = store.object_create("project", "ResearchPolicy", {"parent_policy_id": "parent", "post_promotion_hindsight": [{"observed_improvement": -0.1}, {"unexpected_failure": "scope regression"}]}, "FIXTURE", "PRODUCTION")
    controller = MetaResearchController(store, PolicyLab(), mode="AUTO_PROJECT")
    controller.policy_pin("project", policy["id"])

    regression = controller.monitor_promotions("project")[0]

    assert regression["data"]["rollback_decision"] == "PENDING"


def test_user_feedback_requires_actual_outcome_evidence_before_campaign_candidate():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    recorded = controller.feedback_record("project", "Too many low-value diagnostics")

    assert recorded["opportunity"]["status"] == "PENDING_EVIDENCE_REVIEW"
    assert controller.feedback_validate("project", recorded["feedback"]["id"])["status"] == "PENDING_EVIDENCE_REVIEW"
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    validated = controller.feedback_validate("project", recorded["feedback"]["id"])

    assert validated["status"] == "CANDIDATE"
    assert validated["data"]["expected_value_of_improvement"] == 0.4


def test_ranker_readiness_refuses_sparse_observational_history():
    readiness = MetaResearchController(Store(), PolicyLab()).ranker_readiness("project")

    assert readiness["data"]["decision"] == "DO_NOT_TRAIN_POLICY_MODEL"
    assert "sufficient_eligible_decisions" in readiness["data"]["blockers"]


def test_model_change_creates_one_compatibility_opportunity():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())

    created = controller.model_change_detect("project", "openai", "new-model")

    assert created["data"]["required_evaluation"] == "COMPACT_COMPATIBILITY_BENCHMARK"
    assert store.objects_list("project", "MetaResearchAgenda")[0]["data"]["required_evaluation"] == "COMPACT_COMPATIBILITY_BENCHMARK"
    assert controller.model_change_detect("project", "openai", "new-model") is None


def test_literature_scout_extracts_candidate_only_policy_transfer_with_provenance():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    opportunity = store.object_create("project", "ImprovementOpportunity", {"target_component": "critic", "observed_failure": "critic misses", "expected_value_of_improvement": 0.5, "scope": "PROJECT"}, "FIXTURE", "CANDIDATE")
    request = store.object_create("project", "LiteratureScoutRequest", {"opportunity_id": opportunity["id"], "question": "How to improve critic behavior?"}, "FIXTURE", "PREPARED")
    evidence = store.object_create("project", "EvidenceUnit", {"excerpt": "A distinct critique procedure improved falsification."}, "FIXTURE", "CANDIDATE")

    transfers = controller.literature_scout_complete("project", request["id"], [evidence["id"]])

    assert transfers[0]["data"]["evidence_id"] == evidence["id"]
    assert transfers[0]["data"]["comparison"] == "NOVEL_CANDIDATE"
    assert transfers[0]["data"]["authority"] == "EVIDENCE_CANDIDATE_ONLY"
    assert controller._candidate_sources("project", opportunity)["LiteraturePolicyTransfer"] == [transfers[0]["id"]]
    assert controller.literature_scout_complete("project", request["id"], [evidence["id"]])[0]["id"] == transfers[0]["id"]


def test_model_change_runs_compatibility_instead_of_mutating_policy():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    controller.model_change_detect("project", "openai", "new-model")

    result = controller.run_once("project")

    assert result["decision"] == "COMPATIBILITY_EVALUATED"
    opportunity = store.objects_list("project", "ImprovementOpportunity")[0]
    assert opportunity["data"]["compatibility_experiment_id"] == "compatibility"
    assert opportunity["data"]["provider_adapter_candidate_id"] == "adapter-candidate"


def test_meta_state_exposes_candidates_and_provider_compatibility():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    store.object_create("project", "ResearchPolicyPatch", {}, "FIXTURE", "SUPPORTED_ON_BENCHMARK")
    store.object_create("project", "PolicyExperiment", {"benchmark_version": "provider-compatibility-v3", "provider": "openai"}, "FIXTURE", "CROSS_MODEL_UNVERIFIED")

    state = controller.state_get("project")

    assert len(state["policy_candidates"]) == 1
    assert state["model_provider_compatibility"][0]["data"]["provider"] == "openai"


def test_meta_state_reports_benchmark_composition_when_bench_available():
    store = Store()
    lab = PolicyLabService(store, ResearchBrainBench(Path(__file__).parents[1] / "research_bench"))

    health = MetaResearchController(store, lab).state_get("project")["benchmark_health"]

    assert health["episodes"] > 0
    assert health["domain_distribution"]
    assert health["split_distribution"]


def test_policy_health_report_aggregates_lineage_calibration_and_failure_views():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    parent = store.object_create("project", "ResearchPolicy", {"version": 1}, "FIXTURE", "ROLLED_BACK")
    current = store.object_create("project", "ResearchPolicy", {"version": 2, "parent_policy_id": parent["id"], "applicability": {"scope": "PROJECT"}, "policy_benchmark_prediction": 0.4, "post_promotion_hindsight": [{"observed_improvement": 0.2}], "known_failure_modes": ["scope regression"], "provenance": {"source_type": "POLICY_PATCH"}}, "FIXTURE", "PRODUCTION")
    store.object_create("project", "ProviderAdapterCandidate", {"provider": "openai"}, "FIXTURE", "CANDIDATE")
    store.object_create("project", "PolicyNegativeResult", {"failure_mode": "overfit"}, "FIXTURE", "REJECTED")

    report = controller.policy_health_report("project")

    assert report["current_production_policy"]["id"] == current["id"]
    assert [item["version"] for item in report["policy_lineage"]] == [2, 1]
    assert report["real_world_policy_calibration"]["mean_realized_improvement"] == 0.2
    assert report["model_adapters"][0]["data"]["provider"] == "openai"
    assert report["recent_rollbacks"][0]["id"] == parent["id"]
    assert report["known_policy_failure_modes"] == ["scope regression"]


def test_meta_research_roi_reports_observed_yield_and_keeps_causal_savings_unknown():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    store.object_create("project", "MetaResearchCampaign", {"budget": {"token_budget": 100, "candidate_budget": 2}}, "FIXTURE", "COMPLETED")
    store.object_create("project", "ResearchPolicyPatch", {}, "FIXTURE", "REJECTED")
    store.object_create("project", "ResearchPolicyPatch", {}, "FIXTURE", "SUPPORTED_ON_BENCHMARK")
    store.object_create("project", "PolicyExperiment", {}, "FIXTURE", "COMPLETED")
    store.object_create("project", "ResearchDecisionOutcome", {"label": "INVALID"}, "FIXTURE", "RESULT_INSPECTED")
    store.object_create("project", "ResearchDecisionOutcome", {"label": "ZERO_INFORMATION"}, "FIXTURE", "RESULT_INSPECTED")

    roi = controller.meta_research_roi("project")

    assert roi["meta_research_budget_ceiling"]["token_budget"] == 100.0
    assert roi["policy_candidate_rejection_rate"] == 0.5
    assert roi["invalid_experiment_rate"] == 0.5
    assert roi["zero_information_action_rate"] == 0.5
    assert roi["estimated_future_research_cost_avoided"] is None


def test_auto_project_runs_closed_meta_cycle_and_promotes_supported_patch():
    store = Store()
    lab = PolicyLabService(store, ResearchBrainBench(Path(__file__).parents[1] / "research_bench"))
    controller = MetaResearchController(store, lab, mode="AUTO_PROJECT")
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    result = controller.run_once("project")

    assert result["decision"] == "CAMPAIGN_STARTED"
    assert result["promoted_policy"]["status"] == "PRODUCTION"
    assert result["promotion_preflight"]["eligible"] is True
    assert "methods improve experiment_selection" in result["literature_request"]["data"]["question"]
    assert store.objects_list("project", "MetaWorldModel")
    assert store.objects_list("project", "PolicyExperiment")
    pattern = store.objects_list("project", "MetaStrategyPattern")[0]
    assert pattern["data"]["observed_effect"] == "BENCHMARK_SUPPORTED_AWAITING_REAL_WORLD_HINDSIGHT"
    run = store.objects_list("project", "ImprovementRun")[0]
    assert run["data"]["meta_campaign"]["candidate_sources"]["MetaWorldModel"]
    assert run["data"]["meta_campaign"]["candidate_sources"]["LiteratureScoutRequest"]
    assert run["data"]["auto_promotion_preflight"]["policy_experiment_id"]
    assert store.objects_list("project", "MetaResearchCampaign")[0]["status"] == "COMPLETED"


def test_end_to_end_autonomous_improvement_promotes_and_survives_positive_real_hindsight():
    store = Store()
    lab = PolicyLabService(store, ResearchBrainBench(Path(__file__).parents[1] / "research_bench"))
    controller = MetaResearchController(store, lab, mode="AUTO_PROJECT")
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    campaign = controller.run_once("project")
    promoted = campaign["promoted_policy"]
    lab.record_hindsight(promoted["id"], observed_improvement=0.2, observed_cost=1.0, decision_ids=["prospective-decision-1"])
    lab.record_hindsight(promoted["id"], observed_improvement=0.1, observed_cost=1.1, decision_ids=["prospective-decision-2"])

    assert campaign["decision"] == "CAMPAIGN_STARTED"
    assert promoted["status"] == "PRODUCTION"
    assert controller.monitor_promotions("project") == []
    assert len(store.objects_list("project", "PolicyHindsight")) == 2
    assert not store.objects_list("project", "PolicyRegression")


def test_end_to_end_autonomous_promotion_rolls_back_after_repeated_real_regression():
    store = Store()
    lab = PolicyLabService(store, ResearchBrainBench(Path(__file__).parents[1] / "research_bench"))
    controller = MetaResearchController(store, lab, mode="AUTO_PROJECT")
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")

    promoted = controller.run_once("project")["promoted_policy"]
    parent_id = promoted["data"]["parent_policy_id"]
    lab.record_hindsight(promoted["id"], observed_improvement=-0.2, observed_cost=1.2, unexpected_failure="invalid experiment rate increased")
    lab.record_hindsight(promoted["id"], observed_improvement=-0.1, observed_cost=1.1, unexpected_failure="invalid experiment rate increased")

    regressions = controller.monitor_promotions("project")

    assert regressions[0]["data"]["rollback_decision"] == "ROLLED_BACK"
    assert lab.ensure_production_policy("project")["id"] == parent_id
    assert promoted["status"] == "ROLLED_BACK"
    negatives = store.objects_list("project", "PolicyNegativeResult")
    assert any(item["data"].get("proposal") == promoted["id"] for item in negatives)


def test_active_campaign_resumes_after_interruption_without_creating_a_second_claim():
    store = Store()
    controller = MetaResearchController(store, CampaignPolicyLab(store))
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")
    opportunity = controller.detect_opportunities("project")[0]
    campaign = store.object_create("project", "MetaResearchCampaign", {"fingerprint": f"meta-campaign:{opportunity['id']}"}, "FIXTURE", "RUNNING")

    result = controller.run_once("project")

    assert result["decision"] == "CAMPAIGN_RESUMED"
    assert result["campaign"]["id"] == campaign["id"]
    assert len(store.objects_list("project", "MetaResearchCampaign")) == 1


def test_restart_recovers_completed_policy_lab_run_without_rerunning_it():
    store = Store()
    lab = CampaignPolicyLab(store)
    controller = MetaResearchController(store, lab)
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")
    opportunity = controller.detect_opportunities("project")[0]
    campaign = store.object_create("project", "MetaResearchCampaign", {"fingerprint": f"meta-campaign:{opportunity['id']}"}, "FIXTURE", "RUNNING")
    run = store.object_create("project", "ImprovementRun", {"recommendation": "REJECT_OR_REVISE", "source_context": {"MetaResearchCampaign": [campaign["id"]]}}, "FIXTURE", "COMPLETED")

    result = controller.run_once("project")

    assert result["decision"] == "CAMPAIGN_RECOVERED"
    assert result["improvement"]["improvement_run"]["id"] == run["id"]
    assert lab.kwargs is None


def test_domain_promotion_requires_matching_mode_and_cross_project_transfer():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    store.object_create("project", "ResearchPolicy", {}, "FIXTURE", "PRODUCTION")
    patch = store.object_create("project", "ResearchPolicyPatch", {"implementation_change": {"enabled": True}, "applicability": {"scope": "DOMAIN"}}, "FIXTURE", "SUPPORTED_ON_BENCHMARK")
    store.object_create("project", "PolicyExperiment", {"candidate_patch_id": patch["id"], "splits": {"held_out": ["episode"]}, "regressions": [], "transfer_classification": "CROSS_PROJECT_SUPPORTED"}, "FIXTURE", "CROSS_PROJECT_SUPPORTED")

    assert "autonomy_mode_must_be_AUTO_DOMAIN" in controller._promotion_preflight("project", patch["id"], controller.config_get("project")["data"])["reasons"]
    config = controller.config_update("project", {"mode": "AUTO_DOMAIN"})

    assert controller._promotion_preflight("project", patch["id"], config["data"])["eligible"] is True


def test_scheduler_prioritizes_open_meta_agenda_over_raw_opportunity_value():
    store = Store()
    lab = CampaignPolicyLab(store)
    controller = MetaResearchController(store, lab)
    low_priority = store.object_create("project", "ImprovementOpportunity", {"target_component": "ranking", "observed_failure": "ranking issue", "expected_value_of_improvement": 0.9, "scope": "PROJECT"}, "FIXTURE", "CANDIDATE")
    high_priority = store.object_create("project", "ImprovementOpportunity", {"target_component": "critic", "observed_failure": "critic issue", "expected_value_of_improvement": 0.4, "scope": "PROJECT"}, "FIXTURE", "CANDIDATE")
    store.object_create("project", "MetaResearchAgenda", {"opportunity_id": low_priority["id"], "priority": 0.1}, "FIXTURE", "OPEN")
    store.object_create("project", "MetaResearchAgenda", {"opportunity_id": high_priority["id"], "priority": 0.95}, "FIXTURE", "OPEN")

    controller.run_once("project")

    assert lab.kwargs["component"] == "critic"


def test_repeated_benchmark_overprediction_creates_calibration_opportunity():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    store.object_create("project", "ResearchPolicy", {"policy_benchmark_prediction": 0.8, "post_promotion_hindsight": [{"observed_improvement": 0.1}, {"observed_improvement": 0.2}]}, "FIXTURE", "PRODUCTION")

    opportunities = controller.monitor_calibration("project")

    assert len(opportunities) == 1
    assert opportunities[0]["data"]["source"] == "POLICY_HINDSIGHT"
    assert opportunities[0]["data"]["causal_status"] == "UNRESOLVED"
