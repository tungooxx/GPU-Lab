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


def test_paused_autonomy_never_starts_campaign():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    controller.config_update("project", {"paused": True})
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "INVALID", "action_type": "TRAINING_RUN"}, "FIXTURE", "RESULT_INSPECTED")

    assert controller.run_once("project")["decision"] == "PAUSED"


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


def test_model_change_runs_compatibility_instead_of_mutating_policy():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    controller.model_change_detect("project", "openai", "new-model")

    result = controller.run_once("project")

    assert result["decision"] == "COMPATIBILITY_EVALUATED"
    opportunity = store.objects_list("project", "ImprovementOpportunity")[0]
    assert opportunity["data"]["compatibility_experiment_id"] == "compatibility"


def test_meta_state_exposes_candidates_and_provider_compatibility():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    store.object_create("project", "ResearchPolicyPatch", {}, "FIXTURE", "SUPPORTED_ON_BENCHMARK")
    store.object_create("project", "PolicyExperiment", {"benchmark_version": "provider-compatibility-v3", "provider": "openai"}, "FIXTURE", "CROSS_MODEL_UNVERIFIED")

    state = controller.state_get("project")

    assert len(state["policy_candidates"]) == 1
    assert state["model_provider_compatibility"][0]["data"]["provider"] == "openai"


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
    assert store.objects_list("project", "MetaWorldModel")
    assert store.objects_list("project", "PolicyExperiment")
    run = store.objects_list("project", "ImprovementRun")[0]
    assert run["data"]["meta_campaign"]["candidate_sources"]["MetaWorldModel"]
    assert run["data"]["auto_promotion_preflight"]["policy_experiment_id"]
    assert store.objects_list("project", "MetaResearchCampaign")[0]["status"] == "COMPLETED"


def test_active_campaign_claim_prevents_duplicate_retry_after_interruption():
    store = Store()
    controller = MetaResearchController(store, PolicyLab())
    for _ in range(2):
        store.object_create("project", "ResearchDecisionOutcome", {"label": "LOW_VALUE", "action_type": "DIAGNOSTIC"}, "FIXTURE", "RESULT_INSPECTED")
    opportunity = controller.detect_opportunities("project")[0]
    campaign = store.object_create("project", "MetaResearchCampaign", {"fingerprint": f"meta-campaign:{opportunity['id']}"}, "FIXTURE", "RUNNING")

    result = controller.run_once("project")

    assert result["decision"] == "CAMPAIGN_IN_PROGRESS"
    assert result["campaign"]["id"] == campaign["id"]
