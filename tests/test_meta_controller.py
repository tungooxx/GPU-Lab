from __future__ import annotations

import uuid

from gpu_lab.meta_controller import MetaResearchController


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
