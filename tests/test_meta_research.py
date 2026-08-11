import hashlib
import json
import uuid

from gpu_lab.meta_research import MetaResearchService


class FakeStore:
    def __init__(self):
        self.items = []

    def create(self, kind, status="ACTIVE", data=None):
        item = {
            "id": str(uuid.uuid4()),
            "project_id": "project",
            "kind": kind,
            "status": status,
            "data": data or {},
        }
        self.items.append(item)
        return item

    def state_get(self, project_id):
        assert project_id == "project"
        return {"objects": list(reversed(self.items))}

    def objects_list(self, project_id, kind=None, statuses=None, limit=100, data_filters=None):
        rows = [
            item
            for item in reversed(self.items)
            if item["project_id"] == project_id
            and (kind is None or item["kind"] == kind)
            and (not statuses or item["status"] in statuses)
            and all(item["data"].get(key) == value for key, value in (data_filters or {}).items())
        ]
        return rows if limit is None else rows[:limit]

    def meta_lesson_create(self, project_id, data):
        fingerprint = hashlib.sha256(
            json.dumps(data["basis"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        existing = next(
            (
                item
                for item in self.items
                if item["kind"] == "MetaLesson"
                and item["data"].get("basis_fingerprint") == fingerprint
            ),
            None,
        )
        if existing:
            return {**existing, "idempotent_replay": True}
        return self.create(
            "MetaLesson", "RESULT_INSPECTED", {**data, "basis_fingerprint": fingerprint}
        )


def populated_store():
    store = FakeStore()
    store.create("AgendaItem", "RESOLVED")
    store.create("Claim", "SUPPORTED")
    store.create("Claim", "REFUTED")
    store.create("Hypothesis", "REFUTED")
    store.create(
        "Hypothesis",
        "SURVIVES_INITIAL_TEST",
        {"similar_dead_hypothesis_ids": ["dead"]},
    )
    store.create("Contradiction", "RESOLVED")
    store.create("Anomaly", "RESOLVED")
    store.create("Reproduction", "PARTIAL")
    store.create(
        "NegativeResult", "ACTIVE", {"failed_assumption": "static correlation is causal"}
    )
    store.create(
        "NegativeResult", "ACTIVE", {"failed_assumption": "static correlation is causal"}
    )
    store.create(
        "ResearchDecision",
        "SELECTED",
        {
            "actual_information_gain": "HIGH",
            "dead_ideas_retrieved": ["dead"],
            "selected_action": {"action_type": "FROZEN_DIAGNOSTIC"},
            "hindsight_assessment": None,
        },
    )
    store.create(
        "ExperimentRun", "RESULT_INSPECTED", {"actual_gpu_hours": 0.5}
    )
    store.create("ExperimentRun", "RESULT_NOT_INSPECTED")
    store.create("ExperimentRun", "RUNNING")
    return store


def test_progress_tracks_scientific_learning_without_fake_probability():
    service = MetaResearchService(populated_store())

    progress = service.progress("project")

    assert progress["metrics"]["uncertainties_resolved"] == 1
    assert progress["metrics"]["claims_strengthened"] == 1
    assert progress["metrics"]["claims_falsified"] == 1
    assert progress["metrics"]["hypotheses_eliminated"] == 1
    assert progress["metrics"]["duplicate_ideas_avoided"] == 1
    assert progress["metrics"]["experiments_saved_by_negative_memory"] == 1
    assert progress["metrics"]["information_gained_per_gpu_hour"] == 6.0
    assert progress["warning"].endswith("not calibrated scientific probabilities.")


def test_meta_review_prioritizes_existing_work_and_rejects_campaign_prematurity():
    store = populated_store()
    service = MetaResearchService(store)

    review = service.meta_review("project")
    replay = service.meta_review("project")

    assert review["data"]["campaign_readiness"] == "DO_NOT_BUILD_YET"
    assert "fewer than five inspected experiments" in review["data"]["campaign_readiness_reasons"]
    assert review["data"]["uninspected_run_ids"]
    assert review["data"]["unfinished_run_ids"]
    assert review["data"]["incomplete_reproduction_ids"]
    assert review["data"]["repeated_failed_assumptions"] == ["static correlation is causal"]
    assert any("Inspect available" in item for item in review["data"]["recommendations"])
    assert replay["id"] == review["id"]
    assert replay["idempotent_replay"] is True
