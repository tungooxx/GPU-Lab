from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import pytest

from gpu_lab.brain import ResearchBrain
from gpu_lab.errors import GPUError
from gpu_lab.research import ResearchStore

TEST_DATABASE_URL = os.getenv("GPU_LAB_TEST_DATABASE_URL")


def test_temporal_cutoff_requires_an_aware_timestamp():
    with pytest.raises(GPUError) as error:
        ResearchStore._normalize_as_of("2026-08-11T12:00:00")

    assert error.value.error_type == "INVALID_TEMPORAL_CUTOFF"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_future_objects_updates_events_and_lexical_hits_do_not_leak():
    store = ResearchStore(TEST_DATABASE_URL)
    project = store.project_create(f"temporal-{time.time_ns()}", "Can future records leak?")
    project_id = project["project_id"]
    before_object = datetime.now(UTC)
    time.sleep(0.01)
    hypothesis = store.object_create(
        project_id,
        "Hypothesis",
        {"mechanism": "visible mechanism"},
        "HYPOTHESIS_CREATED",
    )
    after_object = datetime.now(UTC)
    time.sleep(0.01)
    store.object_update(
        hypothesis["id"],
        {"mechanism": "future forbidden mechanism"},
        "WEAKENED",
        "HYPOTHESIS_WEAKENED",
    )
    after_update = datetime.now(UTC)

    assert store.objects_list(project_id, as_of=before_object) == []
    old = store.object_get(hypothesis["id"], as_of=after_object)
    assert old["status"] == "ACTIVE"
    assert old["data"]["mechanism"] == "visible mechanism"
    assert store.search(project_id, "future forbidden", as_of=after_object) == []
    assert all(
        event["event_type"] != "HYPOTHESIS_WEAKENED"
        for event in store.events(project_id, as_of=after_object)
    )
    current = store.object_get(hypothesis["id"], as_of=after_update)
    assert current["status"] == "WEAKENED"
    assert current["data"]["mechanism"] == "future forbidden mechanism"


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_embedding_created_after_cutoff_does_not_leak():
    store = ResearchStore(TEST_DATABASE_URL)
    if not store.vector_available:
        pytest.skip("pgvector is unavailable")
    project = store.project_create(f"temporal-vector-{time.time_ns()}", "Can vectors leak?")
    hypothesis = store.object_create(
        project["project_id"],
        "Hypothesis",
        {"mechanism": "bounded vector"},
        "HYPOTHESIS_CREATED",
    )
    before_embedding = datetime.now(UTC)
    time.sleep(0.01)
    store.embedding_set(hypothesis["id"], [1.0, 0.0])
    after_embedding = datetime.now(UTC)

    assert (
        store.semantic_search(
            project["project_id"], [1.0, 0.0], as_of=before_embedding
        )
        == []
    )
    hits = store.semantic_search(
        project["project_id"], [1.0, 0.0], as_of=after_embedding
    )
    assert [str(item["id"]) for item in hits] == [hypothesis["id"]]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_historical_brain_step_is_non_mutating_and_cannot_see_future_run():
    store = ResearchStore(TEST_DATABASE_URL)
    brain = ResearchBrain(store)
    project = store.project_create(f"temporal-brain-{time.time_ns()}", "Which test is next?")
    project_id = project["project_id"]
    brain.world_model_create(project_id, "Temporal model", "fixture")
    agenda = brain.agenda_create(project_id, "Temporal agenda")
    hypothesis = store.object_create(
        project_id,
        "Hypothesis",
        {"mechanism": "a visible mechanism"},
        "HYPOTHESIS_CREATED",
    )
    brain.agenda_item_create(
        agenda["id"],
        "Does the frozen diagnostic discriminate?",
        5,
        5,
        "fixture",
        [hypothesis["id"]],
        candidate_experiments=[{"action_type": "FROZEN_DIAGNOSTIC"}],
    )
    cutoff = datetime.now(UTC)
    time.sleep(0.01)
    store.object_create(
        project_id,
        "ExperimentRun",
        {"experiment_id": "future-run", "inspection": None},
        "EXPERIMENT_FINISHED",
        "completed",
    )

    before_count = store.objects_count(project_id, "ResearchDecision")
    result = brain.brain_step(project_id, cutoff.isoformat(), persist=False)

    assert result["selected_action"]["action_type"] == "FROZEN_DIAGNOSTIC"
    assert result["decision_id"] is None
    assert result["verification_status"] == "TEMPORAL_DRY_RUN"
    assert store.objects_count(project_id, "ResearchDecision") == before_count
