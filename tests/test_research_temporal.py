from __future__ import annotations

import os
import time
import uuid
from datetime import UTC, datetime

import psycopg
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
def test_revision_and_event_are_invisible_before_their_transaction_commits():
    store = ResearchStore(TEST_DATABASE_URL)
    project = store.project_create(f"commit-time-{time.time_ns()}", "When is state visible?")
    hypothesis = store.object_create(
        project["project_id"],
        "Hypothesis",
        {"mechanism": "committed old state"},
        "HYPOTHESIS_CREATED",
    )
    event_id = str(uuid.uuid4())
    with psycopg.connect(TEST_DATABASE_URL) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE research_objects SET status='WEAKENED',data=%s::jsonb WHERE id=%s",
            ('{"mechanism":"not committed at cutoff"}', hypothesis["id"]),
        )
        cursor.execute(
            "INSERT INTO research_events(id,project_id,event_type,subject_id,payload,created_at) "
            "VALUES(%s,%s,'FUTURE_TRANSACTION_EVENT',%s,'{}'::jsonb,clock_timestamp())",
            (event_id, project["project_id"], hypothesis["id"]),
        )
        cutoff_before_commit = datetime.now(UTC)
        time.sleep(0.01)

    historical = store.object_get(hypothesis["id"], as_of=cutoff_before_commit)
    event_types = {
        event["event_type"]
        for event in store.events(project["project_id"], as_of=cutoff_before_commit)
    }

    assert historical["status"] == "ACTIVE"
    assert historical["data"]["mechanism"] == "committed old state"
    assert "FUTURE_TRANSACTION_EVENT" not in event_types


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
def test_migration_stamps_a_legacy_embedding_at_the_migration_boundary():
    store = ResearchStore(TEST_DATABASE_URL)
    if not store.vector_available:
        pytest.skip("pgvector is unavailable")
    project = store.project_create(
        f"legacy-vector-{time.time_ns()}", "When did the legacy vector become visible?"
    )
    hypothesis = store.object_create(
        project["project_id"],
        "Hypothesis",
        {"mechanism": "legacy bounded vector"},
        "HYPOTHESIS_CREATED",
    )
    store.embedding_set(hypothesis["id"], [1.0, 0.0])
    with psycopg.connect(TEST_DATABASE_URL) as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE research_objects SET embedding_updated_at=NULL WHERE id=%s",
            (hypothesis["id"],),
        )
        cursor.execute(
            "DELETE FROM research_object_versions WHERE object_id=%s", (hypothesis["id"],)
        )

    before_migration = datetime.now(UTC)
    time.sleep(0.01)
    store._migrate()
    after_migration = datetime.now(UTC)

    with psycopg.connect(TEST_DATABASE_URL) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT embedding_updated_at,legacy_backfill FROM research_object_versions "
            "WHERE object_id=%s",
            (hypothesis["id"],),
        )
        revision = cursor.fetchone()
    assert revision[0] is not None
    assert revision[1] is True
    assert (
        store.semantic_search(
            project["project_id"], [1.0, 0.0], as_of=before_migration
        )
        == []
    )
    assert [
        str(item["id"])
        for item in store.semantic_search(
            project["project_id"], [1.0, 0.0], as_of=after_migration
        )
    ] == [hypothesis["id"]]


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="GPU_LAB_TEST_DATABASE_URL is not configured")
def test_upgrade_places_preexisting_unfinalized_history_at_a_conservative_boundary():
    store = ResearchStore(TEST_DATABASE_URL)
    project = store.project_create(f"upgrade-{time.time_ns()}", "Can old history leak?")
    hypothesis = store.object_create(
        project["project_id"],
        "Hypothesis",
        {"mechanism": "pre-upgrade state"},
        "HYPOTHESIS_CREATED",
    )
    with psycopg.connect(TEST_DATABASE_URL) as connection, connection.cursor() as cursor:
        for table in (
            "research_object_versions",
            "research_project_versions",
            "research_events",
        ):
            cursor.execute(
                f"ALTER TABLE {table} ALTER COLUMN commit_token DROP NOT NULL"
            )
        cursor.execute(
            "UPDATE research_object_versions SET committed_at=NULL,commit_token=NULL "
            "WHERE object_id=%s",
            (hypothesis["id"],),
        )
        cursor.execute(
            "UPDATE research_project_versions SET committed_at=NULL,commit_token=NULL "
            "WHERE project_id=%s",
            (project["project_id"],),
        )
        cursor.execute(
            "UPDATE research_events SET committed_at=NULL,commit_token=NULL "
            "WHERE project_id=%s",
            (project["project_id"],),
        )
    before_upgrade = datetime.now(UTC)
    time.sleep(0.01)

    store._migrate()
    after_upgrade = datetime.now(UTC)

    with pytest.raises(GPUError) as error:
        store.object_get(hypothesis["id"], as_of=before_upgrade)
    assert error.value.error_type == "TEMPORAL_OBJECT_NOT_VISIBLE"
    visible = store.object_get(hypothesis["id"], as_of=after_upgrade)
    assert visible["legacy_backfill"] is True
    assert visible["data"]["mechanism"] == "pre-upgrade state"
    assert any(
        event["event_type"] == "HYPOTHESIS_CREATED"
        for event in store.events(project["project_id"], as_of=after_upgrade)
    )


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
