from contextlib import contextmanager

import pytest

from gpu_lab.errors import GPUError
from gpu_lab.research import ResearchStore


class TransactionCursor:
    def __init__(self, *, fail_on_version=False):
        self.executions = []
        self.fail_on_version = fail_on_version

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, params=None):
        if self.fail_on_version and "'WorldModelVersion'" in statement:
            raise RuntimeError("forced version insert failure")
        self.executions.append((statement, params))


class TransactionConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self.persisted = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_args):
        self.rolled_back = exc_type is not None
        self.committed = exc_type is None
        self.persisted = [] if self.rolled_back else list(self._cursor.executions)
        return False

    def cursor(self):
        return self._cursor


def store_with_transaction(*, fail_on_version=False):
    store = object.__new__(ResearchStore)
    cursor = TransactionCursor(fail_on_version=fail_on_version)
    connection = TransactionConnection(cursor)

    @contextmanager
    def connect():
        with connection:
            yield connection

    store._connect = connect
    return store, connection, cursor


def test_world_model_create_atomic_links_version_and_events():
    store, connection, cursor = store_with_transaction()

    result = store.world_model_create_atomic("project-id", "model", "fixture")

    model, version = result["world_model"], result["version"]
    assert connection.committed is True
    assert connection.rolled_back is False
    assert model["data"]["version"] == 1
    assert model["data"]["current_version_id"] == version["id"]
    assert version["data"]["world_model_id"] == model["id"]
    assert version["data"]["parent_version_id"] is None
    events = [
        params[2]
        for statement, params in cursor.executions
        if statement.startswith("INSERT INTO research_events")
    ]
    assert events == [
        "WORLD_MODEL_CREATED",
        "WORLD_MODEL_VERSION_CREATED",
        "WORLD_MODEL_VERSION_ADVANCED",
    ]


def test_world_model_create_atomic_rolls_back_mid_transaction_failure():
    store, connection, _cursor = store_with_transaction(fail_on_version=True)

    with pytest.raises(RuntimeError, match="forced version insert failure"):
        store.world_model_create_atomic("project-id", "model", "fixture")

    assert connection.committed is False
    assert connection.rolled_back is True
    assert connection.persisted == []


def test_world_model_child_rejects_invalid_status_before_persistence():
    store = object.__new__(ResearchStore)

    with pytest.raises(GPUError, match="not-a-status") as error:
        store.world_model_child_create(
            "model",
            "Phenomenon",
            {},
            "PHENOMENON_CREATED",
            "node_ids",
            {},
            child_status="not-a-status",
        )

    assert error.value.error_type == "INVALID_RESEARCH_OBJECT_STATUS"


def test_causal_edge_update_rejects_invalid_status_before_persistence():
    store = object.__new__(ResearchStore)

    with pytest.raises(GPUError, match="not-a-status") as error:
        store.causal_edge_update_atomic("edge", {}, "not-a-status", {}, [], None)

    assert error.value.error_type == "INVALID_RESEARCH_OBJECT_STATUS"


@pytest.mark.parametrize(
    ("hypothesis_status", "agenda_status"),
    [("not-a-status", "ACTIVE"), ("SUPPORTED", "not-a-status")],
)
def test_result_assessment_rejects_invalid_status_before_persistence(
    hypothesis_status, agenda_status
):
    store = object.__new__(ResearchStore)

    with pytest.raises(GPUError, match="not-a-status") as error:
        store.result_assessment_apply(
            run_id="run",
            decision_id="decision",
            hypothesis_id="hypothesis",
            agenda_item_id="agenda",
            evidence_data={},
            hypothesis_transition=hypothesis_status,
            rationale="test",
            inspection={},
            agenda_status=agenda_status,
            actual_information_gain="LOW",
        )

    assert error.value.error_type == "INVALID_RESEARCH_OBJECT_STATUS"
