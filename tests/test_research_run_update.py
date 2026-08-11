from contextlib import contextmanager

from gpu_lab.research import ResearchStore


class _Cursor:
    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, statement, _params=None):
        self.statements.append(statement)

    def fetchone(self):
        return {
            "id": "run-id",
            "project_id": "project-id",
            "kind": "ExperimentRun",
            "status": "RESULT_INSPECTED",
            "data": {"result": "already assessed"},
        }


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self._cursor


def test_run_update_locks_before_preserving_concurrently_inspected_result():
    store = object.__new__(ResearchStore)
    cursor = _Cursor()

    @contextmanager
    def connect():
        yield _Connection(cursor)

    store._connect = connect

    result = store.run_update("run-id", {"status": "running"})

    assert result["status"] == "RESULT_INSPECTED"
    assert result["already_final"] is True
    assert "FOR UPDATE" in cursor.statements[0]
    assert all(not statement.startswith("UPDATE") for statement in cursor.statements)
