import pytest

from gpu_lab.errors import GPUError
from gpu_lab.execution_validity import aggregate_episode_attestations, normalize_execution_attestation


def reset_name_error():
    return {
        "technical_status": "FAIL",
        "episode_state": "TECHNICAL_ERROR",
        "stages": {
            "runtime_initialized": True,
            "environment_reset": False,
            "initial_observation_received": False,
            "planner_called": False,
            "plan_parsed": False,
            "executor_started": False,
            "environment_actions_executed": 0,
            "scientific_eligibility_evaluated": False,
            "scientific_metric_evaluated": False,
        },
        "technical_errors": [
            {"stage": "environment_reset", "type": "NameError", "message": "name 'r' is not defined"}
        ],
    }


def test_replay_residual_v21_reset_errors_are_technical_not_inconclusive():
    summary = aggregate_episode_attestations([reset_name_error() for _ in range(32)])

    assert summary["technical_status"] == "FAIL"
    assert summary["technical_invalid"] is True
    assert summary["attempted_n"] == 32
    assert summary["technical_valid_n"] == 0
    assert summary["protocol_reached_n"] == 0
    assert summary["eligible_n"] is None
    assert summary["qualified_n"] is None
    assert summary["measured_n"] == 0
    assert summary["uniform_failures"] == [
        {"stage": "environment_reset", "type": "NameError", "message": "name 'r' is not defined", "count": 32}
    ]


def test_pass_attestation_cannot_hide_missing_measurement_stage():
    invalid = reset_name_error()
    invalid["technical_status"] = "PASS"
    invalid["technical_errors"] = []
    with pytest.raises(GPUError, match="PASS requires"):
        normalize_execution_attestation(invalid)
