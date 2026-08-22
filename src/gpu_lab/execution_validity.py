"""Execution validity contracts kept separate from scientific outcomes.

The runner may report a scientifically negative trajectory only after it has
reached the preregistered measurement stage.  This module deliberately makes
runtime errors and missing stages technical facts rather than observations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .errors import GPUError


EPISODE_STATES = {
    "TECHNICAL_ERROR",
    "PROTOCOL_REACHED_NOT_ELIGIBLE",
    "ELIGIBLE",
    "QUALIFIED",
}
REQUIRED_STAGES = (
    "runtime_initialized",
    "environment_reset",
    "initial_observation_received",
    "planner_called",
    "plan_parsed",
    "executor_started",
    "scientific_eligibility_evaluated",
    "scientific_metric_evaluated",
)
PROTOCOL_STAGES = REQUIRED_STAGES[:6]


def normalize_execution_attestation(attestation: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one runner attestation without trusting its label."""
    if not isinstance(attestation, dict):
        raise GPUError("INVALID_EXECUTION_ATTESTATION", "Attestation must be an object")
    stages = attestation.get("stages")
    errors = attestation.get("technical_errors", [])
    if not isinstance(stages, dict) or not isinstance(errors, list):
        raise GPUError("INVALID_EXECUTION_ATTESTATION", "stages and technical_errors are required")
    normalized_stages: dict[str, bool | int] = {}
    for stage in REQUIRED_STAGES:
        value = stages.get(stage, False)
        if not isinstance(value, bool):
            raise GPUError("INVALID_EXECUTION_ATTESTATION", f"{stage} must be boolean")
        normalized_stages[stage] = value
    actions = stages.get("environment_actions_executed", 0)
    if not isinstance(actions, int) or isinstance(actions, bool) or actions < 0:
        raise GPUError("INVALID_EXECUTION_ATTESTATION", "environment_actions_executed must be >= 0")
    normalized_stages["environment_actions_executed"] = actions
    normalized_errors = []
    for error in errors:
        if not isinstance(error, dict) or not all(
            isinstance(error.get(key), str) and error[key].strip()
            for key in ("stage", "type", "message")
        ):
            raise GPUError("INVALID_EXECUTION_ATTESTATION", "Each technical error needs stage, type, message")
        normalized_errors.append(
            {key: error[key].strip() for key in ("stage", "type", "message")}
        )
    stated = attestation.get("technical_status")
    if stated not in {"PASS", "FAIL"}:
        raise GPUError("INVALID_EXECUTION_ATTESTATION", "technical_status must be PASS or FAIL")
    state = attestation.get("episode_state")
    if state is not None and state not in EPISODE_STATES:
        raise GPUError("INVALID_EPISODE_EXECUTION_STATE", str(state))
    complete_measurement = all(normalized_stages[stage] for stage in REQUIRED_STAGES)
    technical_valid = stated == "PASS" and not normalized_errors and complete_measurement
    if stated == "PASS" and not technical_valid:
        raise GPUError(
            "EXECUTION_ATTESTATION_CONTRADICTION",
            "PASS requires all scientific stages and no technical errors",
        )
    if stated == "FAIL" and not normalized_errors:
        raise GPUError("EXECUTION_ATTESTATION_CONTRADICTION", "FAIL requires a technical error")
    if state == "TECHNICAL_ERROR" and not normalized_errors:
        raise GPUError("EXECUTION_ATTESTATION_CONTRADICTION", "TECHNICAL_ERROR requires an error")
    if technical_valid and state == "TECHNICAL_ERROR":
        raise GPUError("EXECUTION_ATTESTATION_CONTRADICTION", "Valid execution cannot be technical error")
    return {
        "schema_version": 1,
        "technical_status": "PASS" if technical_valid else "FAIL",
        "technical_valid": technical_valid,
        "measurement_reached": complete_measurement,
        "protocol_reached": all(normalized_stages[stage] for stage in PROTOCOL_STAGES),
        "episode_state": state or ("TECHNICAL_ERROR" if not technical_valid else "QUALIFIED"),
        "stages": normalized_stages,
        "technical_errors": normalized_errors,
    }


def aggregate_episode_attestations(attestations: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode facts; technical errors never become failed candidates."""
    normalized = [normalize_execution_attestation(item) for item in attestations]
    errors = [error for item in normalized for error in item["technical_errors"]]
    technical_valid_n = sum(item["technical_valid"] for item in normalized)
    protocol_reached_n = sum(item["protocol_reached"] for item in normalized)
    measured_n = sum(item["measurement_reached"] for item in normalized)
    technical_invalid = bool(errors) or technical_valid_n != len(normalized)
    error_counts = Counter((e["stage"], e["type"], e["message"]) for e in errors)
    repeated = [
        {"stage": key[0], "type": key[1], "message": key[2], "count": count}
        for key, count in error_counts.items()
        if count >= 3
    ]
    return {
        "attempted_n": len(normalized),
        "technical_valid_n": technical_valid_n,
        "protocol_reached_n": protocol_reached_n,
        "eligible_n": None if technical_invalid else sum(item["episode_state"] in {"ELIGIBLE", "QUALIFIED"} for item in normalized),
        "qualified_n": None if technical_invalid else sum(item["episode_state"] == "QUALIFIED" for item in normalized),
        "measured_n": measured_n,
        "technical_status": "FAIL" if technical_invalid else "PASS",
        "technical_invalid": technical_invalid,
        "uniform_failures": repeated,
    }
