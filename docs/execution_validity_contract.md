# Execution-validity contract

Scientific interpretation requires a recorded execution attestation.  The
attestation is intentionally independent of exit code, packet schema, and
scientific gates.

## Invariants

- `technical_invalid_precedes_scientific_gate_evaluation`
- `exceptions_are_not_negative_observations`
- `scientific_outcome_requires_measurement_reached`

An attested error or a missing required stage makes the run technically
invalid. `brain_result_assess` then returns `SCIENTIFIC_OUTCOME_INVALID`; the
only valid closure is `research_technical_result_inspect`, which records no
evidence or belief update.

Required stages are runtime initialization, environment reset, initial
observation, planner and plan parsing, executor start, eligibility evaluation,
and scientific metric evaluation. Episode states are `TECHNICAL_ERROR`,
`PROTOCOL_REACHED_NOT_ELIGIBLE`, `ELIGIBLE`, and `QUALIFIED`.

The episode-summary API reports `attempted_n`, `technical_valid_n`,
`protocol_reached_n`, `eligible_n`, `qualified_n`, and `measured_n`. When any
technical error occurs, eligibility and qualification are `null`, not zero.
Three matching technical errors emit a uniform-failure signature so workers can
open an engineering investigation instead of continuing scientific execution.

## Exact-runtime canaries

`runtime_canary_record` stores a passing non-scientific canary under a SHA-256
runtime fingerprint. An experiment plan may set
`require_exact_runtime_canary=true`; `research_experiment_execute` then rejects
execution unless its supplied `runtime_fingerprint` matches a passing canary in
the project. Existing plans remain compatible until they opt in. A fingerprint
should cover the Python executable, package lock, runner and compatibility
patches, environment assets, model/tokenizer, and device configuration.

## Migration

Historical runs without `execution_attestation` remain readable and retain
their prior assessment semantics. New attested runs are protected at the API
boundary. Historical technical failures can be closed with the technical
inspection API; no migration rewrites scientific results automatically.
