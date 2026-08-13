# v2.2 Audit

## Current engineering workflow

GPU-Lab already provides a PostgreSQL research object/event store, local/remote
execution controls, artifact collection, result assessment, and v2.1 epistemic
classification. It did not have a typed record for code implementation work.

## Components to preserve

`ResearchDecision`, experiment/run reservation, immutable events, result
assessment, `LocalRunner`, workspace boundaries, and v2.1 strategy eligibility
remain authoritative. Engineering records are additive and cannot update
hypotheses, causal edges, or strategy memory.

## Gap addressed

`EngineeringTask` captures a frozen implementation request and scientific versus
engineering invariants. `EngineeringResult` captures implementation evidence
with `scientific_result=NOT_ASSESSED`, preserving the separation between code
validity and scientific evidence.

The execution handoff is guarded when a task ID is supplied: the task must link
to the requested experiment, all declared implementation/scientific guards must
be present and pass, and only then is the existing experiment reservation path
allowed to continue. Tasks without a code-change requirement remain compatible
with the prior execution API.

## Verification status

- `VERIFIED_UNIT`: engineering lifecycle and fail-closed guard regressions pass.
- `VERIFIED_UNIT`: measurement guards independently test native-off equivalence,
  intervention-on target change, and held-fixed equality; a no-op or drift is
  persisted as `INVALID_IMPLEMENTATION`, not a scientific result.
- `VERIFIED_UNIT`: `engineering_task_start` requires recorded repository
  inspection and a passing baseline before an EngineeringResult can be saved;
  failed baselines block implementation without changing scientific state.
- `VERIFIED_UNIT`: `engineering_diff_review` records changed files and rejects
  unrelated changes or scientific-variable drift; frozen task fields cannot be
  edited through the generic update operation.
- `IMPLEMENTED_UNVERIFIED`: `tests/test_engineering_postgres.py` and
  `scripts/engineering_e2e_smoke.py` cover restart durability against an
  explicitly supplied PostgreSQL URL. They were not run against the live
  Compose database because its `postgres` hostname is container-only and no
  `GPU_LAB_TEST_DATABASE_URL` was configured in the host environment.
- `VERIFIED_UNIT`: `EngineeringResult` is normalized with bounded files,
  commands, tests, baseline, diff, guard, artifact, implementation-status, and
  unresolved-failure fields. Scientific statuses are rejected and the stored
  scientific result remains `NOT_ASSESSED`.
- `VERIFIED_INTEGRATION`: MCP schema/import compatibility and existing execution
  control tests pass.
- `IMPLEMENTED_UNVERIFIED`: no real scientific experiment has been routed
  through an EngineeringTask in this audit; this layer does not create evidence.
