# v2.2 Audit

Audit basis: current source, MCP schema, PostgreSQL models, tests, and the
provider-neutral workflow study in `docs/integrations/claude-code-study.md`.

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
validity and scientific evidence. The workflow is provider-neutral and can be
driven by ChatGPT, Codex, Claude Code, or a future client through MCP.

## Current execution tooling

The existing gateway exposes local and provider-backed execution, canonical
experiment/run/job mappings, retry-safe execution attempts, artifact and log
retrieval, and explicit scientific result assessment. An engineering task is an
optional prerequisite for code-changing experiments; when supplied, readiness
requires a passing baseline, diff review, guard results, and an accepted
implementation verification status.

## Current code-change representation

`EngineeringTask` stores purpose, task type, repository and relevant files,
parent ResearchDecision/Experiment links, change request, frozen scientific
variables, held-fixed variables, engineering/scientific invariants, prohibited
changes, acceptance and test commands, expected artifacts, and implementation
guards. `EngineeringResult` stores inspected/changed files, commands, tests,
diff identity, artifacts, guard results, verification status, and unresolved
failures.

## Current experiment implementation flow

Research Brain selects the scientific action and freezes the experiment.
EngineeringTask then records how the implementation will realize that design.
The task is inspected and baselined, the implementation is reviewed, machine
guards are evaluated, and only a verified task can be handed to execution.
Scientific execution and result assessment remain separate downstream steps.

## Current test infrastructure

Unit and service tests cover policy ordering, parent-link validation, baseline
gates, native-off/target-changed/held-fixed measurements, diff review, invalid
implementation handling, scientific-result isolation, MCP compatibility, and
restart durability when an explicit PostgreSQL test URL is supplied. Real GPU
scientific execution remains a separate verification category.

## Current git / workspace handling

The gateway records repository paths, inspected files, changed files, base
commit, diff identity, and unexpected changes. The coding-agent policy requires
preserving unrelated user changes and reviewing the resulting diff. Workspace
and command authorization remain enforced by the existing local/remote
execution boundaries; no Claude/Codex shell runtime is embedded.

## Current scientific validity guards

Scientific variables and held-fixed variables are first-class task data.
Machine-readable checks support native-off equality, target change, held-fixed
equality, checksums, and declared guard coverage. A failed guard produces
`INVALID_IMPLEMENTATION` and blocks scientific interpretation; it cannot refute
a hypothesis or teach strategy memory.

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
- `VERIFIED_UNIT`: EngineeringTask parent IDs are validated for object kind,
  project ownership, and decision-to-experiment consistency before persistence.
- `VERIFIED_UNIT`: `CodingExecutionPolicy` exposes a provider-neutral ordered
  phase contract (`RECEIVE` through `HAND_BACK`) and rejects skipped phases;
  its contract is explicitly `scientific_result=NOT_ASSESSED`.
- `VERIFIED_INTEGRATION`: MCP schema/import compatibility and existing execution
  control tests pass.
- `IMPLEMENTED_UNVERIFIED`: no real scientific experiment has been routed
  through an EngineeringTask in this audit; this layer does not create evidence.
