# GPU-Lab / Research OS v3.6.5 authority audit

Date: 2026-08-30

## Scope and result

The v3.6.5 change is implemented as a separate authority model rather than as
additional WorkItem lifecycle states.  The governing rule is enforced by the
controller and exposed over MCP:

> A completed review WorkItem is not a semantic PASS, and a READY WorkItem is
> not execution authority.

The live `gpu-lab` container was checked against the source tree.  SHA-256
values for `lab.py`, `server.py`, `qd.py`, `research.py`, and `brain.py` match
their loaded `/app/src/gpu_lab/*` counterparts.

## Implementation map

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Work lifecycle separate from authority | `authority_class` and `promotion_state` columns on `lab_work_items` | `test_v365_execution_permission_requires_atomic_promotion_after_pass` |
| Verdict-aware dependencies | `REVIEW_VERDICT_EQUALS` and `REVIEW_VERDICT_IN`; review results are immutable | `test_completed_review_work_cannot_satisfy_a_nonpass_verdict_dependency` |
| Derived permissions | `LabController.work_permissions_get` | v3.6.5 promotion test |
| Atomic promotion | `LabController.authority_promote`, work/version fencing, transactional outbox event | v3.6.5 promotion test |
| Independent review result | `lab_review_results`; review class must be `INDEPENDENT_REVIEW` | atomic PASS and MRR tests |
| PASS continuation | `review_pass_transition` writes the immutable PASS result, gate, completed review, and deterministic successor in one transaction | `test_v365_atomic_pass_transition_requires_and_records_immutable_verdict` |
| MRR continuation | `review_material_repair_transition` sets gate `REPAIR_REQUIRED`, supersedes unsafe authority, and marks affected speculative bindings `REBASE_REQUIRED` | `test_v365_material_repair_transition_is_atomic_and_rebases_speculation` |
| Safe speculative labor | `lab_speculative_bindings`, speculation classes, rebase assessment, and explicit no-authority binding checks | MRR and speculation-class tests |
| Reviewer harness isolation | `lab_review_harnesses` and `review_harness_prepare`; confirmation access is rejected | `test_reviewer_harness_is_frozen_preparatory_not_authority` |
| Evidence/reinspection dedupe | `lab_evidence_needs`, explicit exception keys for independent review/replication | `test_evidence_need_suppresses_reinspection_but_allows_explicit_independent_review` |
| Typed deterministic guards | guard predicates including `MUST_BE_FALSE` and `INFORMATIONAL` | `test_typed_preflight_guards_treat_false_and_information_correctly` |
| Conflict detection | `lab_consistency_conflicts` records completed-nonpass review release and invalid promoted speculation | CPDS regression tests |

## CPDS regression

The regression in the specification is covered by a typed dependency that
requires the review result verdict to be PASS.  A review result of
`MATERIAL_REPAIR_REQUIRED` does not satisfy that dependency even when the
review WorkItem is `COMPLETED`.  The repair transition also retires active
authoritative children of that gate and requests speculative rebase rather
than releasing execution.

## MCP surface

The current server exposes `lab_work_create` authority/speculation fields plus
the explicit operations below:

- `review_result_record`
- `scientific_review_pass_transition`
- `scientific_review_material_repair_transition`
- `lab_work_permissions_get`
- `lab_work_authority_promote`
- `speculative_binding_create`
- `speculative_rebase_assess`
- `review_harness_prepare`
- `evidence_need_lookup`

## Verification performed

- Windows local suite: `395 passed, 95 skipped`.
- Fresh isolated PostgreSQL database, focused v3.6.5 Lab tests: `6 passed`.
- Runtime source-hash check: all five critical modules match the live
  container.

The broad database suite is intentionally not claimed here: a prior attempt
against a reused verification database was interrupted during failure output.
The focused v3.6.5 database suite above used a fresh database and is the
database-backed evidence for the authority changes.

## Operational note

TR1 has no host-maintenance privileges: `host_source_write`,
`service_restart`, and `runtime_hash_verify` were removed on 2026-08-30.  Its
host-only task was released with the audited lease reason
`CAPABILITY_REVOKED`; it is available to another worker that has the required
capabilities.

## Discovery-generation reassignment

An expired `DISCOVERY_GENERATION` WorkItem no longer transfers its open
CandidateBatch to a different worker/session.  On the next valid claim the
controller atomically expires the old membership and batch, creates a fresh
session-owned replacement batch with the same frozen operator/distance slot,
and rewrites the WorkItem binding.  Expired batches are excluded from archive
synthesis.  The regression is
`test_expired_independent_generation_work_claim_creates_fresh_batch_lane`.
