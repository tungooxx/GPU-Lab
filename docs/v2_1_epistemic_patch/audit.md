# v2.1 epistemic patch audit

Audit date: 2026-08-12

## Current enums

Before this patch, `ResearchDecision` stored a free-form JSON payload with a
selected action, score, rationale, and (after assessment) an outcome. There was
no orthogonal decision role, execution verification, scientific role, learning
namespace, or closed-cycle status. `ResearchDecisionOutcome` already stored
observed results, realized information gain, hindsight, evidence-family IDs,
and situation transitions.

This patch adds deterministic metadata fields without removing legacy status
values such as `VERIFIED_REAL`.

## Current status semantics

Object status remains the lifecycle/status field. New fields are semantic
dimensions:

- `decision_role`: scientific action versus inspection, recovery, backfill,
  system verification, or benchmark evaluation.
- `scientific_role`: causal test, reproduction, diagnostic, system smoke, etc.
- `execution_verification`: execution reality (`REAL_GPU`, `UNIT`, etc.).
- `scientific_verification`: evidence state, independent of execution reality.
- `cycle_status`: open through closed/abandoned lifecycle.
- `learning_namespace`: `PRODUCTION_SCIENCE`, `BENCHMARK`, or `SYSTEM_TEST`.

## Current strategy-eligibility logic

Previously, a recognized positive/negative outcome could immediately update
strategy patterns. The patch now requires a prospective scientific action, a
closed inspected outcome, known realized information, hindsight, and a
structured information-gain basis. Administrative, legacy, system, contract,
benchmark, and incomplete records are excluded.

## Current historical-backfill logic

Legacy provenance was already preserved, but its semantic role was not
orthogonal. Deterministic classification now maps reconstructed legacy records
to `LEGACY_BACKFILL` and inspection/recovery actions to `RESULT_INSPECTION` or
`ADMINISTRATIVE_RECOVERY` without changing observations or timestamps.

## Current decision-outcome coverage

Outcome records contain reassessable `S_t`, `A_t`, `O_t`, `R_t`, and `S_t+1`
data. The new eligibility report identifies missing hindsight, missing basis,
unknown information, and incomplete cycles explicitly.

## Current critic output

Existing procedural critics remain intact. The patch adds typed epistemic
fields to decision payloads and exposes runner-up/discrimination/null fields
when supplied by the planner; it does not promote critic output to scientific
truth.

## Current meta metrics

Meta-review previously counted all decisions together. It now reports separate
scientific totals, closed scientific cycles, open cycles, administrative
decisions, system-verification decisions, and strategy-eligible cycles. GPU
hour metrics remain unknown when actual accounting is absent.

## Breaking-change risks

The new fields are additive. Existing callers and `VERIFIED_REAL` values remain
valid. Strategy aggregates may become smaller after reclassification; that is
intentional and must be reported rather than hidden.
