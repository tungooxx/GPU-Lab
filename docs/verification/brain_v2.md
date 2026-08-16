# Research Brain v2 verification report

Verification date: 2026-08-12. This report distinguishes working code and integration evidence from
scientific claims. It does not treat a tool call, a submitted job, or a unit test as scientific proof.

## Research Brain v2 summary

Brain v2 extends the preserved PostgreSQL-owned Brain v1 with v1.5 epistemic controls and explicit,
scoped strategy memory. The current branch remains deliberately below the bounded-campaign readiness
gate: key real-world and model-backed evaluation requirements are not yet evidenced.

## Brain v1 components preserved

PostgreSQL scientific objects/events, ResearchState, WorldModel history, agenda/portfolio/decision
ledger, GPU-Lab execution, preregistration, explicit result inspection, QD, branches, MetaLessons,
PaperQA/Paper2Agent isolation, and Streamable MCP remain native and unchanged in ownership.

## Audit findings

The pre-update audit is retained in `docs/brain_v2/audit_before_update.md`. Its listed v1.5/v2 gaps
are now implemented where covered below; its real-science and model-backed gaps remain open.

## Database migrations

The working tree adds additive ResearchStore object kinds, indexes, temporal reconciliation, and
atomic persistence for situations, decision outcomes, null models, strategies, and outcomes. Existing
scientific object IDs are preserved.

## ResearchBrainBench

`research_bench/` contains three sourced frozen episodes: `after_stage4`, `before_ejc`, and
`hasi_before_intervention`. The runner uses `visible_payload` only, records policy versions, and
returns a non-opaque scorecard. More sourced checkpoints are still required.

## Historical episodes

`VERIFIED_INTEGRATION` for the three existing fixtures. The specification's larger historical
trajectory set is incomplete because missing facts must not be fabricated.

## Temporal leakage tests

`VERIFIED_UNIT`: temporal research, epistemics, embeddings, and benchmark tests enforce cutoff-aware
object/event/strategy retrieval and keep hidden benchmark labels outside policy input.

## Policy baselines

`VERIFIED_UNIT`: Current Brain v1, cheapest feasible, maximum expected information, direct LLM-style
without structured memory, random valid, Brain v1.5, and Brain v2 strategy-augmented policies are
implemented and compared without an opaque aggregate score.

## Evidence family

`VERIFIED_UNIT`: `EvidenceFamily` records origin, independence key, derivation/dependency metadata,
and can link derived scientific records.

## Anti-double-counting

`VERIFIED_UNIT`: epistemic services group derived records by evidence family and count independent
empirical origins rather than downstream record count.

## Scope-aware causal support

`VERIFIED_UNIT`: causal support preserves explicit model/architecture/checkpoint/intervention/metric
scope and independent support/contradiction families. A single intervention cannot be universalized.

## BeliefAudit

`VERIFIED_UNIT`: audit reports support, against evidence, scope, dependencies, missing
generalization, closest dead ideas, risks, and recommended next evidence for supported entity types.

## World model consistency

`VERIFIED_UNIT`: the checker returns typed warnings/errors for evidence-free promotion, scope mismatch,
contradictions, missing mechanisms, insufficient replication, and invalid causal configuration.

## Automatic embeddings

`VERIFIED_UNIT`: deterministic canonical text, provider/model/version/dimension/source-hash metadata,
automatic refresh, invalidation, pgvector retrieval, and restart recovery are covered.

## Embedding failure fallback

`VERIFIED_UNIT`: embedding failure leaves lexical, structured, lineage, and exact relationship
retrieval available; it cannot block scientific-state access.

## LLM RESEARCHOPERATORS

`IMPLEMENTED_UNVERIFIED`: typed, bounded, advisory operator interfaces and provenance are present.
Real model-backed operator reliability/variance has not been evaluated.

## Hypothesis generation

`VERIFIED_UNIT` for schema/provenance/screening boundaries. `IMPLEMENTED_UNVERIFIED` for a live
model-backed 3–5-candidate generation smoke.

## QD / dead-idea screening

`VERIFIED_INTEGRATION` for native QD persistence, dead-idea proximity, explicit difference gates,
and restart recovery. Its longitudinal scientific benefit is unverified.

## NullModel critic

`VERIFIED_UNIT`: typed null records, strong-cheap null action preemption, and causal-promotion gates
are present. No real null-control experiment has been executed in this v2 slice.

## PaperQA quality benchmark

`BLOCKED`: no verified 50–100-question model-backed quality benchmark exists yet.

## Negative-evidence retrieval

`VERIFIED_UNIT` for canonical negative-memory retrieval and evidence-family grouping;
`IMPLEMENTED_UNVERIFIED` for a real literature negative-evidence evaluation.

## Paper2Agent verification status

`VERIFIED_INTEGRATION` for the isolated provider contract and safety boundary.
`IMPLEMENTED_UNVERIFIED` for an approved, inspected real model-backed paper conversion.

## Provider failure tests

`VERIFIED_UNIT` / `VERIFIED_INTEGRATION`: provider and embedding outages return structured failure or
safe alternative retrieval without mutating scientific truth.

## Real multi-branch science

`SCIENTIFIC_RESULT_NOT_EXECUTED`: branch persistence and selection are integration-tested, but the
required three preregistered real branches have not been executed and inspected.

## Comparative learning

`VERIFIED_UNIT` for ComparativeLesson persistence and planner retrieval.
`SCIENTIFIC_RESULT_NOT_EXECUTED` for a ComparativeLesson produced from real inspected branches.

## Planner diminishing returns

`VERIFIED_UNIT`: repeated low-value action families contribute transparent diminishing-return
adjustments while reproduction/inspection hard gates retain precedence.

## Decision outcome

`VERIFIED_UNIT`: reassessable decision outcomes atomically store observed result, information gain,
hindsight, post-action situation, and strategy effects. `UNKNOWN`/`BLOCKED` are retained but do not
create strategy patterns.

## Research situation

`VERIFIED_UNIT`: deterministic structured situations exclude project-name dominance and are persisted
with the decision snapshot.

## Project strategy memory

`VERIFIED_UNIT`: project-level strategy patterns use outcome evidence and explicit applicability.

## Domain strategy memory

`VERIFIED_UNIT`: project patterns can promote to domain scope only through sufficient attributable
outcomes.

## Global strategy memory

`VERIFIED_UNIT`: global patterns remain separately scoped and provenance-bearing.

## Strategy counterexamples

`VERIFIED_UNIT`: rejected applicability creates preserved negative-transfer/counterexample provenance.

## Strategy-aware brain_step

`VERIFIED_UNIT`: the planner retrieves applicable/rejected patterns, stores policy versions, and
shows base, positive, negative, and diminishing-return score components before selection.

## Positive transfer test

`VERIFIED_INTEGRATION`: `scripts/strategy_transfer_smoke.py` demonstrates scoped Project A to
Project B retrieval and ranking adjustment.

## Negative transfer test

`VERIFIED_INTEGRATION`: the same smoke demonstrates structured rejection for a superficially similar
Project C with an internal-state-access mismatch.

## V1 vs V1.5 vs V2 benchmark

`VERIFIED_UNIT`: all required policies run against the current three frozen episodes. Held-out
historical quality remains weak evidence until more sourced episodes are available.

## Real GPU tests

`VERIFIED_REAL` for the retained Brain v1 GTX 1650 inspected intervention and changed next decision.
`SCIENTIFIC_RESULT_NOT_EXECUTED` for the required v2 multi-branch science smoke.

## Model-backed tests

`IMPLEMENTED_UNVERIFIED`: PaperQA answer quality, operator generation, and Paper2Agent conversion
need scoped credentials and approved cost.

## Restart tests

`VERIFIED_INTEGRATION`: prior Brain v1, QD, branch, and strategy smoke evidence covers PostgreSQL/
MCP restart recovery for persisted state. After rebuilding the current `gpu-lab` image,
`scripts/brain_v2_mcp_smoke.py` passed against the live Streamable HTTP endpoint, discovering the v2
tools, persisting a ResearchSituation, validating structured EvidenceFamily errors, storing an
`UNKNOWN` DecisionOutcome without creating strategy memory, exporting one policy-transition record,
and comparing the three available benchmark episodes.

## Security tests

`VERIFIED_INTEGRATION`: worker isolation, non-root/capability boundaries, fixed relays, and egress
restrictions remain covered by the existing security suite and worker isolation smoke.

## Transaction tests

`VERIFIED_UNIT`: decision/outcome/strategy updates are atomic and locks prevent partial scientific or
strategic writes.

## Verified real

Only the inherited Brain v1 local GPU end-to-end intervention is `VERIFIED_REAL`.

## Verified integration

MCP/service persistence, QD/branch/restart/security boundaries, strategy transfer smoke, and optional
provider contracts have integration evidence as stated above.

## Verified unit

Temporal access, epistemic accounting, causal scope, embeddings, benchmark policies, null gates,
strategy adjustment/outcomes, and metadata are covered by focused tests.

## Implemented unverified

Model-backed operators/PaperQA/Paper2Agent, real literature quality, and longitudinal branch/QD
value remain unverified.

## Scientific results

No new scientific mechanism is claimed by the v2 work. Existing scientific conclusions retain their
prior explicit scope and provenance.

## Strategy-learning results

The system demonstrates stored, scoped positive transfer and rejection of negative transfer in
integration fixtures. It does not yet prove strategy learning improves real research outcomes.

## Known risks

Three historical episodes are too few for a strong policy conclusion. Strategy weights are heuristic,
not calibrated. Model-backed providers may be unavailable/costly. The v2 real branch-science
acceptance test is still missing.

## Bounded campaign readiness

`DO_NOT_BUILD_YET`. Required real branch learning, broader benchmark evidence, model-backed quality
evaluation, and complete v2 restart/real-science verification are not satisfied.

## Future policy-dataset readiness

`VERIFIED_UNIT`: versioned `(S_t, A_t, O_t, R_t, S_t+1)` export exists. It is not authorization to
train or deploy a learned policy.

## Smoke entrypoints

`scripts/brain_v2_science_smoke.py`, `scripts/brain_v2_llm_smoke.py`, and
`scripts/brain_v2_branch_science.py` are present as gated entrypoints. By default they perform
preflight and report `SCIENTIFIC_RESULT_NOT_EXECUTED` or `IMPLEMENTED_UNVERIFIED`; explicit environment
gates are required before delegating to real GPU or model-backed work. They never fabricate
EvidenceFamilies, ComparativeLessons, or scientific outcomes.

## Next highest-value step

Run a small, preregistered, approved real three-branch GPU experiment; inspect every outcome, create
EvidenceFamilies and a ComparativeLesson, then show that the resulting evidence changes the next
Brain v2 decision. In parallel, add sourced historical checkpoints rather than inventing them.
