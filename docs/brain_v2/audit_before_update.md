# Research Brain v2 pre-update audit

Audit date: 2026-08-11. This audit was produced from the current
`feat/research-brain-v2` source, its tests, the live Compose PostgreSQL schema, and the Brain v1
verification report. Verification labels describe engineering evidence, not scientific truth.

## Current verified features

- `VERIFIED_REAL`: the canonical Python 3.13 / PyTorch 2.6.0+cu124 VRC runtime reproduced saved
  predictions exactly and the Brain v1 smoke completed one inspected GTX 1650 intervention that
  changed the next decision within its fixture scope.
- `VERIFIED_INTEGRATION`: PostgreSQL scientific state, immutable events, pgvector primitives,
  retry-safe local execution identity, MCP discovery, restart recovery, QD persistence, branch
  persistence, optional-worker isolation, and fixed-target relays.
- `VERIFIED_UNIT`: 136 tests pass before the v2 update. They cover Brain gates and scoring, result
  assessment, QD/dead-memory screening, deterministic branches, MetaLessons, provider contracts,
  local execution durability, MCP metadata, and network/credential boundaries.
- `SCIENTIFIC_RESULT_NOT_EXECUTED`: branch infrastructure has not yet produced a real inspected
  multi-branch comparison. PaperQA answer quality and a real Paper2Agent conversion remain
  unverified.

## Current database schema

The live database contains 58 projects. Canonical tables are `research_projects`,
`research_objects`, `research_edges`, `research_events`, and `research_execution_attempts`.
Scientific types are constrained in `RESEARCH_OBJECT_KINDS`; current objects include WorldModels,
versions, agendas, hypotheses/niches/portfolios, decisions/candidates, experiments/runs/artifacts,
branches/nodes/relations, ComparativeLessons, and MetaLessons. pgvector is an optional column on
`research_objects`. Migrations run transactionally under a PostgreSQL advisory lock. Brain v2
temporal snapshots require PostgreSQL `track_commit_timestamp=on`; Compose configures this explicitly
so historical reads use commit visibility rather than pre-commit statement timestamps.

There is no object-revision table, evidence-family representation, embedding metadata table, or
strategy-memory type. `created_at` can bound object creation, but later in-place updates cannot be
reconstructed at historical cutoff T. That is the central temporal-leakage migration gap.

## Current scientific workflow

The implemented path is project/state → reproduction gate → agenda-driven `brain_step()` → typed
candidate/decision persistence → preregistration → exact decision/execution binding → approval when
required → atomic run reservation → real execution → sync/log/artifact retrieval → explicit result
assessment → hypothesis/agenda/decision/ResearchState and optional WorldModel update. A job ID alone
is never scientific evidence.

## Current Brain policy

Brain v1 first selects completed-result inspection, failed-result inspection, unfinished recovery,
or required reproduction. It then scores caller-configured actions with transparent importance ×
discrimination × expected-information × feasibility divided by compute × engineering × risk.
Dead related ideas and deterministic advisory critics are persisted. It has no temporal cutoff,
null-model critic, evidence-independence accounting, diminishing-return adjustment, ResearchSituation,
or reusable strategy prior.

## Current evidence promotion logic

`brain_result_assess()` validates run/decision/hypothesis/agenda/project identity, exact pass/guard
conditions, result-inspection state, evidence references, and allowed transitions. The store applies
related updates in one locked transaction and can version a causal edge/WorldModel. It does not
group derived records by empirical origin, calculate independent support, encode a structured causal
scope/support level, run a belief audit, or gate promotion on WorldModel consistency and null coverage.

## Current WorldModel

WorldModels own typed node IDs, causal edge IDs, and immutable WorldModelVersion objects. Causal edges
carry status and supporting/against/prediction references. Updates are transactional and versioned.
Scope is mostly free text; replication/support level and independent evidence families are absent.
There is no consistency checker.

## Current QD

Native QD provides typed `HypothesisDraft`, niches, lineage/ancestors, lexical and optional vector
retrieval, structured mechanism overlap, dead-idea flags, a scientific-difference gate, and temporary
typed generator/reflector/proximity operators. The generator only validates caller-supplied drafts;
it is not model-backed. There is no dedicated NullModelCritic or full v2 typed candidate schema.

## Current branch system

Deterministic ExperimentBranches store scored nodes, parent/comparison relations, exact inspected-run
results, and confound-preserving ComparativeLessons. Policy order is inspect → recover → execute by
information/cost → compare. No MCTS exists. The branch system is structurally integration-tested but
has not completed the required real three-branch scientific test.

## Current meta-learning support

Progress metrics and `meta_review()` summarize inspected/unfinished work, failed assumptions, branch
comparison debt, GPU hours, and campaign readiness. MetaLessons are durable but are descriptive.
There are no ResearchSituations, DecisionOutcomes, scoped StrategyPatterns, counterexamples,
positive/negative transfer rules, or `(S_t, A_t, R_t, S_t+1)` export.

## Current embedding support

The store can manually save a vector and perform pgvector cosine search. QD falls back to lexical and
structured retrieval when pgvector is unavailable. Automatic canonical text, provider abstraction,
source hashes, provider/model/version/dimension metadata, invalidation, and recomputation are absent.

## Current external providers

PaperQA 2026.3.18 is an optional isolated HTTP worker behind `LiteratureProvider`; contract, health,
provenance normalization, and isolation are integration-tested, while model-backed answer quality is
not. Paper2Agent is an optional isolated subprocess worker behind `ExecutablePaperProvider`; its
contract and authorization boundary are tested, while a real conversion is `IMPLEMENTED_UNVERIFIED`.

## Current security model

The gateway is on the backend network. Untrusted workers are capability-dropped, non-root,
no-new-privileges services on internal networks and reach one authenticated API through fixed-target
relays. A constrained egress proxy blocks private/loopback/host-gateway destinations. Worker-only API
keys are blanked in the gateway, and local jobs inherit a narrow runtime environment rather than
gateway secrets. External text/repository/provider output is not allowed to promote scientific state.

## Missing v1.5 features

- Temporal object revisions and cutoff-aware structured, lexical, event, and vector retrieval.
- EvidenceFamily, dependency notes, and independent-origin counting.
- Structured causal scope/support levels and promotion gates.
- BeliefAudit and WorldModel consistency issues/checks.
- Automatic provider-neutral embeddings with metadata/invalidation/fallback tests.
- ResearchBrainBench episodes, baselines, and non-opaque scorecards.

## Missing v2 features

- Model-backed typed ResearchOperators and hypothesis candidate pipeline.
- NullModelCritic and null/control planner actions.
- Diminishing-return and realized-information telemetry.
- ResearchDecisionOutcome and explicit transition tuples.
- ResearchSituation and project/domain/global ResearchStrategyPatterns.
- Applicability/counterexample-aware strategy retrieval, ranking adjustment, and negative transfer.
- Strategy dataset export and v1/v1.5/v2 held-out comparison.
- Real model-backed literature evaluation, real Paper2Agent conversion, and real inspected branch
  science with comparative learning.

## Migrations required

1. Add `research_object_versions` and transactional capture/backfill for historical visibility.
2. Add temporal/vector metadata needed to reject embeddings created after cutoff T.
3. Extend allowed object kinds for EvidenceFamily, BeliefAudit, consistency issues, null models,
   decision outcomes, situations, strategy patterns/outcomes, operator runs, and benchmark runs.
4. Add focused indexes for origin independence keys, temporal versions, strategy signatures, and
   embedding source hashes. Preserve all current object IDs and events.

## Files to modify

- `src/gpu_lab/research.py`: additive migrations, temporal reads, atomic epistemic/strategy writes.
- `src/gpu_lab/brain.py`: v1.5 gates and strategy-aware transparent ranking while preserving v1.
- New focused native modules for benchmark, epistemics, embeddings, operators, and strategy memory.
- `src/gpu_lab/server.py`: typed MCP façades and compact safe responses.
- Tests, sourced benchmark fixtures, smoke scripts, README/current-state/verification documentation.

## MCP tools to add

At minimum: benchmark run/scorecard, EvidenceFamily create/query/count/group, `belief_audit`,
`world_model_consistency_check`, automatic embedding refresh/status, hypothesis generation/screening,
null critique, decision outcome assessment, situation/strategy retrieval/extraction, strategy dataset
export, and Brain v2 step/status. Mutating tools must retain accurate safety annotations.

## Real test plan

1. Prove cutoff T cannot see later objects, updates, events, strategy records, or embeddings.
2. Prove five records from one experiment count as one independent EvidenceFamily.
3. Prove one VRC intervention remains single-intervention, architecture-scoped support.
4. Run HASI scope/BeliefAudit and intentionally inconsistent WorldModel regressions.
5. Prove automatic embedding metadata/recompute/fallback and restart recovery.
6. Compare policy baselines on sourced historical episodes without future leakage.
7. If scoped credentials work, evaluate PaperQA and operators; otherwise report model-backed quality
   as unverified. Keep Paper2Agent unverified unless one real controlled build is inspected.
8. Execute a preregistered, inexpensive real three-branch GPU experiment, inspect every result,
   create EvidenceFamilies and ComparativeLesson, and prove the next decision changes.
9. Demonstrate same-project learning, positive cross-project transfer, and structured rejection of
   superficially similar negative transfer.
10. Restart database/MCP, rerun security isolation, export policy data, and keep bounded campaigns at
    `DO_NOT_BUILD_YET` unless every stated readiness gate passes.
