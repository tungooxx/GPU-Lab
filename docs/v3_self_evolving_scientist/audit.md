# V3 Self-Evolving Scientist Audit

## Verified v2.5 capabilities

- Durable `ResearchPolicy`, `PolicyHypothesis`, `ResearchPolicyPatch`,
  `PolicyExperiment`, `PolicyNegativeResult`, and `ImprovementRun` records.
- Isolated benchmark evaluation with development, validation, held-out splits,
  hard regression metrics, rejection, bounded revision, and explicit promotion.
- Atomic PostgreSQL production-policy creation, promotion, and rollback.
- Provider-neutral Prompt-as-Code compilation and immutable policy artifacts.
- Policy hindsight, project/model transfer classification, and PaperQA-gated
  targeted literature input.

## Current auto-evaluation and promotion

Candidates are evaluated automatically by default, but production promotion is
explicit. The existing policy service does not autonomously schedule campaigns,
track an autonomy mode, detect production regressions, or auto-rollback.

## Historical benchmark and model coverage

`ResearchBrainBench` has three split-tagged frozen episodes with hard leakage,
scope, bad-action, and premature-architecture regressions. Evaluation is a
deterministic policy runner, not live provider/model execution. Transfer fields
exist, but broad cross-model evidence is caller-supplied.

## Existing runtime and durable state (baseline)

Research objects/events are PostgreSQL-backed; the service has MCP operations,
provider isolation, and a meta-review service. It has no v3 durable
`MetaWorldModel`, `MetaResearchAgenda`, opportunity/controller/configuration,
canary, or policy-regression objects yet.

## Baseline verification

Before v3 changes: `226 passed, 19 skipped` (one existing Pydantic settings
warning). This audit is based on current `policy_lab.py`, `meta_research.py`,
`research.py`, `brain_bench.py`, `engineering.py`, and MCP server sources.

## V3 implementation progress (current branch)

The current implementation adds durable meta-world observations, agenda items,
opportunities, campaign claims/budgets, benchmark gaps, regressions, policy
hindsight, canary/shadow records, evaluator-firewall audits, ranker readiness,
and meta-strategy patterns. It triggers one bounded controller pass after a
decision outcome and a process-only postmortem at five-decision milestones.

Automatic promotion is scoped, preflighted, pinned-policy aware, and records
evaluation provenance. Provider/model changes run an adapter compatibility
check and remain `CROSS_MODEL_UNVERIFIED` unless live cross-model evidence is
separately supplied. Literature scouts create problem-driven requests and only
gather evidence candidates when the isolated provider is available.

## Remaining v3 work

This is not a completion claim. Code-bearing policy patches now create a
bounded v2.2 `EngineeringTask`; a verified `EngineeringResult` automatically
unlocks the pre-registered benchmark, while invalid implementation evidence is
retained as a non-scientific policy negative result. An execution agent still
performs the bounded code change: the controller does not receive arbitrary
repository-write authority. Richer causal meta-diagnostics, live multi-provider
policy evaluation, benchmark-authoring workflow, and prospective canary routing
remain broader integration work. The deterministic benchmark runner should not
be represented as live-model or real-world proof.

Meta-campaign claims are restart-durable: a restarted controller resumes a
claimed campaign, reuses a linked completed `ImprovementRun` when present, and
finishes promotion idempotently. This covers controller restart recovery, not
durable recovery of an external code executor or unavailable provider.

Before an autonomous campaign asks the policy lab for patches, it persists a
trace-grounded `MetaWorldModel` containing competing candidate-generation,
ranking, and critic explanations. These are explicitly marked
`HYPOTHESIS_NOT_ESTABLISHED`; they order distinct candidate mechanisms but do
not convert observational failure patterns into causal claims.

The controller also records an explicit science-vs-meta scheduling decision.
A materially higher-priority unresolved domain agenda item defers a modest
meta campaign; repeated severe invalid outcomes can still preempt that deferral.

`tests/test_meta_controller.py` includes an autonomous improvement acceptance
sequence: repeated LOW_VALUE outcomes launch a project-scoped campaign without
manual `/improve`, promote a held-out supported policy, record prospective
positive policy hindsight, and confirm that no rollback is triggered.

Completed literature scouts now create durable `LiteraturePolicyTransfer`
records. They preserve candidate-evidence provenance, a minimal transferable
principle, explicit unknown extraction fields, and an overlap classification;
they seed later policy invention but never automatically import an external
architecture or establish a scientific claim.

A detected provider/model change now runs the compact compatibility compile and
creates a durable candidate-only `ProviderAdapterCandidate`. The canonical
policy compiler renders selected adapter data, while the candidate remains
ineligible for promotion until genuine live cross-model support is recorded.

The `policy_health_report` MCP tool now provides the compact v3 operational
report: production policy and lineage, scoped overrides, adapter candidates,
benchmark health, real-world calibration, active meta work, recent decisions,
rollbacks, and known policy failure modes.

Policy evaluation includes an adversarial falsification pass over eligible
held-out high-risk episodes. New hard regressions versus the frozen baseline
are stored on the `PolicyExperiment` and block promotion.
