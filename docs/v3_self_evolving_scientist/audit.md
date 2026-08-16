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

This is not a completion claim. In particular, autonomous code-patch execution
through EngineeringTask/EngineeringResult, richer causal meta-diagnostics,
live multi-provider policy evaluation, benchmark-authoring workflow, and
prospective canary routing remain broader integration work. The deterministic
benchmark runner should not be represented as live-model or real-world proof.
