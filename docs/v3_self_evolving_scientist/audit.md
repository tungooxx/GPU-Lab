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

## Existing runtime and durable state

Research objects/events are PostgreSQL-backed; the service has MCP operations,
provider isolation, and a meta-review service. It has no v3 durable
`MetaWorldModel`, `MetaResearchAgenda`, opportunity/controller/configuration,
canary, or policy-regression objects yet.

## Baseline verification

Before v3 changes: `226 passed, 19 skipped` (one existing Pydantic settings
warning). This audit is based on current `policy_lab.py`, `meta_research.py`,
`research.py`, `brain_bench.py`, `engineering.py`, and MCP server sources.
