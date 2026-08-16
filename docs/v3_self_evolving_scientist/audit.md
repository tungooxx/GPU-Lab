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
 opportunities, campaign claims/budgets, benchmark gaps, future-only benchmark
 authoring proposals, regressions, policy
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

At every fifth scientific decision, the durable process-only postmortem now
turns repeated low-value action families into bounded `META_REVIEW`
opportunities. The `MetaLesson` is merely a trigger: each opportunity retains
the underlying `ResearchDecisionOutcome` ids and cannot treat derived review
text as independent evidence.

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
The evidence-gated adapter lifecycle now records live PASS/FAIL evidence and
promotes only supported adapters into a new scoped policy descendant.

The `policy_health_report` MCP tool now provides the compact v3 operational
report: production policy and lineage, scoped overrides, adapter candidates,
benchmark health, real-world calibration, active meta work, recent decisions,
rollbacks, and known policy failure modes.

Policy evaluation includes an adversarial falsification pass over eligible
held-out high-risk episodes. New hard regressions versus the frozen baseline
are stored on the `PolicyExperiment` and block promotion.

The autonomous rollback acceptance test covers a real controller sequence:
auto-promotion, repeated negative prospective hindsight, `PolicyRegression`,
rollback to the parent policy, and preservation of `PolicyNegativeResult`.
Rollback now respects the promoted policy's `PROJECT`, `DOMAIN`, or `GLOBAL`
scope and only runs when the matching autonomy mode is enabled. Every detected
regression also creates a durable calibration-diagnosis agenda question before
the failed policy can be reconsidered.

Recording a prospective `PolicyHindsight` event is itself an autonomous trigger:
the MCP path runs both calibration monitoring and scoped regression/rollback
monitoring, so operators do not need to invoke a second regression-check tool.

The `meta_research_roi` MCP tool reports campaign budget ceilings and observed
candidate, promotion, regression, hindsight, invalid-experiment, and
zero-information rates. It intentionally reports future research cost avoided
as unknown until matched prospective evidence can support a causal estimate.

Each durable meta campaign now records actual candidate, benchmark, revision,
literature, and engineering-task consumption alongside its ceiling budget and
an explicit stop reason.

## Verification status

| Area | Status | Evidence |
| --- | --- | --- |
| Autonomous weakness detection, agenda, bounded campaigns | VERIFIED_UNIT | Controller tests cover recurring outcomes, prioritization, scheduling, budgets, and durable claims. |
| Evaluation, held-out and adversarial falsification | VERIFIED_UNIT | Policy-lab tests cover evaluator firewall, held-out rejection, and high-risk adversarial regression. |
| Project auto-promotion and rollback | VERIFIED_INTEGRATION | End-to-end controller fixtures exercise promotion, positive hindsight, regression detection, rollback, and negative-result preservation. |
| Restart recovery | VERIFIED_UNIT | Durable campaign/run recovery tests prevent duplicate campaign claims and promotion records. |
| Literature scouting and transfer | VERIFIED_INTEGRATION | Configured scout dispatch creates evidence-candidate-only transfer records for later campaigns. |
| v2.2 implementation handoff | VERIFIED_INTEGRATION | `EngineeringTask` and verified `EngineeringResult` gate policy evaluation; the controller cannot write arbitrary repository files. |
| Benchmark authoring handoff | VERIFIED_UNIT | A discovered gap creates a `PREPARED` proposal only. It requires a new independently collected episode and leakage/provenance review; it is never admitted to the current candidate evaluation. |
| Provider adapter evolution | VERIFIED_INTEGRATION | Compatibility, live PASS/FAIL evidence, and evidence-gated policy descendant promotion are covered. |
| Live provider behavior | IMPLEMENTED_UNVERIFIED | Requires a configured provider runner and real held-out provider observations. |
| External coding execution | IMPLEMENTED_UNVERIFIED | Requires a configured engineering executor to perform and verify a bounded task. |
| PostgreSQL/service restart | IMPLEMENTED_UNVERIFIED | Unit-level durable recovery is covered; a deployed PostgreSQL restart smoke remains required. |

The latest local suite result was `266 passed, 19 skipped` with one existing
Pydantic settings warning; Ruff passed. This document is not a claim that
configured external-provider and executor integrations have been proven live.

## Runtime verification (read-only)

The deployed GPU-Lab `/health` endpoint returned `ok` with 148 tools and the
local runner enabled. The isolated Literature relay returned PaperQA `ready`.
The isolated Paper2Agent relay was reachable but reported `needs_credentials`
with credentials not configured, so no live engineering-executor task was
submitted or simulated.
PostgreSQL was reachable read-only with durable `research_objects` and
`research_events` rows present. No live service restart was performed because
it would interrupt the running user deployment.

## V3 requirement coverage

| Requirement | Status | Authoritative evidence |
| --- | --- | --- |
| Meta world model and agenda | VERIFIED_UNIT | `MetaResearchController` persists `MetaWorldModel` and `MetaResearchAgenda`; controller fixtures assert both records. |
| Autonomous triggers and prioritization | VERIFIED_INTEGRATION | Decision-outcome, postmortem, feedback, calibration, and model-change triggers feed one bounded controller pass. |
| Bounded controller and budgets | VERIFIED_UNIT | Durable campaign claims record all configured ceilings, consumption, stop reasons, and restart-resume state. |
| Science-vs-meta scheduling | VERIFIED_UNIT | Controller tests cover deferral to a more valuable domain agenda item and severe-failure override. |
| Targeted literature scouting | VERIFIED_INTEGRATION | The configured service dispatches one bounded problem-driven request and stores candidate-only transfers. |
| Internal invention and competing hypotheses | VERIFIED_UNIT | The policy lab creates mechanism-distinct candidate-generation, ranking, and critic hypotheses after a non-causal diagnosis. |
| Negative policy memory | VERIFIED_UNIT | Semantic fingerprints of `PolicyNegativeResult` prevent casual recreation of failed patches. |
| Bounded v2.2 implementation handoff | VERIFIED_INTEGRATION | Code patches require bounded `EngineeringTask` evidence before benchmark evaluation. |
| Evaluation and evaluator firewall | VERIFIED_UNIT | Frozen development/validation/held-out splits, evaluator audits, and tampering rejection are exercised in policy-lab tests. |
| Benchmark firewall and gap isolation | VERIFIED_UNIT | Candidates cannot inspect hidden data; discovered gaps become future-only authoring proposals, never current evaluation evidence. |
| Falsification, revision, and tournament | VERIFIED_UNIT | Adversarial held-out checks, hard regression rejection, bounded descendants, and tournament ranking have regression coverage. |
| Scoped promotion and autonomy config | VERIFIED_INTEGRATION | Project promotion/rollback sequences and DOMAIN/GLOBAL preflight conditions are covered. |
| Canary and shadow records | VERIFIED_UNIT | Canary stop conditions and counterfactual-unknown shadow records are persisted without fabricating unchosen outcomes. |
| Hindsight, calibration, regression, rollback | VERIFIED_INTEGRATION | Recording hindsight automatically invokes calibration and scoped rollback monitoring; failed policies create negative memory and a calibration agenda. |
| Provider adaptation | VERIFIED_INTEGRATION | Compatibility detection, live PASS/FAIL evidence retention, and scoped adapter descendants are covered; live provider proof is separate. |
| Durable events and restart recovery | VERIFIED_UNIT | Object/event persistence and campaign resume fixtures prevent duplicate runs and promotions. |
| Existing v2.1/v2.2/v2.5 coverage | VERIFIED_INTEGRATION | Full local suite passed with `269 passed, 19 skipped`; skipped PostgreSQL-only tests require an explicitly configured test database. |

## Required final-report outline

The source and test evidence above correspond to the requested v3 report:

- **Meta World Model, Meta Research Agenda, Improvement Opportunities, Autonomous Triggers, Meta Research Controller, Meta-Science Budgets, Science vs Meta-Science Scheduling:** implemented and unit/integration verified.
- **Autonomous Literature Scouting, Internal Policy Invention, Competing Policy Hypotheses, Policy Diagnostics, Policy Dead Memory:** implemented and verified without treating external literature as authority.
- **Automatic Implementation, Automatic Evaluation, Evaluator Firewall, Benchmark Firewall, Automatic Falsification, Automatic Revision, Policy Tournament:** implementation/evaluation paths are verified; live external code execution remains unverified until an executor is configured.
- **Project/Domain/Global Autonomy, Canary, Shadow Evaluation, Policy Hindsight, Calibration, Regression Detection, Automatic Rollback:** scoped paths and safeguards are covered in fixtures. Canary routing remains prospective because it requires real eligible decisions.
- **Model Change Handling, Provider Adapter Evolution, Benchmark Gap Discovery, Meta-Research ROI, Restart Durability:** code and local verification exist; live provider behavior and deployed restart recovery remain unverified.

## Remaining proof gaps

1. Restart the deployed `gpu-lab` service and verify health plus a durable
   campaign resume against PostgreSQL. This is deliberately pending operator
   approval because it interrupts the running service.
2. Configure a real provider to obtain cross-model held-out observations. Until
   then adapter candidates remain correctly marked `CROSS_MODEL_UNVERIFIED`.
3. Configure Paper2Agent only if live code-bearing policy patches must be
   exercised. Its absence does not block v3 semantic policy evolution.
