# v2.5 Research Policy Lab audit

Audit date: 2026-08-16. This document records the implementation that exists on
the `feat/research-brain-v2` branch; it is not inferred from historical prompts.

## Current policy representation

Research behaviour is currently distributed across deterministic Python logic and
version constants rather than represented by one durable policy object. The
important sources are `brain.py` (selection/scoring), `strategy.py`
(`STRATEGY_POLICY_VERSION` and `SCORING_POLICY_VERSION`), and `brain_bench.py`
(the named built-in benchmark policies). `ResearchDecision` stores the selected
action, comparisons, critics, scope, classification and policy-version context.
It is a decision trace, not an immutable, versioned representation of the
research policy that produced it.

## Current strategy memory

`ResearchSituation`, `ResearchDecisionOutcome`, `ResearchStrategyPattern`,
`StrategyOutcome`, `NegativeResult`, `ComparativeLesson`, and `MetaLesson`
already provide scoped process memory. Strategy learning is fail-closed through
`strategy_learning_eligibility`; closed scientific cycles are distinguished from
system, benchmark, and administrative activity. `NegativeResult` and retrieved
strategy patterns can constrain future scientific decisions. There is currently
no separate memory of failed *policy* changes.

## Current benchmarks

`ResearchBrainBench` loads JSON historical episodes from `research_bench/`.
Episodes have a time cutoff, visible and hidden state, frozen candidate actions,
answer labels, provenance, and a multi-metric scorecard. The visible payload
excludes answer keys, costs, tags, expected information, and hidden future state.
Existing fixtures cover three historical scenarios and compare reproducible
heuristic baseline policies. There are no development/validation/held-out splits,
policy experiments, candidate-policy runners, or promotion gates.

## Current meta-review

`MetaResearchService.meta_review` computes research-process counts, detects
unfinished and uninspected work, incomplete reproduction, repeated failed
assumptions, missing branch comparison, and repeated low-value action types. It
persists an idempotent `MetaLesson` and deliberately does not modify scientific
claims. It does not turn these signals into ranked policy weaknesses, policy
hypotheses, patches, or experiments.

## Current closed-cycle coverage

Research decision classification tracks decision role, scientific role,
execution/scientific verification, cycle status, and learning namespace.
Strategy eligibility requires an appropriate scientific closed cycle. This gives
v2.5 a reliable source for internal weakness detection. It does not yet capture
policy-evaluation runs separately from production science.

## Current literature integration

Literature is accessed through an optional isolated PaperQA HTTP worker.
Literature answers are persisted as evidence candidates and are explicitly not
canonical scientific truth. Existing embedding support is local hash embedding
plus structured/lexical retrieval. No path extracts a paper's method into a
research-policy principle or triangulates competing policy methods.

## Current prompt / policy sources

The repository's operative research policy is deterministic code and persisted
decision context, not a mutable system prompt. Provider-specific integration is
confined to isolated literature/executable-paper workers and their validated
settings. No runtime mechanism silently rewrites instructions from external
content.

## Current provider-specific instructions

Settings select `disabled` or HTTP-backed literature, research-operator, and
executable-paper providers. Worker credentials remain server-side. MCP tools are
registered in `server.py` and operationally bounded by approval mechanisms where
execution is dangerous. v2.5 needs a provider-neutral policy compiler that can
emit explicitly recorded provider adapters without changing these safeguards.

## Gaps addressed by v2.5

1. No canonical immutable `ResearchPolicy` or promotion/rollback history.
2. No `PolicyHypothesis`, structured `ResearchPolicyPatch`, `PolicyExperiment`,
   `PolicyNegativeResult`, `ResearchPolicyWeakness`, or `ImprovementRun`.
3. No automatic, bounded `/improve` orchestration from history, user ideas,
   project scope, failure signal, paper, or targeted literature search.
4. No candidate de-duplication against failed policy work, multi-candidate
   comparison, automatic rejection/revision, or transparent promotion criteria.
5. Benchmark coverage and split discipline are insufficient for policy tuning;
   policy evaluation must remain isolated from `PRODUCTION_SCIENCE`.
6. No policy semantic diff, model/project transfer classification, or
   provider-neutral policy export/compiler.
