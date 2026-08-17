# Research Brain v3.1 audit

Audit date: 2026-08-17. This note describes the repository before the v3.1
discovery-search implementation.

## CURRENT

- `ResearchBrain.brain_step()` is the production decision path exposed by
  `server.brain_step`. It reads `ResearchStore.state_get`, the active
  `WorldModel`, agenda, hypotheses, lessons, and historical strategy patterns,
  then atomically persists a `ResearchSituation`, `ResearchActionCandidate`,
  and `ResearchDecision` through `ResearchStore.brain_decision_create`.
- Candidate generation is in `ResearchBrain._candidate_actions()`. It owns
  deterministic prerequisites (uninspected results, recovery, reproduction,
  null controls) and otherwise ranks only agenda-configured candidates. It
  does not construct a scientifically diverse discovery portfolio.
- Candidate ranking is split between `ActionScore.priority` and
  `ResearchStrategyService.adjust_candidates()`. The latter retrieves strategy
  patterns and applies a small diminishing-return adjustment.
- A durable `HypothesisPortfolio` exists, but it is a snapshot of active/dead
  hypotheses and niches, not a portfolio of alternatives for one decision.
- QD/lineage supports parent hypotheses, niches, structured similarity and
  negative-result proximity screening. Similarity is correctly advisory, but
  no architecture-family or scientific-distance policy is used by `brain_step`.
- `ResearchOperatorService` offers typed, advisory hypothesis generation and
  critic calls. It validates model output and stores provenance, but is not
  orchestrated by the production Brain decision path.
- `ResearchBrainBench` provides frozen policy episodes and comparison of
  existing policies. It has no v3.1 search-regime/portfolio regressions or
  production-v3.1 shadow comparison.
- The dashboard is a WorldModel graph and operational activity view. It does
  not render decision portfolios or strategic regime state.

## MISSING

- Typed `SearchRegime`, `ScientificDistance`, `FrontierGap`, `StagnationState`,
  `BreakthroughSignal`, and decision-scoped `CandidatePortfolio`.
- Deterministic regime transitions that turn local saturation/frontier gap into
  widened candidate generation.
- Portfolio adequacy and fake-diversity validation for open-ended decisions.
- Structured scientific-distance classification based on changed dimensions.
- Frontier comparability data and state-aware selection behavior.
- Discovery branching for a scientifically refuted partial breakthrough.
- State version/freshness provenance on every production decision.
- v3.1 benchmark episodes, shadow utility, v3.1 smoke, and point-cloud shadow
  report.

## PARTIALLY IMPLEMENTED

- The existing strategy service detects repeated low-information action types
  and changes their score, but does not widen the search space.
- `ResearchDecision` already stores a runner-up index, candidate comparison,
  and a basic rationale. It does not preserve a portfolio, regime, distance
  coverage, frontier state, or a scientific selected-vs-runner-up explanation.
- `ResearchSituation` carries a stage and basic resource constraints, but not
  strategic freshness, saturation, frontier, or breakthrough context.
- QD/negative memory identifies related dead ideas, but currently constrains
  local candidates rather than requiring a broader scientific radius.

## BUGGY / NOT WIRED

- Production decisions still persist `brain-v2-strategy-augmented-v1`; v3.1
  behavior is absent from the actual `brain_step` path.
- The existing `HypothesisPortfolio` name masks a different concept: it cannot
  prove that a decision considered multiple scientific alternatives.
- A single configured candidate is selected for open-ended work because no
  portfolio adequacy gate exists. The only current one-candidate exception is
  implicit (`ARTIFACT_ANALYSIS`/`REPRODUCTION`), not durable provenance.
- `project.state` is returned by `state_get` alongside canonical records but
  decisions do not record freshness/version metadata sufficient to prevent or
  diagnose stale snapshot use.

## VERIFIED

- Scientific execution/result separation, frozen experiment controls,
  immutable event history, run/job identity, inspected-result handling,
  EvidenceFamily independence, scope-aware evidence, and held-out firewalls
  are implemented in the current Brain/ResearchStore and are not replaced by
  v3.1.
- Decisions, candidates, situations, world-model records, branches, negative
  results, and strategy outcomes are persisted as Research OS objects with
  event provenance.
- LLM operators are advisory and typed; they do not directly mutate scientific
  truth.
