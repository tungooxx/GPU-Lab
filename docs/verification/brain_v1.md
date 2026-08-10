# Brain v1 verification report

Verification date: 2026-08-11. This report separates implementation evidence from scientific
evidence. A successful tool call or unit test is not treated as proof of a mechanism.

## Existing components reused

- PostgreSQL scientific objects, graph edges, immutable ResearchEvents, ResearchState, and pgvector.
- Preregistration, atomic/idempotent experiment execution, local runner, persistent environments,
  logs, artifacts, reproduction gates, negative results, and MCP observability.

## New native components

- Typed WorldModel entities, evidence-bearing CausalEdges, and WorldModelVersion history.
- ResearchAgenda and AgendaItem, HypothesisPortfolio, ResearchActionCandidate, and ResearchDecision.
- Deterministic `brain_step()` with unfinished-result recovery, reproduction gating, dead-idea
  retrieval, advisory critics, and explicit information/cost/risk components.
- `brain_result_assess` for inspected evidence and provenance-bearing state transitions.
- HASI historical benchmark fixture and a real Brain E2E smoke.

## Real MCP, database, GPU, and learning tests

`uv run python scripts/brain_e2e_smoke.py` passed against the Docker MCP endpoint and PostgreSQL.
The run reproduced the baseline, found a related negative mechanism through pgvector-backed search,
selected `CAUSAL_INTERVENTION`, submitted a retry-safe canonical run/job mapping, executed CUDA on
an NVIDIA GeForce GTX 1650, recovered the completed result as `INSPECT_RESULT`, and explicitly
inspected the evidence. The edge changed from `HYPOTHESIZED_CAUSAL` to
`INTERVENTION_SUPPORTED`; the following `brain_step()` changed to `GENERALIZATION`.

After `docker compose restart postgres gpu-lab`, the same project recovered three decisions, one
inspected run, five WorldModel versions, and continued with `GENERALIZATION`.

## Verification classification

- **VERIFIED_REAL:** local GPU experiment execution, inspected experiment evidence, evidence-driven
  hypothesis/edge/state update, and changed next Brain action.
- **VERIFIED_INTEGRATION:** PostgreSQL/MCP persistence, WorldModel version recovery, decision/run
  recovery, and continuation after restart.
- **VERIFIED_UNIT:** policy scoring, invalid action handling, finite agenda validation, the HASI
  reproduction-before-intervention gate, and unavailable-provider alternative action.
- **UNVERIFIED:** scientific generality of HASI, PaperQA, Paper2Agent, quality-diversity operators,
  branch search, meta-review, and autonomous campaigns.

## Known risks and next highest-value build step

Brain v1 uses deterministic heuristics and caller-authored candidate experiments. Its real smoke
proves the scientific state machine learns from inspected evidence; it does not establish that one
intervention generalizes across models or datasets. The next build step is an optional,
provenance-preserving `LiteratureProvider` adapter, followed by executable-paper isolation only after
license and dependency verification.
