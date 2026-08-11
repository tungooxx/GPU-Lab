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
- Native mechanistic niches, ancestry, active/dead proximity retrieval, typed advisory QD
  operators, and explicit-difference gates for related failed ideas.
- Deterministic ExperimentBranches, ExperimentNodes, BranchRelations, and ComparativeLessons with
  inspected-result and unfinished-work gates; no MCTS policy is present.

## Real MCP, database, GPU, and learning tests

`uv run python scripts/brain_e2e_smoke.py` passed against the Docker MCP endpoint and PostgreSQL.
The run reloaded an actual saved VRCNet prediction with exact equality, found a related negative
mechanism through pgvector-backed search,
selected `CAUSAL_INTERVENTION`, submitted a retry-safe canonical run/job mapping, executed CUDA on
an NVIDIA GeForce GTX 1650, ran the frozen HASI hierarchical-state intervention, parsed the
persisted result artifact, recovered the completed result as `INSPECT_RESULT`, and explicitly
inspected the evidence. The edge changed from `HYPOTHESIZED_CAUSAL` to
`INTERVENTION_SUPPORTED`; the following `brain_step()` changed to `GENERALIZATION`.
The smoke also proved that the exact execution attempt remained `RESERVED` before explicit human
approval, then reused the same canonical run/job mapping after approval without submitting twice.

Candidate/decision persistence, result assessment, causal-edge/version updates, agenda appends, and
ResearchState fact appends now use locked PostgreSQL transactions so a failed request cannot leave a
partially promoted scientific result.

After `docker compose restart postgres gpu-lab`, the same project recovered three decisions, one
inspected run, five WorldModel versions, and continued with `GENERALIZATION`.

## Verification classification

- **VERIFIED_REAL:** local GPU experiment execution, inspected experiment evidence, evidence-driven
  hypothesis/edge/state update, and changed next Brain action.
- **VERIFIED_INTEGRATION:** PostgreSQL/MCP persistence, WorldModel version recovery, decision/run
  recovery, continuation after restart, and live QD discovery/dead-memory screening/lineage/niche
  recovery through `scripts/qd_e2e_smoke.py`. `scripts/branch_e2e_smoke.py` verifies branch MCP
  discovery, deterministic selection, typed-relation persistence, and restart recovery; its
  candidate experiments were deliberately not executed, so it is not scientific evidence.
- **VERIFIED_INTEGRATION:** the Compose backend has explicit gateway priority, so host/Caddy MCP
  calls remain on the allowed backend CIDR after restart while a direct Paper2Agent-worker MCP call
  still receives 403. Restart readiness now proves `tools/list` instead of trusting public health.
- **VERIFIED_UNIT:** policy scoring, invalid action handling, finite agenda validation, the HASI
  reproduction-before-intervention gate, unavailable-provider alternative action, and QD niche,
  lineage, dead-idea, vector/structured-proximity, noncanonical embedding-cache behavior, branch
  priority/recovery/inspection policy, and confound-preserving comparative memory.
- **VERIFIED_INTEGRATION:** PaperQA 2026.3.18 import/API compatibility, isolated worker health,
  authenticated gateway-to-worker connectivity, provenance normalization, canonical candidate
  import, unavailable-provider behavior, a non-root/capability-dropped worker with writable paper
  storage, and absence of Vast/SSH/PostgreSQL credentials in the worker environment.
- **VERIFIED_INTEGRATION:** the Paper2Agent HTTP/provider contract, generated MCP initialization,
  tool discovery and invocation against a real fixture MCP, exact upstream image checkout, pinned
  Claude Code runtime, non-root worker health, retry survival after client disconnect, network
  isolation, server-verified parameter-bound approval records, and absence of
  Vast/SSH/PostgreSQL/OpenAI credentials.
- **UNVERIFIED:** scientific generality of HASI, real model-backed PaperQA answer quality,
  longitudinal scientific value of QD/branch search, meta-review, and autonomous campaigns. The isolated
  Paper2Agent provider contract is implemented and unit-tested, but a model-backed build remains
  `IMPLEMENTED_UNVERIFIED` until a scoped Claude credential is supplied for the multi-hour run.

## Known risks and next highest-value build step

Brain v1 uses deterministic heuristics and caller-authored candidate experiments. PaperQA is now
integrated as an isolated optional provider, so the next literature task is a credentialed
model-backed quality evaluation rather than another adapter. Its real smoke proves the scientific
state machine learns from inspected evidence; it does not establish that one intervention
generalizes across models or datasets. The next build step is to benchmark the deterministic QD and
branch policies against additional historical episodes before deciding whether meta-review or a
campaign runtime is justified; a real Paper2Agent conversion should run separately when a specific
paper and task-scoped credential are approved.
