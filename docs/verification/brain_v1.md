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
- Research progress metrics and idempotent process-only MetaLessons with an evidence threshold for
  even evaluating a bounded campaign pilot.

## Real MCP, database, GPU, and learning tests

`uv run python scripts/brain_e2e_smoke.py` passed against the Docker MCP endpoint and PostgreSQL.
The run reloaded an actual saved VRCNet prediction with exact equality, found a related negative
mechanism through pgvector-backed search,
selected `CAUSAL_INTERVENTION`, reserved a retry-safe canonical run/job mapping, and only after
explicit human approval executed CUDA on an NVIDIA GeForce GTX 1650, ran the frozen HASI
hierarchical-state intervention, parsed the
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
  candidate experiments were deliberately not executed, so it is not scientific evidence. It also
  persisted/recovered the current `DO_NOT_BUILD_YET` campaign-readiness MetaLesson.
- **VERIFIED_INTEGRATION:** external workers live on internal-only networks and reach their one
  authenticated API through fixed-target relays. A public-address-only egress proxy permits HTTP(S)
  package/paper access while rejecting private, loopback, host-gateway, and MCP destinations. Direct
  worker calls to both `gpu-lab:8000/mcp` and `host.docker.internal:8000/mcp` receive 403, while
  GitHub and Crossref remain reachable. Restart readiness proves `tools/list` instead of trusting
  public health.
  This is reproducible with `uv run python scripts/worker_isolation_smoke.py` while both optional
  worker profiles are running.
- **VERIFIED_UNIT:** policy scoring, invalid action handling, finite agenda validation, the HASI
  reproduction-before-intervention gate, unavailable-provider alternative action, and QD niche,
  lineage, dead-idea, vector/structured-proximity, noncanonical embedding-cache behavior, branch
  priority/recovery/inspection policy, confound-preserving comparative memory, progress metrics,
  MetaLesson idempotency, and campaign-prematurity detection.
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
  longitudinal scientific value of QD/branch search and autonomous campaigns. The isolated
  Paper2Agent provider contract is implemented and unit-tested, but a model-backed build remains
  `IMPLEMENTED_UNVERIFIED` until a scoped Claude credential is supplied for the multi-hour run.

## Known risks and next highest-value build step

Brain v1 uses deterministic heuristics and caller-authored candidate experiments. PaperQA is now
integrated as an isolated optional provider, so the next literature task is a credentialed
model-backed quality evaluation rather than another adapter. Its real smoke proves the scientific
state machine learns from inspected evidence; it does not establish that one intervention
generalizes across models or datasets. Meta-review is implemented and currently rejects campaign
automation as premature. The next build step is to add sourced historical benchmark episodes and
inspected branch comparisons; a real Paper2Agent conversion should run separately when a specific
paper and task-scoped credential are approved.
