# GPU-Lab / Research OS current state

Audit date: 2026-08-11. Verification labels distinguish source inspection, unit tests,
integration tests, and real execution.

## What exists

- `GPUService` owns Vast lifecycle, SSH checkout/environment setup, remote experiment jobs, logs,
  and artifacts. SQLite stores operational instance/job/audit data.
- `LocalRunner` executes detached commands inside one mounted workspace, manages persistent Python
  environments, streams logs, and exposes artifacts and GPU/runtime status.
- `ResearchStore` owns PostgreSQL projects, typed scientific objects, graph edges, immutable events,
  lexical negative-memory retrieval, and optional pgvector embeddings.
- Scientific objects include papers, evidence, claims, mechanisms, anomalies, contradictions,
  hypotheses, predictions, experiments/runs/artifacts, reproductions, negative results, lessons,
  WorldModel nodes/edges/versions, AgendaItems, portfolios, action candidates, decisions, and
  ResearchState views.
- MCP exposes the execution and scientific service layer through Streamable HTTP. Tool metadata,
  schemas, safety annotations, and structured output are present for every discovered tool.

## What works

- Local and Vast execution services, preregistration, reproduction records, hypothesis proximity,
  immutable events, activity logging, and the web activity/terminal views are implemented.
- Research execution now reserves a canonical `experiment_id ↔ run_id ↔ job_id` mapping before
  process creation. Stable attempt keys make submission retry-safe, and sync accepts run or job ID.
- `local_env_prepare` resolves either an exact requirements file or a project directory and accepts
  an explicit Python executable.
- Brain v1 selects one agenda-driven action using inspect/recovery and reproduction gates followed
  by a transparent information-per-cost heuristic. Every candidate and decision is stored with its
  state snapshot, evidence references, critic advice, rationale, and costs.
- Explicit result assessment records experiment evidence and updates the run, hypothesis,
  AgendaItem, ResearchDecision, ResearchState, and optional causal edge/WorldModelVersion.

## What is verified

- **VERIFIED_UNIT:** 21 tests cover provider normalization, path safety, structured output metadata,
  local requirements resolution, environment command construction, local job idempotency, action
  scoring, the HASI reproduction gate, and unavailable-provider fallback behavior.
- **VERIFIED_INTEGRATION:** Docker PostgreSQL migration, MCP discovery, atomic execution reservation,
  repeated submission, immutable `EXPERIMENT_STARTED` identity, job-ID sync, logs/artifacts, and
  completed-run persistence.
- **VERIFIED_REAL:** Python 3.13 with PyTorch 2.6.0+cu124 reproduced saved VRCNet predictions exactly
  (`maxabs=0`). This runtime is canonical for VRC internal interventions.
- **VERIFIED_REAL:** `scripts/brain_e2e_smoke.py` used PostgreSQL, MCP, the persistent canonical
  environment, and a GTX 1650 to prove `brain_step before -> real evidence -> brain_step after`.
- **VERIFIED_INTEGRATION:** after restarting PostgreSQL and GPU-Lab, the same project recovered its
  decisions, inspected run, WorldModel history, and evidence-dependent next action.

## What is partial

- pgvector storage/search exists, while automatic scientific embedding generation does not.
- Literature records and retrieval exist, but PaperQA is not integrated.
- Reproduction execution exists, but Paper2Agent is not integrated.
- Hypothesis similarity uses lexical structure unless callers provide embeddings.
- Brain v1 critics are deterministic advisory checks; LLM-backed typed ResearchOperators are not
  integrated.

## What is missing

- Automatic embedding generation, hypothesis niches/lineages, full comparative/meta-research
  memory, experiment branches, and autonomous campaign runtime.
- PaperQA and executable-paper provider adapters.
- Additional historical benchmark episodes beyond the permanent HASI gate.

## What should be reused

- Preserve PostgreSQL scientific objects/events, pgvector support, execution services, LocalRunner,
  Vast provider, preregistration, reproduction, negative results, lessons, audit DB, MCP transport,
  and activity UI.
- Add Brain v1 as typed native services over the existing ResearchStore.
- Wrap literature and executable-paper engines behind optional provider interfaces.

## What should not be rewritten

- Do not duplicate SSH, GPU lifecycle, repository checkout, process control, logs, artifacts, or
  telemetry in the research brain.
- Do not replace PostgreSQL scientific truth with external agent memory.
- Do not implement a prompt-only brain, full MCTS, agent swarm, or campaign runner before one
  evidence-sensitive `brain_step()` works end to end.
