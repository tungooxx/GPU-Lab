# GPU-Lab / Research OS current state

Audit date: 2026-08-10. Verification labels distinguish source inspection, unit tests,
integration tests, and real execution.

## What exists

- `GPUService` owns Vast lifecycle, SSH checkout/environment setup, remote experiment jobs, logs,
  and artifacts. SQLite stores operational instance/job/audit data.
- `LocalRunner` executes detached commands inside one mounted workspace, manages persistent Python
  environments, streams logs, and exposes artifacts and GPU/runtime status.
- `ResearchStore` owns PostgreSQL projects, typed scientific objects, graph edges, immutable events,
  lexical negative-memory retrieval, and optional pgvector embeddings.
- Scientific objects already include papers, evidence, claims, mechanisms, anomalies,
  contradictions, hypotheses, predictions, experiments/runs/artifacts, reproductions, negative
  results, lessons, and ResearchState views.
- MCP exposes the execution and scientific service layer through Streamable HTTP. Tool metadata,
  schemas, safety annotations, and structured output are present for every discovered tool.

## What works

- Local and Vast execution services, preregistration, reproduction records, hypothesis proximity,
  immutable events, activity logging, and the web activity/terminal views are implemented.
- Research execution now reserves a canonical `experiment_id ↔ run_id ↔ job_id` mapping before
  process creation. Stable attempt keys make submission retry-safe, and sync accepts run or job ID.
- `local_env_prepare` resolves either an exact requirements file or a project directory and accepts
  an explicit Python executable.

## What is verified

- **VERIFIED_UNIT:** 15 tests cover provider normalization, path safety, structured output metadata,
  local requirements resolution, environment command construction, and local job idempotency.
- **VERIFIED_INTEGRATION:** Docker PostgreSQL migration, MCP discovery, atomic execution reservation,
  repeated submission, immutable `EXPERIMENT_STARTED` identity, job-ID sync, logs/artifacts, and
  completed-run persistence.
- **VERIFIED_REAL:** Python 3.13 with PyTorch 2.6.0+cu124 reproduced saved VRCNet predictions exactly
  (`maxabs=0`). This runtime is canonical for VRC internal interventions.

## What is partial

- pgvector storage/search exists, while automatic scientific embedding generation does not.
- Literature records and retrieval exist, but PaperQA is not integrated.
- Reproduction execution exists, but Paper2Agent is not integrated.
- ResearchState is a canonical view; it is not yet a versioned mechanistic WorldModel.
- Hypothesis similarity uses lexical structure unless callers provide embeddings.

## What is missing

- Native versioned WorldModel, ResearchAgenda, hypothesis portfolio/niches, ResearchDecision ledger,
  typed action candidates, and `brain_step()`.
- Result-analysis transitions that update predictions, hypotheses, world-model edges, and agenda
  items from inspected evidence.
- Brain benchmark episodes, evidence-dependent next-action regression, restart durability test, and
  autonomous campaign runtime.

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
