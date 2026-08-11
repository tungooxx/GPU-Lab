# GPU Lab MCP

GPU Lab is a minimal, safe MCP experiment control plane for Vast.ai GPUs. It deliberately exposes structured lifecycle, repository, environment, job, log, and artifact operations rather than a general SSH shell.

## Run

```bash
cp .env.example .env
# set VAST_API_KEY and GPU_LAB_SSH_PRIVATE_KEY_PATH in .env
uv sync
uv run gpu-lab-mcp                         # stdio for local MCP clients
uv run gpu-lab-mcp --transport streamable-http # HTTP at /mcp (port 8000)
uv run gpu-lab-mcp --transport sse              # legacy SSE clients at /sse (port 8000)
```

For development, connect a local MCP client to the stdio process. For production, run the streamable HTTP transport behind authenticated HTTPS/private networking. The MCP gateway, not ChatGPT, holds provider and SSH credentials.

## Tools

The MCP exposes GPU lifecycle, local and Vast execution, logs/artifacts, literature records,
reproduction, canonical scientific state, and Brain v1 operations. Use MCP discovery for the full
typed list. Brain tools include `world_model_create`, `world_model_get`, `world_entity_create`,
`causal_edge_create`, `causal_edge_update`, `research_agenda_create`,
`research_agenda_item_create`, `hypothesis_portfolio_get`, `brain_step`, and
`brain_result_assess`.

Use `gpu_search` before `gpu_create`; creation requires a specific offer rather than making an unbounded cost decision. `gpu_destroy` requires `confirmation="DESTROY"`.

## Research OS workflow

With `docker compose up -d --build`, PostgreSQL holds durable Research OS state alongside the
GPU-Lab service. Start every serious investigation with `research_state_get`, then use this
sequence:

```text
research_project_create
→ paper_ingest / paper_evidence_create / claim_create
→ anomaly_create / hypothesis_related / hypothesis_create
→ experiment_plan_register
→ research_experiment_execute / research_experiment_sync
→ research_assess / negative_result_create / lesson_create
→ research_state_update / research_state_get
```

`experiment_plan_register` freezes the question, competing explanations, intervention and
control, metrics, expected direction, pass/fail interpretations, and estimated time/cost before
execution. `research_experiment_execute` first reserves a canonical PostgreSQL mapping of
`experiment_id`, `run_id`, and `job_id`, then submits a deterministic local job. Pass a stable
`execution_attempt_uuid` when retrying a request; retries return the same mapping and do not launch
a duplicate job. If no key is supplied, identical requests use an automatically derived key.
`research_experiment_sync` accepts either `run_id` or `job_id` and always returns the complete
canonical mapping. Commands, runtime, logs, exit code, and artifacts are preserved in the run and
immutable event history. Do not assess a hypothesis from reasoning alone when a local or provider
experiment is feasible.

Local environments are persistent under `GPU_LAB_LOCAL_ENV_ROOT`. Docker Compose stores that root
in the `gpu-lab-envs` named volume, so environments remain available across container rebuilds and
are not installed on the Windows workspace mount. `local_env_prepare` accepts an
exact requirements file or a directory containing `requirements.txt`, plus an explicit
`python_executable`. The canonical VRCNet internal-intervention runtime verified in August 2026 is
the environment named `vrc-py313-torch260-cu124`, created with Python 3.13 and containing PyTorch
2.6.0+cu124. It reproduced saved predictions exactly (`maxabs=0`). This is a project-specific real
verification, not a claim that the gateway container itself uses that runtime.

`paper_ask` is retrieval only: use its cited `EvidenceUnit` IDs when creating a claim. It does not
turn retrieved prose into a scientific conclusion. `hypothesis_create` checks related failed ideas
and requires `scientific_difference` when a proposed mechanism resembles stored negative knowledge.

For a reproduction, use `reproduction_prepare`, `reproduction_run`, `reproduction_sync`, and
`reproduction_compare`. A successful process is only `PARTIAL` until its observed metric is compared
with the reported metric and tolerance; it becomes `REPRODUCED` only after that explicit comparison.

## Optional PaperQA literature worker

PaperQA is isolated from GPU/Vast/SSH/PostgreSQL credentials. Set a strong
`GPU_LAB_LITERATURE_WORKER_TOKEN`, set `GPU_LAB_LITERATURE_PROVIDER=paperqa-http`, configure only
the model/metadata credentials needed by PaperQA, and start:

```bash
docker compose --profile literature up -d --build
```

Use `literature_provider_status`, `literature_search`, and `literature_ask` for read-only candidates.
`literature_gather` imports provenance-rich candidates into PostgreSQL and can create an unresolved
scoped Claim, but it never marks a claim or hypothesis supported. The PaperQA index is a replaceable
cache; PostgreSQL remains scientific truth.

## Local CLI

```bash
uv run gpu-lab gpu list
uv run gpu-lab gpu status vast_123
uv run gpu-lab experiment status exp_abc
uv run gpu-lab experiment logs exp_abc --tail 100

cd /workspace/GPU-Lab

GPU_LAB_ALLOWED_HOSTS='127.0.0.1:*,localhost:*,chucky-lab.com' \
uv run gpu-lab-mcp --transport streamable-http
```
```bash
cd ~/.ssh

ssh-keygen -t ed25519 \
  -f /root/.ssh/gpu_lab_ed25519 \
  -N "" \
  -C "gpu-lab-self-ssh"

cat /root/.ssh/gpu_lab_ed25519.pub >> /root/.ssh/authorized_keys

chmod 600 /root/.ssh/gpu_lab_ed25519
chmod 600 /root/.ssh/authorized_keys
```
## Security

The server never returns API keys or private keys, only allows repository and artifact paths under `GPU_LAB_REMOTE_ROOT`, refuses dirty checkouts, bounds logs/artifact reads, and does not enable `remote_exec` unless `GPU_LAB_ENABLE_REMOTE_EXEC=true`. SSH requires `GPU_LAB_SSH_KNOWN_HOSTS` by default. Vast hosts can change, so the explicitly named `GPU_LAB_SSH_ALLOW_UNVERIFIED_HOSTS=true` override is available for development only; production should pin host keys or use a trusted gateway network.

## Limitations

The execution gateway currently has one provider (Vast), SQLite operational state, tmux-based
remote jobs, detached local jobs, and a compact CLI. Scientific Research OS state and immutable
scientific events are stored separately in PostgreSQL with optional pgvector retrieval.
`artifact_download` and `experiment_summary` are intentionally not exposed yet; large remote files
remain on the worker. Vast endpoint shapes are normalized defensively and still require live
provider integration checks before production use. Brain v1 has a native versioned WorldModel,
ResearchAgenda, hypothesis portfolio, decision ledger, deterministic information-per-cost policy,
unfinished-work recovery, and explicit result assessment. Its real smoke has exercised the vertical
slice on PostgreSQL and a local GTX 1650. PaperQA is integrated as an optional isolated provider;
its real model-backed answer quality is not yet verified. Paper2Agent is also integrated behind an
optional isolated executable-paper worker pinned to an audited upstream commit. Its provider,
generated-MCP inspection/invocation, network isolation, and canonical-truth boundary are verified,
but a paid model-backed paper conversion has not been run. Quality-diversity operators, experiment
branching, and campaign automation remain later milestones.

Enable the Paper2Agent worker only with a task-scoped Anthropic credential and explicit approval of
the upstream model cost:

```powershell
$env:GPU_LAB_EXECUTABLE_PAPER_PROVIDER = "paper2agent-http"
$env:GPU_LAB_EXECUTABLE_PAPER_WORKER_TOKEN = "<random-task-scoped-token>"
$env:ANTHROPIC_API_KEY = "<task-scoped-key>"
docker compose --profile paper-agents up -d --build
```

Instead of an API key, you may authenticate Claude Code interactively into its isolated persistent
volume with `docker compose --profile paper-agents run --rm paper2agent claude`. The credential is
not mounted into GPU-Lab or PostgreSQL.

The worker accepts public GitHub repositories only. A generated tool remains non-evidentiary until
it is used inside a preregistered GPU-Lab reproduction or experiment and the result is inspected.
