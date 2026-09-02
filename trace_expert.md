# GPU-Lab Expert Trace

## Scope

This file is a compact expert handoff for diagnosing and repairing GPU-Lab / Research OS execution behavior. It is an index to the raw trace exports, not a replacement for them.

## Raw traces

- `trace_export.md` — full historical MCP trace export.
- `trace_export_point_cloud_2026-08-16_2026-08-17.md` — point-cloud research trace for the named interval.

Both exports are intentionally preserved unchanged.

## Canonical source and provenance

- Repository: `D:\Newgeneration\MCP-GPT`
- Active branch: `fix/research-os-control-plane`
- Canonical source mount introduced in commit `6cbc4fc`:
  `/workspace/gpu-lab-source` (read-only inside GPU-Lab)
- Inspect source/runtime alignment through `runtime_code_version`.
- A running container must be recreated before the source mount becomes available.

## Execution safety state

- Remote execution is launched in a dedicated process group.
- Cancellation distinguishes:
  - `cancelled`: process group termination was verified;
  - `CANCELLATION_PENDING_VERIFICATION`: termination is not yet proven;
  - `CANCELLATION_INCOMPLETE`: the process group remains alive.
- `experiment_status` must not turn a verified cancellation into `unknown`.
- `experiment_submit(timeout_seconds=...)` now enforces a remote wall-time limit and records `MAX_WALL_SECONDS` on timeout.

## Provenance and scheduler repairs already present

- Repository checkout resolves a fetched revision to one verified commit SHA before detached checkout.
- Dependency updates require explicit terminal status predicates unless an existence-only dependency is explicitly requested.
- Stale gate versions can be superseded without superseding their scientific subject.

## Open high-priority work

1. Full-path GPU memory canary for expensive scientific paths.
2. Structured CUDA OOM event capture and fail-fast handling.
3. Cost/idle/no-progress circuit breakers beyond wall-time enforcement.
4. Consolidated execution-gate diagnostics.
5. More complete GPU process/VRAM telemetry in experiment status.

## Investigation rules

- Treat technical execution failure as technical failure, never as scientific evidence.
- Do not restart a live MCP service merely to inspect source; use `runtime_code_version` first.
- Do not run expensive scientific workloads before their exact runtime and memory canary are valid.
- Preserve raw trace exports and record commit SHA, container/runtime version, run ID, and job ID with every diagnosis.
