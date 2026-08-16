# Claude Code / Codex Workflow Study

## Scope

This is a design study, not a source import. The repository was inspected at
`tanbiralam/claude-code` and compared with the official Claude Code and Codex
workflow concepts: explore before editing, explicit planning, tool permission
boundaries, reusable skills, subagents, hooks, and progressive verification.

## External repository decision

The `tanbiralam/claude-code` repository is a large TypeScript reconstruction.
Its README describes missing Anthropic-internal modules and no-op stubs. No
standalone license file was available in the audited checkout. It is therefore
**DO NOT USE** as a runtime dependency and no source is copied into GPU-Lab.

The official Claude Code and Codex products are also not dependencies of the
scientific kernel. Provider-specific clients may drive the MCP surface, but
they cannot own ResearchState, evidence, hypothesis status, or experiment
truth.

## Portable patterns adopted

| External pattern | Native GPU-Lab boundary |
| --- | --- |
| Explore / plan / implement / test loop | `CodingExecutionPolicy` and `EngineeringTask` |
| Tool schemas and permission checks | typed MCP tools, workspace/security gates, and existing approval records |
| Reusable skills/procedures | `docs/ENGINEERING_EXECUTION_POLICY.md` and the provider-neutral policy contract |
| Hooks around tool execution | deterministic engineering guards, baseline gates, diff review, and execution readiness checks |
| Subagents / parallel workers | isolated PaperQA/Paper2Agent workers; no durable per-agent scientific memory |
| Persistent session memory | PostgreSQL `EngineeringTask`, `EngineeringResult`, events, and artifacts |

## Deliberately not adopted

- No Claude-specific SDK, prompt runtime, or copied agent implementation.
- No unrestricted shell or bypass-permissions mode for scientific execution.
- No separate agent memory that can diverge from PostgreSQL scientific truth.
- No automatic promotion from passing code/tests to scientific support.
- No general-purpose skill marketplace or autonomous multi-agent swarm.

## Verification boundary

Engineering hooks may block an implementation, detect drift, or mark an
implementation invalid. Only the existing Research OS assessment path may
update hypotheses, causal edges, WorldModel state, or strategy learning.
`EngineeringResult.scientific_result` remains `NOT_ASSESSED`.
