# External research-engine decisions

Audit date: 2026-08-11. These decisions protect PostgreSQL as the only canonical scientific memory.
They must be revisited before adding or upgrading an external dependency.

## PaperQA

- Official source: <https://github.com/Future-House/paper-qa>
- License: Apache-2.0.
- Audited API family: `paper-qa>=2026.3.18,<2027`, Python 3.11+, Apache-2.0. PaperQA's CalVer
  policy does not guarantee compatibility across releases, so the upper bound is intentional.
- Runtime: LiteLLM-compatible model access, optional Crossref/Semantic Scholar keys, and a large
  replaceable local document/index cache.
- Decision: **OPTIONAL DEPENDENCY in a SEPARATE WORKER**, behind `LiteratureProvider`.
- Implementation: the base gateway contains only `HttpLiteratureProvider`; the Compose
  `literature` profile installs PaperQA. The worker receives only its scoped token, paper volume,
  and optional literature/model credentials—not Vast, SSH, PostgreSQL, or repository secrets.
- Boundary: results become provenance-rich evidence candidates. PaperQA never updates canonical
  claims, hypotheses, WorldModel edges, or scientific statuses directly.

## Paper2Agent

- Upstream source: <https://github.com/jmiao24/Paper2Agent>
- Audited upstream commit: `e573687e15f345e3f375cd0851373d588e436be3` (default branch HEAD
  at audit time).
- Architecture: long-running coding-agent workflow that inspects paper repositories and generates
  tested MCP servers; it depends on Claude Code, external model authentication, Python/FastMCP,
  and creates an isolated environment, reports, tests, generated code, and replaceable worker state.
  Upstream estimates 30 minutes to 3+ hours and about USD 15 for a complex repository.
- License: MIT for Paper2Agent. Claude Code is a separately licensed external runtime and is only
  installed in the optional worker image; no upstream Paper2Agent source is copied into our package.
- Decision: **SUBPROCESS WORKER in a SEPARATE SERVICE** behind `ExecutablePaperProvider`.
- Implementation: the `paper-agents` Compose profile checks out the exact audited upstream commit,
  pins its Claude Code runtime, accepts only public GitHub repositories, and receives only its
  scoped token plus Anthropic credential. PostgreSQL, Vast, SSH, GPU, and literature credentials are
  absent. Network policy prevents the worker subnet from calling GPU-Lab's MCP endpoint directly.
- Paid builds and generated-code invocation require a separate server-authenticated `ActionApproval`
  record bound to an approver label, exact parameter hash, rationale, and expiry. Approval parameters
  themselves are not persisted, avoiding storage of tool arguments that may contain secrets.
- Boundary: generated tools are unverified executable-paper candidates until GPU-Lab smoke and
  reproduction gates pass. Worker verification is capped at `VERIFIED_INTEGRATION`; it cannot
  claim `VERIFIED_REAL` or mutate canonical claims, hypotheses, experiments, or WorldModel state.

## Kaimen Co-Scientist

- Official source: <https://github.com/Kaimen-Inc/Co-Scientist>
- License: Apache-2.0.
- Architecture: Python multi-agent supervisor with its own SQLite queue/state and API or subscription
  CLI backends.
- Decision: **PROMPT / DESIGN INSPIRATION and ALGORITHM PORT**, not a runtime dependency.
- Reuse concepts: generation, reflection, ranking, evolution, proximity, and meta-review as temporary
  typed `ResearchOperator` calls. Do not import Kaimen's SQLite state or make it canonical.

## IDEAgent

- Official source identified by the paper: <https://github.com/declare-lab/IDEAgent>
- Architecture: quality-diversity idea evolution using lineages, completed/rejected idea comparison,
  repair, and refinement.
- Decision: **DESIGN INSPIRATION ONLY** until the repository license and dependency surface are
  verified directly. Port only the abstract niche/lineage/search policy into native typed services.
- Boundary: QD scores affect which hypothesis to test, never which hypothesis is scientifically true.

## MARS reflective search

- Paper: <https://arxiv.org/abs/2602.02660>
- Relevant concepts: budget-aware branch search, modular experiment construction, and comparative
  reflective memory.
- No authoritative implementation/license was identified in this audit.
- Decision: **PAPER / DESIGN INSPIRATION ONLY**. Brain v1 will use deterministic heuristic action
  scoring and branch records; it will not implement MCTS yet.

## Dependency policy

Base GPU-Lab must work without any of these engines. Heavy or conflicting systems belong in extras,
subprocesses, or isolated services. External workers receive only task-scoped inputs and never Vast,
SSH, database, repository, or unrelated API credentials.
