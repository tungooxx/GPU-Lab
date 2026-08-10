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
- Architecture: long-running coding-agent workflow that inspects paper repositories and generates
  tested MCP servers; it depends on an external coding-agent runtime and creates its own outputs.
- License: no clear repository license was confirmed during this audit.
- Decision: **DO NOT IMPORT OR COPY SOURCE** until licensing is explicit. Keep an
  `ExecutablePaperProvider` boundary and later evaluate a subprocess/container integration using
  only authorized public interfaces.
- Boundary: generated tools are unverified executable-paper candidates until GPU-Lab smoke and
  reproduction gates pass. They cannot become scientific truth themselves.

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
