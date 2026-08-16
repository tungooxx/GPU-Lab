# v2.5 Research Policy Lab summary

## Actual v2.2 audit

See `audit.md`. The policy lab reuses the existing durable object store,
strategy-learning eligibility, temporal benchmark fixtures, and isolated
literature worker boundary.

## Research policy representation

`ResearchPolicy` is versioned, immutable-by-promotion, and has explicit parent,
provenance, semantic policy sections, provider-adapter data, applicability, and
known failure modes. Production policy is not changed by `/improve`.

## Policy hypothesis, patch, experiment, and negative result

`PolicyHypothesis`, `ResearchPolicyPatch`, `PolicyExperiment`, and
`PolicyNegativeResult` are durable typed objects. Patches have structured change
metadata and semantic fingerprints. Equivalent failed patches are rejected before
benchmarking. Invalid/no-op patches are explicitly recorded without evaluation.

## Improvement run and interface

`improve_start` supports internal history, user idea, paper, failure, component,
and targeted search modes. It generates several distinct candidates, evaluates
them automatically by default, rejects hard regressions by default, and performs
bounded revision. High-level tools expose status, compare, export, promote,
rollback, transfer classification, and hindsight recording.

## Benchmark coverage and isolation

Frozen episodes are split into development, validation, and held-out partitions.
The benchmark only exposes `visible_payload`; policy experiments persist under
the `BENCHMARK` namespace and do not write WorldModel, evidence, agenda, or
production strategy state. Evaluation compares a candidate with the v2 baseline
and blocks leakage, scope, bad-action, and premature-architecture regressions.

## Transfer, portability, promotion, and rollback

Transfer evidence is classified as project-specific, cross-project-supported,
rejected, or model-sensitive. Policy export is provider-neutral and serializes
adapter data without executing it. Promotion is explicit; rollback retains
history. Post-promotion hindsight records observed improvement, cost, unexpected
failure, and calibration data.

## Security / external content

Paper and literature content is untrusted content. Search requires the isolated
PaperQA provider and is explicit when unavailable. Extracted principles are small
semantic adaptations; external text cannot authorize commands or policy changes.

## Verified unit and integration checks

- Full test suite after the policy-lab changes: `218 passed, 19 skipped`.
- Ruff: `ruff check src tests` passed.
- Focused policy, benchmark, and MCP metadata tests cover automatic evaluation,
  duplicate rejection, held-out regression rejection, no-op rejection, bounded
  revision, transfer classification, export, rollback, and hindsight isolation.

## Implemented but unverified / remaining risks

- Historical corpus coverage is still the small existing fixture set; more
  real-project episodes are needed before broad promotion claims.
- Cross-project and cross-model classifications accept supplied evidence; no
  external model/provider evaluation is claimed.
- Targeted literature search is provider-dependent and has not been executed in
  this environment.
- CodeRabbit review was explicitly waived by the user. Local test and lint
  evidence remains the verification basis for this handoff.
- No autonomous continuous self-evolution is implemented; this is intentional.
