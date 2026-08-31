# v3.6.6 final implementation report

## Scope and verification

v3.6.6 is an additive correction layer. It preserves v3.6.5 authority,
execution provenance, and confirmation sealing. Verification on 2026-08-31:
`400 passed, 111 skipped` on Windows, plus PostgreSQL correction, discovery,
Lab, and scheduler regressions. The live container was rebuilt after the
control-plane fixes and its loaded source hashes match its mounted checkout.

## A–D. Audit, gap, burden, and taxonomy

- **A. Review-system audit:** `ScientificSelfCorrectionService.production_audit`
  reports review-like work, layer classifications, repeated chains, missing
  major-design correction, V5 coverage, interpretation coverage, and
  confirmation integrity.
- **B. Scientific self-correction gap:** the audit exposes un-attacked
  assumptions, missing discriminators, material unresolved challenges, and
  correction deficits instead of treating a generic review as closure.
- **C. Generic code-review burden:** `LabController.create_work` suppresses
  generic code-review roles under `LEAN_ENGINEERING_REVIEW_V366`; the audit
  identifies deterministic-review conversion opportunities.
- **D. Taxonomy:** durable records and cockpit output distinguish design attack,
  implementation fidelity, result interpretation, and deterministic
  engineering.

## E–L. Scientific design correction

- **E. Schema changes:** `correction_v366.py` registers `ScientificDesign`,
  `DesignAssumption`, `ScientificDesignAttack`, `DesignChallenge`,
  `DesignAttackSynthesis`, `RequiredDiscriminator`,
  `ImplementationFidelityCheck`, `ResultInterpretationAttack`,
  `ReviewTargetClassification`, hindsight, and regression-proposal objects.
- **F. DesignAssumption:** first-class typed, centrality-ranked, falsifiable
  assumption records have provenance, support, and challenge fields.
- **G. ScientificDesignAttack:** attacks are bound to frozen design hash,
  attacker worker/session, frozen lineage, independent batch, and structured
  verdicts; they cannot admit evidence.
- **H. DesignChallenge:** challenges hold target assumption, alternative or
  degenerate explanation, failed controls, prediction, discriminating test,
  severity, verification, and resolution.
- **I. Competing explanations:** competing and degenerate explanations are
  structured attack/challenge inputs, not only prose.
- **J. DesignAttackSynthesis:** synthesis deduplicates only after peer freeze;
  it records unresolved material challenges, discriminators, repairs, and a
  non-voting verdict.
- **K. Independent isolation:** one session may not fill multiple independent
  slots; peer attacks are hidden until freeze; high-risk designs require
  distinct workers and operators before an acceptance synthesis.
- **L. ScientificDesignRisk:** LOW/MODERATE/HIGH/CONFIRMATORY risk determines
  required independent attack coverage. A material challenge still dominates
  regardless of apparent attacker consensus.

## M–P. Engineering and PRE_SCIENCE

- **M. EngineeringVerificationRisk:** deterministic, science-critical, and
  high-ambiguity risks are distinct from design risk.
- **N. Targeted fidelity:** `ImplementationFidelityCheck` binds exact
  implementation reference and science-to-code map; fidelity confirmation does
  not satisfy design correction.
- **O. Deterministic verification:** regression proposals preserve repeated
  mechanical findings as deterministic tests; generic review work is gated.
- **P. PRE_SCIENCE integration:** correction records grant no execution
  authority. Existing exact PRE_SCIENCE, gate, and deployment-readiness paths
  remain the only execution boundary.

## Q–Z. Interpretation, planner, policy, UI, and persistence

- **Q. ResultInterpretationAttack:** attributed, deduplicated interpretation
  attacks test nulls, causal specificity, intervention validity, scope, and
  measurement without mutating result evidence.
- **R. Scheduler changes:** correction work is separately classified; stale
  discovery work is dynamically retired before allocation, and stale rounds do
  not block Brain ranking.
- **S. WorkPlanner changes:** accepted design synthesis atomically materializes
  its bounded successor; V5 and interpretation preview APIs provide explicit
  authorized-materialization candidates rather than fabricating work.
- **T. Permission integration:** all correction mutations use existing active
  worker/session checks; no correction path bypasses lease or authority fences.
- **U. ResearchPolicy:** `prompt_compiler.py` includes correction runtime modes
  and policy context exposes the selected correction mode.
- **V. Prompt compiler:** design attack, implementation fidelity, result
  interpretation, and deterministic engineering prompt modes are explicit.
- **W. Worker context:** correction-layer identity is exposed through canonical
  worker context and independent batch membership.
- **X. Chucky UI:** Cockpit projections and dashboard sections separate design,
  fidelity, and interpretation correction.
- **Y. Events:** attack, challenge, synthesis, fidelity, interpretation,
  regression, classification, and stale-discovery transitions emit durable
  research events/outbox-compatible provenance.
- **Z. Migrations:** correction batch/membership/synthesis tables are created
  idempotently by `ScientificSelfCorrectionService._migrate`.

## AA–AF. Historical classification and V5

- **AA. Read-only historical classification:** the classifier produces a
  shadow classification; inferred labels are only persisted through an explicit
  provenance-preserving backfill operation.
- **AB. V5 audit:** `v5_missing_attack_work_preview` reports current V5
  assumption/challenge/coverage gaps without creating science or authority.
- **AC. A3/A5/A6:** legacy extraction can preserve inspected source material as
  partial conceptual attacks. It does not claim missing historical evidence.
- **AD. A4:** implementation/adversarial fidelity remains distinct and cannot
  complete design correction unless a real design attack is recorded.
- **AE. Missing V5 attacks:** missing attack work is visible as a bounded
  proposal preview and requires authorized planner materialization.
- **AF. V5 regressions:** tests prove GRU-256 + learned-G fidelity does not
  satisfy idea-level correction and that partial coverage remains incomplete.

## AG–AL. Regression, performance, and limitations

- **AG. CPDS/v3.6.5 regressions:** broad tests retain verdict-aware authority,
  permissioned parallelism, exact version fencing, and confirmation sealing.
- **AH. Generic-code-review suppression:** correction tests verify generic
  review roles are not the default engineering path.
- **AI. Targeted-fidelity tests:** fidelity and design acceptance are tested as
  non-interchangeable records.
- **AJ. Interpretation tests:** interpretation attacks are provenance-bound,
  deduplicated, and cannot admit evidence.
- **AK. Development/confirmation:** existing hard-sealed confirmation and
  development/confirmation separation tests remain green.
- **AL. Performance and limitations:** correction is bounded to design risk and
  science-critical ambiguity; deterministic engineering takes the fast path.
  No synthetic historical V5 evidence is generated, and no planner bypass is
  introduced. Historical A3/A5/A6 mapping and actual work materialization still
  require inspected evidence and authorized planner action respectively.
