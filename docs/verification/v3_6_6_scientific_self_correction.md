# v3.6.6 scientific self-correction audit

This release adds an additive correction triplet without changing existing
evidence admission, execution authority, or confirmation controls.

| Layer | Durable record | It answers | It cannot do |
| --- | --- | --- | --- |
| Design | `ScientificDesignAttack` / `DesignAttackSynthesis` | Is this idea coherent and discriminating enough to test? | certify code or admit evidence |
| Implementation | `ImplementationFidelityCheck` | Do exact bytes realize the accepted design? | count as scientific design correction |
| Interpretation | `ResultInterpretationAttack` | Does a result support the stated claim? | mutate the result or admit evidence |

The model provides first-class assumptions, challenges, competing and
degenerate explanations, required discriminators, peer-isolated attack batches,
and synthesis without majority voting. A material unresolved challenge blocks a
`RETAIN_FOR_TEST` synthesis regardless of how many other attackers found no
issue.

The read-only classifier labels records as design attack, implementation
fidelity, result interpretation, deterministic engineering, or unknown. An
explicit idempotent backfill can persist an inferred label with confidence and
provenance without rewriting the historical artifact.

Verification on 2026-08-31:

- Windows suite: `399 passed, 107 skipped`.
- Fresh PostgreSQL v3.6.6 integration: targeted correction checks passed,
  including current-lineage and legacy-backfill regressions.
- Covered invariants: peer isolation, fidelity != design correction, one
  material challenge overrides apparent majority, independent duplicate
  challenge origins collapse only during synthesis, discriminator creation,
  interpretation attack not admitting evidence, and the V5 GRU-256 + learned
  G assumption map remaining incomplete after an implementation-fidelity pass,
  bounded V5 missing-attack preview, persistent inferred classification,
  atomic design finalization/repair successor materialization, and
  interpretation-attack deduplication/provenance, high-risk independent-worker
  and operator coverage, partial assumption coverage, current-lineage freezing
  with technical-failure separation, and provenance-preserving legacy design
  attack extraction.

All v3.6.6 rollout flags default enabled for this release and are enforced at
their public mutation boundaries; operators can selectively disable a feature
without disabling unrelated correction layers. Existing v3.6.5 permissioned
authority and hard-sealed confirmation paths remain the source of truth.
