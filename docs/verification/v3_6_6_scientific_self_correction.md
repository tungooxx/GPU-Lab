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

The initial shadow classifier is read-only and labels records as design attack,
implementation fidelity, result interpretation, deterministic engineering, or
unknown. Existing history is not rewritten.

Verification on 2026-08-31:

- Windows suite: `396 passed, 97 skipped`.
- Fresh PostgreSQL v3.6.6 integration: `4 passed`.
- Covered invariants: peer isolation, fidelity != design correction, one
  material challenge overrides apparent majority, independent duplicate
  challenge origins collapse only during synthesis, discriminator creation,
  and interpretation attack not admitting evidence.

Remaining rollout work is intentionally feature-flagged: planner policy and UI
can consume these typed records after production shadow-audit review. Existing
v3.6.5 permissioned authority and hard-sealed confirmation paths remain the
source of truth.
