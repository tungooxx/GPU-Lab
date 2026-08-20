# v3.2.2 Coordination Audit — Read Only

Audit time: 2026-08-20 (Asia/Saigon)

Scope: PostgreSQL project `a7bdabba-d07e-44ca-b572-cdef6d3210b2`
(`PlanCarry: LLM Plan-State Persistence`).  All queries were `SELECT` only;
this audit changed no scientific object, WorkItem, lease, or worker state.

## Existing

The deployed database is still the pre-v3.2.2 Lab schema: the
`scientific_gates` relation does not yet exist.  Consequently there are no
live, backfilled gate or authority assertions to report until the controlled
v3.2.2 deployment runs its additive migration.

Current work counts are:

| Status | Count |
| --- | ---: |
| COMPLETED | 89 |
| INVALIDATED | 2 |
| READY | 2 |
| RESULT_READY | 2 |
| WAITING_DEPENDENCY | 7 |

There were zero active leases and therefore zero expired active leases at the
time of audit.  A title-based scan found zero active tasks labeled fallback,
retry, or recovery.  This is only a legacy-data observation, not proof that
no historical fallback semantics ever existed.

## Dependency findings

None of the sampled `WAITING_DEPENDENCY` items was already satisfied:

- The frozen binding cohort waits for a hypothesis at `INCONCLUSIVE` while it
  requires `SUPPORTED`.
- Two untouched/cross-environment replication tasks wait for a hypothesis at
  `ACTIVE` while they require `SUPPORTED`.
- The v2.5 discovery task waits for an engineering repair currently `READY`
  and an adversarial re-audit currently `WAITING_DEPENDENCY`; both require
  `COMPLETED`.
- The re-audit itself waits for that same engineering repair to complete.

The legacy preregistration review has a referenced object that is absent from
the current `research_objects` table, so it remains correctly non-actionable
until repaired or deliberately invalidated.  This report does not repair it.

## Reusable / partial

- WorkItem dependency records, leases, worker sessions, and event records are
  durable PostgreSQL primitives that v3.2.2 reuses.
- Existing `WAITING_DEPENDENCY` state already preserves the intended scientific
  ordering, but it lacks typed `ScientificGate` ownership and preflight
  evidence.
- Historical invalidated work is preserved, which is compatible with the new
  supersession model.

## Missing before deployment

- Database-enforced gate authority identity.
- Immutable deterministic preflight records.
- Gate-driven dependency resolution and wake deduplication.
- Authority, gate, subject-version, recovery-policy, and stale-context fields
  on the live schema.

## Root cause and migration posture

The observed waiting tasks are principally scientific prerequisites that have
not yet reached their required statuses, not evidence of a worker queue that
should be force-unlocked.  The v3.2.2 migration is additive and intentionally
does not infer gate authority or independent-review history from these legacy
rows.  Any future backfill must be explicitly marked `BACKFILL` and must not
be used as prospective coordination telemetry.
