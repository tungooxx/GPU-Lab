# Historical epistemic reclassification

Run date: 2026-08-12

The deterministic `epistemic_reclassification()` process was run twice against
the current PostgreSQL Research OS. It only adds classification metadata and
`EPISTEMIC_CLASSIFICATION_BACKFILLED` events; it does not alter observations,
predictions, hypotheses, artifacts, or historical timestamps.

## First run

| Measure | Count |
| --- | ---: |
| Total decisions | 174 |
| Scientific actions | 35 |
| Result inspections | 59 |
| Legacy backfills | 80 |
| Administrative recovery | 0 |
| System verification | 0 |
| Provider contract tests | 0 |
| Strategy-learning eligible | 0 |
| Closed cycles | 4 |
| Open/incomplete cycles | 170 |
| Records classified | 174 |

## Idempotency check

The second run changed **0** records and emitted no additional classification
events. This proves the migration is idempotent for the current database.

## Interpretation

The trace’s historical dominance of inspection/backfill records is now visible
as metadata rather than being silently treated as strategy evidence. The zero
eligible strategy cycles is conservative: historical records without proof of
prospective production-science selection, complete hindsight, and structured
information-gain basis fail closed. This is a data-quality finding, not a loss
of scientific history.
