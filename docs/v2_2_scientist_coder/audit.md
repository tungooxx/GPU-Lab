# v2.2 Audit

## Current engineering workflow

GPU-Lab already provides a PostgreSQL research object/event store, local/remote
execution controls, artifact collection, result assessment, and v2.1 epistemic
classification. It did not have a typed record for code implementation work.

## Components to preserve

`ResearchDecision`, experiment/run reservation, immutable events, result
assessment, `LocalRunner`, workspace boundaries, and v2.1 strategy eligibility
remain authoritative. Engineering records are additive and cannot update
hypotheses, causal edges, or strategy memory.

## Gap addressed

`EngineeringTask` captures a frozen implementation request and scientific versus
engineering invariants. `EngineeringResult` captures implementation evidence
with `scientific_result=NOT_ASSESSED`, preserving the separation between code
validity and scientific evidence.
