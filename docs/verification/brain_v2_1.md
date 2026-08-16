# Research Brain v2.1 epistemic-contract verification

Verification date: 2026-08-12

This report distinguishes implementation and integration evidence from
scientific evidence. A passing software test is not a scientific result.

## Implemented and verified

- Decision roles, scientific roles, execution verification, scientific
  verification, cycle status, learning namespace, and strategy-learning role
  are additive metadata on persisted decisions.
- Strategy eligibility fails closed for legacy backfills, system smokes,
  contract tests, missing outcomes, incomplete cycles, unknown information
  gain, missing hindsight, and missing information-gain basis.
- Historical reclassification is deterministic and idempotent. Live DB run:
  174 decisions, 0 changes on repeat, 4 closed cycles, 170 open/incomplete
  cycles, and 0 eligible strategy-learning cycles.
- Brain decisions persist central uncertainty, runner-up comparison,
  prospective hindsight, candidate comparison, and hard-block findings.
- Scientific discrimination critics report nulls, controls, confounds,
  architecture prematurity, and can hard-block missing predictions.
- Meta-review separates scientific, administrative, and system-verification
  activity and reports closed-cycle coverage.
- Read-only `decision_epistemic_audit` and eligibility/reclassification MCP
  tools are exposed by the live service.

## Verification evidence

- Ruff: passed.
- Focused epistemic/strategy/MCP tests: 15 passed.
- Existing baseline before this patch: 179 passed, 18 skipped.
- Live MCP smoke: `VERIFIED_INTEGRATION`.
- Live PostgreSQL reclassification: idempotent on two consecutive runs.
- Container rebuild/restart: GPU-Lab healthy and MCP tool discovery returned
  127 tools, including `decision_epistemic_audit`.

## Implemented but not verified as real science

- No new causal GPU experiment is claimed by this patch.
- Real scientific result inspection, scoped hypothesis transition, and
  world-model promotion require a valid preregistered experiment and inspected
  artifacts; software smoke tests do not satisfy that gate.
- Broader historical benchmark coverage remains incomplete; only the existing
  frozen benchmark episodes are evidenced.
- Model-backed operator/PaperQA quality evaluation and real Paper2Agent
  conversion remain separate integration work.

## Remaining risks

- `historical_reclassification_report` performs an idempotent classification
  backfill despite its report-like name; callers should treat it as a write
  operation.
- The live database currently has no strategy-eligible closed cycles, so
  production strategy transfer has not been scientifically demonstrated.
- GPU-hour accounting and information-per-GPU-hour remain unknown when the
  runtime does not provide actual accounting.

## Next highest-value step

Run one genuinely discriminating, preregistered causal experiment through the
canonical GPU-Lab execution and inspection path, then verify that its scoped
evidence changes the next `brain_step` recommendation.
