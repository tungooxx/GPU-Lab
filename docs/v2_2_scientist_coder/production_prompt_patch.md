# Production Prompt Patch

When a selected scientific action requires code changes, create or reuse an
`EngineeringTask` before executing the canonical experiment. Record the intended
variable, held-fixed variables, prohibited changes, and implementation guards.
Run baseline, intervention-off, intervention-on, and held-fixed checks as
applicable. Record an `EngineeringResult`; engineering smoke and passing tests
are not scientific evidence. Only submit and assess the scientific experiment
after implementation verification succeeds.
