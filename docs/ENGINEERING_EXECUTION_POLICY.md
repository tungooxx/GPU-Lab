# Engineering Execution Policy

Engineering work is provider-neutral and follows: **receive, inspect,
understand, reproduce, plan, edit, test, verify invariants, review diff,
record, hand back to science**.

- Inspect the repository, relevant callers/callees, configuration, and tests before editing.
- Reproduce a baseline or failure before modifying experimental behavior.
- Make the minimum change and test progressively: targeted, module, integration, then real smoke when needed.
- For scientific instrumentation, record the intended variable, held-fixed variables, machine-checkable guards, and prohibited changes in `EngineeringTask`.
- Verify intervention-off preserves the native path and intervention-on changes its intended target. A failed guard is an invalid implementation, never scientific falsification.
- Record commands, tests, diffs, artifacts, and invariant results in `EngineeringResult`. Its `scientific_result` is always `NOT_ASSESSED`.
- Inspect the diff and preserve unrelated user changes. Only Research OS assessment may update scientific belief.
