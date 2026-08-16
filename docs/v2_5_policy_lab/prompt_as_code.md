# Prompt-as-Code Summary

## Canonical Research Policy

`ResearchPolicy` is canonical. Runtime instructions are deterministic compiled
artifacts and never a writable source of scientific policy.

## Prompt Compiler and Generated Artifacts

`PromptCompiler` compiles Generic, ChatGPT, Claude, Codex, OpenAI API, and
Claude API forms. Each artifact records policy version, provider, semantic
sections, provenance, timestamp, hash, and an approximate token count. The
rendered content starts with `GENERATED FILE — DO NOT EDIT MANUALLY`.

## Core Invariant Protection

Nine core epistemic invariants are embedded in every canonical policy. Ordinary
patches cannot remove or replace them; such an attempt is rejected with
`CORE_POLICY_INVARIANT_VIOLATION`. A `CORE_POLICY_CHANGE` requires a separate,
stronger review path.

## Improve Prompt Mode

`improve_start(..., prompt=True)` creates normal, evidence-gated policy
hypotheses and prompt-presentation patches. It compiles baseline and candidate
forms before the existing blinded benchmark, stores hashes/token deltas, and
leaves production unchanged until explicit promotion.

## Runtime Context

`research_policy_context_get(project_id, provider="CHATGPT")` returns the
active compiled context. It intentionally excludes dynamic research state;
callers retrieve that through normal Research OS tools.

## Remaining Risks

Benchmark policy selection remains deterministic rather than live-model prompt
execution. Token counts are stable whitespace estimates, not provider tokenizer
measurements. Prompt compression and skill extraction remain future candidate
patch mechanisms rather than autonomous file generation.
