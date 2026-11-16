---
status: open
kind: branch-split
topic: moe-offload-five-stage-two-owner-plan
repo: sglang
branch: moe_offload
reviewed_head: cd4d904842c60be213e57e9f1189695121e2548d
created: 2026-11-16
---

# MoE offload: staged work in a fork integration branch

## Workflow decision

Use the fork's `moe_offload` as the integration branch, updated only by reviewed PRs.
Base independent feature branches on the same integration commit, then update them after
each ordered merge. Do not make every parallel branch depend on another unmerged branch.

## Dependency findings

- Stage 1 is the foundation and merges first. No later stage is fully independent of its
  runtime and acceptance result.
- Stage 2 format/loading work can be implemented alongside Stage 1 after joint interface
  prep. Its standalone tests use the ordinary unpaged NPU path; paged validation and merge
  wait for Stage 1. Choose one format first, not the whole format matrix.
- Stage 3 asynchronous transfer can be implemented alongside Stage 2 once Stage 1's
  interface is stable. It treats host rows as opaque final-layout bytes. Its NPU validation
  needs the correct eager runtime. Graph execution is not independent: it follows async
  transfer measurements and readiness semantics, in a separate PR.
- Stage 4 TP can be designed and implemented on BF16 independently of Stage 2 and async
  transfer. Final distributed validation must rerun against the integrated transfer path.
  Fixed EP needs a separate ownership/readiness design; EPLB is not an independent add-on.
- Stage 5 policy work is not independent: prefetch decisions need representative async and
  serving traces. Trace collection and XPU lookahead investigation can start early. Exposed
  miss/copy timing is already needed for Stage 3 acceptance, so instrumentation is not a
  Stage-5-only deliverable.

## Two persistent workstreams

### Owner A: execution and transfers

Stage 1 BF16 runtime -> Stage 3 async eager transfer -> Stage 3 graph execution. Own wave
execution, slot/runtime behavior, transfer readiness, and related correctness tests.

### Owner B: formats and parallelism

Stage 2 one native format -> Stage 4 TP on BF16 -> fixed-EP evaluation if approved. Own
format conversion/loading, per-format cache/sizing, distributed ownership, and their tests.
Standalone format and BF16 TP work must not require Owner A's unmerged async branch.

Trace infrastructure can be a small early separate PR by either owner with no overlapping
files. Stage 5 policy implementation starts only after async and serving traces exist.

## Ordered integration PRs

0. Joint narrow seams if needed: format registration, per-format repack/validation.
1. Stage 1 BF16 single-NPU correctness.
2. Stage 2 one native format; subsequent formats can be separate PRs.
3. Stage 3 asynchronous eager transfer with exposed-miss/copy measurements.
4. Stage 3 graph execution, designed from async measurements.
5. Stage 4 TP; fixed EP separately after its design review.
6. Stage 5 trace-driven residency/prefetch changes.

Development may overlap; validation and merging follow this order unless the RFC is
explicitly changed. Every merge reruns earlier acceptance tests on the combined code.

## Not doing

- Do not assign Stage 3 graphs or Stage 5 prefetch as a first-week independent branch.
- Do not validate every format x transfer x parallelism combination at each stage; keep
  fixed-K BF16 as the common integration baseline and add one new dimension per PR.
