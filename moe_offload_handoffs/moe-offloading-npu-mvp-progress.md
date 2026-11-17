---
status: open
kind: progress
topic: moe-expert-ram-offload
repo: sglang
branch: moe_offload
created: 2026-11-09
---

# MoE offloading MVP: contract layer implemented

Implements steps 1-5 of `moe-offloading-npu-mvp-plan.md` (local part) as `b51aacedb9` on
`moe_offload` (from reviewed head `0afb736d0a`). Not pushed, no PR.

## What landed

- Shared `RunnerContract` hooks (CUDA defaults byte-identical): `runner_config`
  (None = shared paged config), `host_alloc` (None = memfd + cudaHostRegister default),
  `validate`. Registry `register_runner_contract` in paged_experts/runners.py; consulted
  by `contract_for` after the triton/Marlin checks.
- `hardware_backend/npu/moe/paged_experts.py`: `AscendRunnerContract` — runner config
  `num_experts=num_local_experts=K`, combine on; waves add `[tokens, hidden]` partials
  (outside-wave routes run as slot 0, weight 0); `finish` applies routed_scaling_factor
  once; host alloc via `alloc_with_pin_memory` (ALLOC_MEMORY_FUNCS["npu"]); `validate`
  rejects graphs and unset `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT`. Registered from the
  bottom of `layers/moe/moe_runner/ascend.py` (import side effect).
- `method.py`: provisional runner -> contract -> rebuild only when the contract config
  differs; `build_store(step, alloc=self.contract.host_alloc)`.
- No NPU format needed: BF16 NPU layers use shared `UnquantizedFusedMoEMethod`, which
  `UnquantizedFormat` already claims.
- CPU tests in test/registered/unit/layers/moe/paged_experts/test_paged_experts.py:
  combined-output forward (hits/misses/eviction/padding/1- and multi-wave/scaling 2.5),
  mutants (wave replacement, missing and double scaling) proven red on the forward test,
  startup validation accept/reject, `contract_for` ascend registration case.
  Suite: 41 passed + 63 subtests (9 TestLoading/TestLoads failures are pre-existing
  Windows mmap gaps, identical on the base commit).

## Next (in order)

1. NPU synthetic gate on the box (plan section "NPU synthetic gate"): BF16 E=16, K=4,
   top-k=2, paged vs unpaged Ascend runner, same ND weights. Box work needs per-command
   approval; check the box is free first.
2. Full-model gate: Qwen3.5-35B-A3B, one NPU, eager, fixed K<E, compare logprobs vs
   same-config unpaged; measure the noise floor before setting any tolerance.
3. Record K/E, hit/miss (from `Wave.loads`), NPU memory, pinned host bytes, page-in time
   (temporary sync instrumentation).

## Not yet decided / traps

- `routed_scaling_factor`: contract applies `moe_runner_config.routed_scaling_factor
  or 1.0` once, mirroring the triton path. Qwen3.5 likely carries None (fused in topk).
  If the full-model gate mismatches, check the model's `apply_routed_scaling_factor_on_output`.
- Leftover `stash@{0}` in sglang is a duplicate snapshot of this work (stash-pop
  collision with the .claude/skills junction); drop it manually if unwanted.
