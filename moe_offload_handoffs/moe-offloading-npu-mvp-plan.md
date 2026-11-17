---
status: open
kind: investigation
topic: moe-expert-ram-offload
repo: sglang
branch: upstream/main
reviewed_head: 0afb736d0aab96e3ab80b4c54a376213a1db67a6
created: 2026-11-06
updated: 2026-11-09
---

# MoE offloading on Ascend: smallest testable MVP

## Decision

- **Adopt Paged Experts; do not write another offloader.** Start a new branch from the reviewed `upstream/main`, not the DCP testing branch.
- MVP: exact routing, BF16 experts, one NPU, TP=EP=PP=DP=1, eager execution, explicit K, short prompts, concurrency 1. Use the existing Qwen3.5-35B-A3B checkpoint for the full-model smoke test after a tiny synthetic layer passes.
- Use `--enable-paged-experts --paged-experts-num-resident K --disable-cuda-graph` and set the existing `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`. ND BF16 weights avoid host-side NZ packing in this MVP.
- Keep the router, top-k, weights, and NPU grouped GEMMs unchanged. Synchronous, host-planned page-in is the correctness baseline; speed is measured, not an MVP acceptance gate.

## Verified state at the reviewed head

- [`EagerExecutor`](sglang/python/sglang/srt/layers/moe/paged_experts/executor.py#L25-L102) already does one batched routing-count readback, host LRU planning, waves when more than K experts are routed, slot-ID remapping, and `copy_` page-in. It needs no scheduler integration. Its one-wave path skips `run_wave` but still calls `contract.finish` ([executor.py:82-101](sglang/python/sglang/srt/layers/moe/paged_experts/executor.py#L82-L101)).
- [`HostExpertStore`](sglang/python/sglang/srt/layers/moe/paged_experts/store.py#L23-L129) is the CUDA-specific part: it uses `memfd` plus `cudaHostRegister`; graph mode uses a CUDA device-pointer gather. MVP needs only pinned allocation and `page_in`, not `bind_device_gather` or `gather`.
- [`contract_for`](sglang/python/sglang/srt/layers/moe/paged_experts/runners.py#L168-L183) accepts only Triton and Marlin. Their contract returns uncombined `[token, top-k, hidden]` entries; the Ascend runner returns its normal weighted top-k sum as `[token, hidden]`.
- [`pre_permute_standard_to_ascend`](sglang/python/sglang/srt/layers/moe/moe_runner/ascend.py#L347-L399) passes `runner_config.num_experts` to the routing op. Once top-k IDs mean physical slots, both `num_experts` and `num_local_experts` must be K. The executor's earlier routing histogram still uses logical E.
- [`NPUUnquantMoEMethod`](sglang/python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py#L961-L998) calls `npu_format_cast` after loading. The ND override returns the original tensor ([utils.py:285-289](sglang/python/sglang/srt/hardware_backend/npu/utils.py#L285-L289)), so the existing unquantized post-load checks remain usable.
- Existing guards require single-device execution and reject EPLB and speculative decoding ([method.py:407-471](sglang/python/sglang/srt/layers/moe/paged_experts/method.py#L407-L471)). Explicit K avoids auto-K materialization in the first test.
- Reuse NPU pinned allocation through [`alloc_with_pin_memory`](sglang/python/sglang/srt/mem_cache/pool_host/common.py#L457-L469), already selected for NPU by [`ALLOC_MEMORY_FUNCS`](sglang/python/sglang/srt/mem_cache/pool_host/common.py#L557-L564).

## Implementation

1. **Extend the existing runner contract, with CUDA defaults unchanged.** Add optional hooks for runner configuration, host allocation, and startup validation. Keep the current Triton/Marlin configuration and CUDA registered allocation as defaults.
2. **Add an NPU-owned `AscendRunnerContract`** under `hardware_backend/npu/moe/`, registered when the existing Ascend runner is imported. It supplies the existing pinned allocator and rejects graphs and NZ mode before checkpoint loading.
3. **Let the contract configure the Ascend runner.** Select it from the provisional runner, then rebuild only if its configuration changes. Set `num_experts=num_local_experts=K` and keep normal combine enabled; later auto-K materialization reuses the selected contract.
4. **Merge complete Ascend wave outputs.** `run_wave` calls `run_on_slots`; routes outside the wave use slot 0 with weight 0. Sum `[token, hidden]` partials across waves. `finish` applies `routed_scaling_factor` once to the combined output for both the one-wave and multi-wave paths. Do not implement Triton-style `no_combine` for Ascend.
5. **Thread the host allocator through `build_store` into `HostExpertStore`.** Reuse `ExpertLoads`, `PostLoad`, LRU `plan_waves`, and current-stream `copy_` unchanged. Persistent pinned sources plus same-stream ordering make page-in complete before the grouped GEMM; add no per-copy synchronize. No NPU conditionals in shared model/layer code and no new kernel.

## Validation and acceptance

- Local: run the existing [`test_paged_experts.py`](sglang/test/registered/unit/layers/moe/paged_experts/test_paged_experts.py) suite. Add one CPU forward case for a combined-output runner covering hits, misses, eviction, padding, one- and multi-wave steps, and non-unit routed scaling. Prove it fails with wave replacement instead of addition and with missing or double scaling. Add no duplicate LRU or loader tests.
- NPU synthetic gate: BF16 E=16, K=4, top-k=2; compare paged and unpaged Ascend runners using the same ND weights. Cover one-token decode, enough tokens to force multiple waves, padded routes, repeated steps that evict experts, and non-unit routed scaling.
- Full-model gate: Qwen3.5-35B-A3B on one NPU, eager, fixed K<E, one short request. Compare output logprobs against the same TP1 model using generic CPU offload if that path supports the checkpoint; use the same attention and MoE backends. Do not invent a numerical tolerance: first measure the same-config noise floor.
- Record allocated/peak NPU memory, actual pinned host bytes, K/E, hit/miss counts, and page-in time. Derive hits/misses from `Wave.loads`; use temporary synchronizing instrumentation for copy timing, not a permanent hot-path sync. The memory claim is arithmetic until measured: device expert storage should scale as K/E, while the host holds all E experts. Check allocator rounding and available host RAM before loading.
- MVP is done when all three correctness gates pass and measured device expert storage matches the K/E arithmetic. Record whole-process memory separately. No throughput or latency claim is required.

## Ascend reference and later work

- Updated vllm-ascend main `743e1f1cf272575a0ccd74b1f89221bc5045e1a6` has no merged expert offload. PR #18604 remains open at `f4fdabee161b53868ae9deb00d5225ba5785f2c7` (checked 2026-11-09); the update from the previously reviewed head is a rebase plus naming cleanup, not a new transfer design.
- Its eager path batches route D2H, uses ACL H2D copies on the current stream, then synchronizes ([transfer.py](https://github.com/vllm-project/vllm-ascend/blob/f4fdabee161b53868ae9deb00d5225ba5785f2c7/vllm_ascend/expert_offload/transfer.py#L9-L51)). Its graph path uses `torch.npu._launch_host_func` and a side stream ([expert_offload_manager.py:223-238](https://github.com/vllm-project/vllm-ascend/blob/f4fdabee161b53868ae9deb00d5225ba5785f2c7/vllm_ascend/expert_offload/expert_offload_manager.py#L223-L238)).
- NZ/quant support is separate: the PR copies physical native storage, including scale strides and padding, and chunks pinned allocations to reduce power-of-two rounding ([expert_offload_manager.py:74-132](https://github.com/vllm-project/vllm-ascend/blob/f4fdabee161b53868ae9deb00d5225ba5785f2c7/vllm_ascend/expert_offload/expert_offload_manager.py#L74-L132)).
- Do not port its full-expert prefill/overflow pool: SGLang's existing waves handle more than K routed experts using only K slots. After the MVP, measure async side-stream copies and events; port the host-callback graph mechanism if needed; then TP shards; then GLM W4A8/NZ weights and scales. Add prediction only after traces show exposed misses worth hiding.

## Not doing

No full-expert overflow pool, EP/EPLB, PP/DP, speculative decoding, CPU expert compute, SSD tier, adaptive top-k, low-rank surrogates, new cache policy, or predictive prefetch in the MVP. No performance claim until the exact eager path passes correctness and memory validation.
