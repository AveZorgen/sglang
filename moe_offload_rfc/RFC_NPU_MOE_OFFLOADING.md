# [RFC] [Ascend] MoE expert offloading through Paged Experts

- **Status:** Proposed
- **Roadmap:** Ascend MoE 2026
- **Scope:** SGLang runtime on Ascend NPUs

## Summary

We propose adding MoE expert offloading on Ascend NPUs by extending SGLang's existing **Paged Experts** implementation, not by writing a separate Ascend offloader.

Paged Experts keeps all `E` routed experts in pinned host memory and only `K` expert slots per MoE layer on the device. It copies routed-but-missing experts into those slots and rewrites logical expert IDs to physical slot IDs. Routing and expert computation still run on the device.

The first delivery will be a correctness-focused, single-NPU, eager implementation for unquantized experts. Later stages will add Ascend-native formats, asynchronous transfers, graphs, tensor parallelism, and trace-driven prefetch. Each stage will be independently validated and leave CUDA behavior unchanged.

We plan to support the implementation, NPU validation, documentation, and benchmark publication for this roadmap item.

## Motivation and goals

Routed experts account for much of the weight memory of large MoE models, but each token selects only a small subset. Keeping every expert on device can prevent a model from fitting or leave too little memory for the KV cache.

Generic CPU offload transfers weights at module or layer granularity. Paged Experts transfers only selected, non-resident experts. On Ascend, this should:

- serve models whose full expert set does not fit in device memory;
- trade device expert capacity for a larger KV cache;
- keep routing and expert computation on the NPU;
- reuse the existing loader, host store, residency policy, waves, and sizing logic.

The initial goal is capacity and correctness. Throughput gains are later goals and require Ascend measurements. All functionality stays behind the existing opt-in interface, and unsupported configurations fail at startup.

## Existing foundation

[Paged Experts](https://github.com/sgl-project/sglang/pull/42557) already provides the core mechanism on CUDA: a pinned host store, `K` resident slots, LRU demand paging, multi-wave execution, slot remapping, automatic sizing, and an optional disk cache for faster startup.

The `moe_offload` prototype adds an Ascend runner contract, NPU pinned-host allocation, and Ascend's output-combination rules. CPU tests cover the shared contract and multi-wave behavior. Synthetic-layer and full-model NPU validation are still pending, so this is not yet a validated Ascend feature.

The existing disk cache accelerates model loading. It is not a runtime SSD-offload tier and does not provide expert prefetch.

## Scope and design principles

1. Preserve the router, top-k selection, routing weights, and expert computation.
2. Reuse Paged Experts rather than duplicate its residency and execution machinery.
3. Keep Ascend-specific code under the NPU backend; shared hooks retain current CUDA defaults.
4. Reject unsupported configurations instead of silently falling back.
5. Add support in small stages with correctness and memory evidence at each stage.

The roadmap excludes CPU expert compute, CUDA UVM-style host read-through, runtime SSD offload, approximate routing, adaptive top-k, surrogate experts, new grouped-GEMM kernels, and speculative decoding. Approximate execution needs a separate proposal and explicit opt-in.

## Proposed architecture

Each MoE layer holds all routed experts in pinned host memory and `K` physical slots on the NPU. Host and device representations must use the final weight and scale layout expected by the Ascend runner.

Routing remains unchanged over logical `E` experts. Paged Experts loads missing selections into slots and remaps expert IDs to slot IDs. The existing Ascend grouped MoE computation then runs on those slots. Steps needing more than `K` experts use waves; their outputs are combined so every selected expert contributes once and routed scaling is applied once.

Shared code owns loading, storage, residency, wave planning, and execution. A registered runner contract supplies only backend-specific behavior:

- runner configuration for physical slots;
- pinned host allocation;
- startup validation;
- wave execution and output combination.

The first version uses host-planned, synchronous page-in on the current stream in eager mode. Later versions may use a transfer stream and events for overlap. Graph support requires stable slot storage and explicit transfer dependencies; its design follows eager and asynchronous validation on Ascend.

## Delivery plan

### Stage 1: Single-NPU correctness MVP

**Scope:** One NPU; TP=EP=PP=DP=1; BF16 ND experts; eager execution; explicit `K < E`; exact routing and synchronous LRU paging. Reject graphs, unsupported layouts, parallelism, EPLB, and speculative decoding.

**Required work:** Finalize the shared and Ascend runner contracts, reuse the NPU pinned allocator, validate physical-slot configuration and output combination, and add only tests not already covered by Paged Experts.

**Acceptance:** Existing CPU and CUDA tests show no regression. A synthetic Ascend layer matches unpaged execution across hits, misses, eviction, padding, and multiple waves. An Ascend-supported MoE model passes an end-to-end smoke test. Measured device expert storage follows the `K/E` arithmetic while host storage holds all `E` experts.

### Stage 2: Ascend-native formats and loading

**Scope:** Add the quantization formats and physical layouts required by target models, including packed weights and scales, without per-step conversion.

**Required work:** Define how each format pages weights, scales, padding, and metadata as one expert object. Ensure host rows contain the final device representation. Validate automatic `K` sizing and the startup disk cache for each format.

**Acceptance:** Per-format synthetic tests and target-model checks match unpaged execution. Unsupported format and layout combinations are rejected before serving.

### Stage 3: Asynchronous transfer and graph execution

**Scope:** Reduce exposed demand-miss time without changing routing, then support the appropriate Ascend graph mode.

**Required work:** Add side-stream copies and event-based readiness. Measure copy/compute overlap and queue contention. Choose host-planned or device-planned graph execution from those results, while retaining a correct eager path.

**Acceptance:** Eager and graph execution pass the same correctness tests. No slot is consumed before its transfer completes. Benchmarks report exposed miss time, copy time, and achieved overlap, not only hit rate.

### Stage 4: Parallelism and serving coverage

**Scope:** Add tensor parallelism first, then evaluate fixed expert parallelism and broader serving configurations at several concurrency levels.

**Required work:** Page each TP shard with deterministic slot plans and no added routing-critical-path coordination. Define EP ownership and readiness before enabling EP. Treat EPLB placement changes separately because they invalidate slot mappings.

**Acceptance:** Distributed tests show no deadlock, misrouting, or slot divergence. Results include TTFT, TPOT, throughput, tail latency, and memory use. Unsupported combinations fail explicitly.

### Stage 5: Trace-driven prefetch and residency improvements

**Scope:** Optimize only after traces show that demand misses remain significant, reusing device-independent Paged Experts ideas where suitable.

**Required work:** Collect routing, hit/miss, transfer, and exposed-stall traces. Evaluate lookahead prefetch and residency policies against the same traces. Actual routing remains authoritative; wrong predictions may waste bandwidth but never change outputs.

**Acceptance:** Compare against synchronous paging under the same memory budget. Report useful, late, and dropped prefetch bytes plus end-to-end metrics. Reject policies that improve hit rate without improving serving metrics.

## Validation and reporting

Every stage reports measurements separately from arithmetic estimates. Correctness comparisons use the same checkpoint, attention backend, MoE backend, and execution configuration, with paging as the only intended difference. Establish the same-configuration noise floor before choosing a numerical tolerance.

Benchmarks include allocated and peak device memory, actual pinned host bytes, `K/E`, hit/miss counts, transferred bytes, demand page-in time, exposed miss time, TTFT, TPOT, throughput, and tail latency where applicable.

Baselines include the unpaged model when it fits and generic CPU offload when it supports the checkpoint. Cross-hardware results are not Ascend performance claims.

The PR sequence follows the stages above, with asynchronous transfer and graph support in separate PRs. Each PR must remain correct and useful on its own. A separate Ascend offloader is rejected because it would duplicate Paged Experts; CPU compute and approximate execution remain separate efforts.

## Open questions

1. Is the registered runner-contract interface the right long-term boundary for non-CUDA Paged Experts backends?
2. After eager validation, should Ascend graphs prioritize host-planned transfers or device-planned residency?
3. Which prefetch mechanisms from the XPU Paged Experts proposal can be shared across backends?

## Related work

- [SGLang Paged Experts](https://github.com/sgl-project/sglang/pull/42557) and [documentation](https://docs.sglang.io/docs/advanced_features/paged_experts)
- [XPU Paged Experts and Lookahead Expert Scheduling RFC](https://github.com/sgl-project/sglang/issues/46520)
- [vLLM-Ascend expert-offload proposal](https://github.com/vllm-project/vllm-ascend/pull/18604)
