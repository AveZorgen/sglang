---
status: proposed
kind: design
topic: moe-offload-ascend-decode-graphs
repo: AveZorgen/sglang
branch: moe_offload
reviewed_head: 678513e978f9a2d5f37de3435ba314955662ae57
created: 2026-11-18
---

# Decode graphs for Paged Experts on Ascend

Prototype a **Triton-Ascend page-in kernel over mapped host memory** on A2/A3. This keeps routing, residency decisions, and weight movement in one `torch.npu.NPUGraph` replay. Use host-submitted ACL DMA as the comparison path and as a candidate if mapped reads are unreliable or slower on the target stack. Choose a backend per validated hardware and software configuration. This is the graph portion of Stage 3 in the [integration plan](moe-offload-stage1-two-owner-split.md).

## Repository and branch

Seed the `moe_offload` integration branch in `AveZorgen/sglang` from TamirBaydasov's `moe_offload` at `678513e978`. Base the graph feature branch on `origin/moe_offload` after the eager transfer work lands, and target the same integration branch for review. `origin` is AveZorgen's fork and `upstream` is `sgl-project/sglang`. [Starting commit](https://github.com/TamirBaydasov/sglang/commit/678513e978f9a2d5f37de3435ba314955662ae57) · [Integration fork](https://github.com/AveZorgen/sglang)

## Problem and scope

Each layer has `E` logical experts in host RAM and `K` physical slots in HBM. On each decode step the router can choose different experts, and the slots can contain different weights. The current Ascend executor reads a routing histogram to the CPU, plans waves in Python, and issues copies for specific expert/slot pairs. An ordinary graph replay cannot rerun that Python. Enabling graphs also selects the shared `DeviceExecutor`, whose planner and host-memory gather currently use CUDA kernels. [Eager and device executors](../python/sglang/srt/layers/moe/paged_experts/executor.py) · [CUDA implementation, SGLang #42557](https://github.com/sgl-project/sglang/pull/42557)

The first release covers one NPU, unquantized BF16 ND weights, and explicit `K`. Capture full decode only when the **padded route tensor** has at most `K` elements. For standard decode, this is `padded_batch_size × top_k <= K`. Larger batches and prefill run the existing eager waves, so they can use more than `K` distinct experts without allocating a full-layer HBM pool. The Ascend runner still sees `K` physical experts and combines the original router weights with the projected outputs. Native NZ layouts, quantization, auto sizing, parallelism, and speculative decode are outside this first release. [Ascend runner contract](../python/sglang/srt/hardware_backend/npu/moe/paged_experts.py) · [Decode capture cap](../python/sglang/srt/layers/moe/paged_experts/method.py#L425-L442)

## Primary path: Triton-Ascend and mapped host memory

```text
Router top-k → device residency planner → mapped-weight gather → Ascend MoE
                    │                            │                    ↑
                    └─ remapped IDs and weights ─┼────────────────────┘
                                                 └─ writes fixed HBM slots
```

1. **Bind the weight store at load time.** Use an owning, runtime-compatible host allocation and map each final ND weight range with `aclrtHostRegisterV2` and `aclrtHostGetDevicePointer`. Keep a CPU view for eager copies and pass the device-visible address to Triton through a device tensor. Allocate and register outside capture. Verify registration flags, alignment, pinned capacity, and address validity on each target stack. Keep the allocations and slot tables alive until all graphs using them are retired and in-flight work is drained. [CANN host registration](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2229.html) · [vLLM-Ascend Engram host buffer](https://github.com/vllm-project/vllm-ascend/blob/bd42eea01d8312c4df469fabaed45f77a12df4d3/vllm_ascend/models/deepseek_v41/engram/npu.py#L145-L210)
2. **Plan residency on the NPU.** Use the existing `DeviceResidency` representation. Start with one Triton program owning each layer's update to avoid races between insertions. Identify distinct requested experts, preserve every requested resident, then assign misses to the least recently used slots among the remaining residents. Produce `src[K]`, `dst[K]`, `count[1]`, and the remapped IDs and weights. Repeated routes share a slot. Padding gets a valid slot ID and zero weight. Maintain both expert-to-slot and slot-to-expert maps. [Device residency buffers](../python/sglang/srt/layers/moe/paged_experts/residency.py#L62-L108)
3. **Gather missing expert rows.** For each paged tensor, launch a fixed grid over at most `K` copies and the tiles of one expert row. Each program checks its copy index against `count` before reading the plan or touching host memory. For active entries, compute the source and destination offsets from the current expert and slot and copy the tile. Start with aligned, unmasked reads from mapped host memory. Handle a tail only with a verified access pattern that stays inside the host allocation and destination row. On a cache hit the gather issues no weight reads. This path uses NPU kernel loads and stores. Measure it separately from runtime DMA. [Engram Triton gather](https://github.com/vllm-project/vllm-ascend/blob/bd42eea01d8312c4df469fabaed45f77a12df4d3/vllm_ascend/ops/triton/engram_lookup.py#L67-L123) · [Reported masked-load fault on 910B2C](https://github.com/triton-lang/triton-ascend/issues/2723)
4. **Consume the slots on the graph stream.** Keep planning, gathers, and the existing runner ordered on the same stream. The runner reads the updated weights only after all gathers complete. Compile and warm up before capture. Keep the same backing tensors for slot weights and persistent residency across graph shapes. At a graph/eager transition, reuse `DeviceResidency.load_into` and `store_from` to synchronize the host planner with the NPU's actual slot contents. [Ascend capture and warmups](../python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py#L398-L446)

The first performance question is whether mapped reads sustain useful bandwidth at **expert-row sizes** on A2/A3. Engram establishes an access pattern, and the A2 masked-load report establishes a compatibility risk. Both the hardware and the Triton/CANN versions matter.

## Alternative: host callback and ACL DMA

Capture a device-to-host copy of the fixed-size route tensor into persistent pinned memory. After that copy, a stream-ordered callback reads the current routes and derives the same one-wave placement with the host LRU policy. A host transfer context submits the missing rows via `aclrtMemcpyAsync` on a separate stream, or via `aclrtMemcpyBatchAsync` where available. Wait for the transfers to complete before publishing the slot map. The captured graph then uploads the map, remaps IDs, and runs MoE. Keep the route buffer, map, host weights, and HBM slots alive through replay. Serialize updates to each layer's slots and map. [vLLM-Ascend callback and publication](https://github.com/vllm-project/vllm-ascend/blob/f687e5ba2d672a9f10eebc6b516ddcc160060f05/vllm_ascend/distributed/expert_transfer/expert_offload_manager.py#L197-L237) · [DMA submission and fence](https://github.com/vllm-project/vllm-ascend/blob/f687e5ba2d672a9f10eebc6b516ddcc160060f05/vllm_ascend/distributed/expert_transfer/expert_weight_transfer.py#L9-L38)

The ACL copy APIs take source and destination addresses from the host at submission. Capturing an ordinary H2D copy of an expert row would replay that row's addresses. Batching reduces host submissions and still requires a host copy plan. The documented `aclrtMemcpyAsyncWithDesc` supports intra-device copies only. A device planner followed by a general H2D DMA launch from device-produced addresses would require a different runtime primitive. [CANN batch copy](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1957.html) · [CANN descriptor copy](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1062.html)

vLLM-Ascend reports this callback approach on 950PR/A5 using `torch.npu._launch_host_func` and a side stream. Treat it as a reference for ordering. For A2/A3, verify that the callback is captured and called once per replay, that the side stream can make progress while the graph waits, and that errors reach the worker. CANN documents restrictions on task submission and synchronization from callbacks. Use a supported callback and transfer worker contract for the target runtime, and never block on work queued behind the same callback. A partial transfer failure must fail the step before GEMM can consume inconsistent weights. [vLLM-Ascend PR #18604](https://github.com/vllm-project/vllm-ascend/pull/18604) · [CANN callback restrictions](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1959.html)

## SGLang integration

| Area | Proposed change |
| --- | --- |
| `PagedExpertsMoEMethod._build` and `DeviceExecutor` | Let the runner contract supply the two device operations: bind the store's gather and invoke the planner. Keep the existing CUDA defaults. Build the Ascend graph path after the final slot tensors exist. Reuse shared slot remapping, eager waves, and runner output handling. |
| `AscendRunnerContract` | Replace the blanket graph rejection with validation for full decode, eager prefill, explicit `K`, and the supported ND and runtime configuration. Keep the eager configuration available if graph setup is unsupported. Import or build Triton only when the graph path is requested. |
| Capture and dispatch | Reuse the per-layer `K // top_k` cap. Check the padded capture shape and ensure replay selects an existing eligible graph. Route overflow to eager before replay. Guard against accidental eager page-in during NPU capture. |
| Lifecycle | Finish, retire, and drain graph work before releasing mappings or slot storage. If a runtime page-in fails after mutating state, fail the affected worker rather than falling through to eager with a stale slot map. |

Keep the current user-facing controls. With graph support implemented on a validated stack, the intended configuration is `SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1`, `--enable-paged-experts`, `--paged-experts-num-resident K`, and `--cuda-graph-config '{"decode":{"backend":"full"},"prefill":{"backend":"disabled"}}'`. `--disable-cuda-graph` still selects the eager baseline. [Method construction and dispatch](../python/sglang/srt/layers/moe/paged_experts/method.py#L284-L343) · [Graph configuration parser](../python/sglang/srt/model_executor/cuda_graph_config.py#L246-L288)

## Validation and decision gates

| Gate | What must be shown |
| --- | --- |
| Transfer microbenchmark on each target A2/A3 stack | Capture one graph that copies from a mapped allocation using device-provided indices. Change the source and slot between replays and check every byte against an ACL copy. Cover zero, one, and `K` misses, and use realistic `w13` and `w2` row sizes. Measure mapped gather time and ACL DMA time separately. For the callback candidate also measure the full route D2H → callback → H2D → map-ready latency. |
| Synthetic MoE | With `E >= 4`, `K = 2`, and `top_k = 2`, warm up with `[0, 1]`, then replay `[0, 1] → [1, 3] → [3, 2]` using the same captured graph. Expect zero, one, and one page-ins and output parity with the unpaged Ascend runner. Add duplicate IDs, padding, and a non-unit routed scaling factor. With `E >= 6` and `K = 4`, alternate between captured batch sizes 1 and 2 and interleave an eager step with 3 tokens and 6 distinct experts. Verify the eager step changes residency and the next replay observes that change. |
| Model smoke test and timing | Use Qwen3.5-35B-A3B with the Stage 1 BF16 ND baseline. Compare tokens and logprobs against eager paged execution at matching settings. Establish the numeric tolerance from repeated baseline runs before judging parity. Compare eager paged and graph paged decode at the same `K`, batch size, and route workload. Record p50/p95 decode step time, hit/miss counts, page-in time, pinned host bytes, and HBM peak. |

Land the graph change after the eager correctness and Stage 3 transfer baselines pass. Select mapped Triton gather if it passes replay correctness on the target stack and improves the measured decode critical path. Select callback DMA only if it passes the same tests and its end-to-end cost justifies it. Keep eager mode on target configurations where neither graph path passes. There is no A2/A3 hardware result for this design yet.
