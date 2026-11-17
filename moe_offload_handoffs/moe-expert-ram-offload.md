---
status: open
kind: investigation
topic: moe-expert-ram-offload
repo: sglang
branch: dcp_ascend_testing
reviewed_head: d783606ec3c7b6bbd428691d0abcc3bae9859b27
created: 2026-11-02
---

# MoE expert weights offload to RAM: upstream state and NPU path

Question investigated: does expert→RAM offloading exist in sglang, and can it be
adapted for sglang-on-NPU. Searched 2026-11-02 via `gh search` + local tree.

## Findings

### 1. note — merged core mechanism: Paged Experts (PR sglang#42557, merged 2026-10-19)

The feature as described exists and is merged. Lives in
`python/sglang/srt/layers/moe/paged_experts/` (store.py, executor.py, method.py,
runners.py, formats.py). Design: K resident expert slots per layer on device, all E
experts in pinned host RAM (`cudaHostRegister`), copy in only routed-and-missing
experts per step, device-side `topk_ids → slot` rewrite so the fused MoE kernel is
unchanged and the planning step is CUDA-graph capturable. Measured 12× (int4) /
30× (FP8) over `--cpu-offload-gb` on a 16 GB card.

- **CUDA-specific seams to port:** `store.py:35` `cudaHostRegister` via
  `torch.cuda.cudart()`; `store.py:46` `torch.cuda.synchronize()`;
  `executor.py:48` `torch.cuda.is_current_stream_capturing()`;
  `executor.py:79-81` `.pin_memory()` + `non_blocking` H2D (torch-level, portable);
  `bind_device_gather` builds a device pointer table (needs an NPU equivalent —
  Ascend has no flat device-pointer indirection like CUDA's; likely needs the
  slot index rewrite to stay tensor-based, which it already is).
- **UVM is not portable:** sglang#20126 (closed, bot-stale after 125 d) did
  UVM read-through for GLM-5-FP8 on 8×H20 with frequency-based resident selection
  and draft-aware config — its mechanism cannot port to Ascend (no UVM); only its
  frequency-based resident-selection idea is reusable.

### 2. note — generic offloaders already in tree

`python/sglang/srt/utils/offloader.py`: `OffloaderV1` (`--cpu-offload-gb`,
streams whole modules per forward) and `OffloaderV2` (`--offload-group-size`,
N-buffer prefetch pool, CUDA-graph aware; cf. vllm#29941). sglang#24531 (closed,
stale) extended V2 to all-weight offload at >5× V1 throughput. These offload
everything routed or not — the wrong granularity for MoE, but V2's prefetch/sync
machinery is the reference for hiding page-in latency.

### 3. note — existing NPU work

- sglang#43824 (open): one-line fix making `OffloaderV1` request pinned CPU
  storage on NPU so the copy-back survives graph capture — evidence that
  pinned-host + H2D + graph capture works on the Ascend stack.
- sglang#34673 (closed, WIP-stale): KTransformers CPU-expert offload for
  DeepSeek-V4 on Ascend 910B — computes offloaded experts on CPU (different
  philosophy: CPU compute vs Paged Experts' all-device compute). Reopenable but
  large; kt kernels are a fork dependency.
- sglang#46520 (open RFC): XPU extensions to Paged Experts — Lookahead Expert
  Scheduling (next-layer expert prefetch, training-free), SSD tier, shared slot
  pool. LES is device-agnostic and is the answer to Paged Experts' known weak
  spot (synchronous gather on miss, and falling behind under concurrency at
  small K/E).
- vllm-ascend has no expert-weight offload (only KV offload, `sfa_kv_offload.py`);
  nothing to adopt from there.
- Issue sglang#43139 asks for a plugin-provided weight offloader — relevant if an
  NPU paged-experts port should live outside shared code.

### 4. should-fix-if-pursued — gaps for NPU, in order

1. Device abstraction: paged_experts calls `torch.cuda.*` directly (4 sites
   above). Per CLAUDE.md ordering, a small hardware_backend hook (pin/register,
   is-capturing, sync) is the intended shape — CUDA default byte-identical.
2. Host→NPU copy: use `copy_` on a side torch_npu stream with pinned host source;
   verify overlap with compute on the box before promising latency hiding.
3. Parallelism: `check_paged_experts_compat` (paged_experts/method.py:~300) hard-rejects
   any tp/ep/pp/dp > 1 and `--enable-eplb` — a single-GPU scope guard, not a technical
   blocker. **TP is feasible and the natural first step:** the router is replicated
   under TP, so topk_ids (and therefore the page-in set and slot plan) are identical
   on every rank with no cross-rank coordination; each rank pages in its own 1/TP
   weight shard. Device MoE memory per rank = K/E × expert_params/TP; host RAM per
   rank = expert_params/TP. Needs: relax the assert, verify slot-plan determinism
   under graph capture, TP-format tests. **EP/EPLB stays out of scope** (per-rank
   expert sets differ → gather lands on the token critical path; EPLB breaks the
   slot mapping). DCP is orthogonal (attention-side) and needs no interaction.
   Goal stated 2026-11-02: fit a bigger model on A2/A3 hardware via freed device
   memory; feature stands alone, no DCP coupling.
4. Graph capture on NPU: slot planning is capturable on CUDA; the planning ops'
   compatibility with NPU compiled-graph/graph capture must be verified on the box.

## Not doing

- UVM/read-through port (#20126 mechanism) — no UVM on Ascend.
- KTransformers CPU-compute path (#34673) — fork-kernel dependency, CPU compute
  for MoE contradicts the all-device-compute goal.
- SSD tier (from #46520) — not needed for the box's RAM size.

## Minimal validation if pursued

Local: CPU unit tests for the slot-table / residency logic mirroring
`test/registered/` style. Box: GLM-5.2 with `--enable-paged-experts`-style flag,
compare tok/s and memory vs `--cpu-offload-gb` baseline; check box H2D bandwidth
overlaps compute before any perf claim.
