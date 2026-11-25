"""Paged experts on Ascend: the runner contract of the existing Ascend MoE runner.

The Ascend runner applies router weights and combines top-k outputs in finalize
routing. Each wave returns a partial ``[tokens, hidden]`` output. Summing the waves
matches the unpaged runner's output, and callers handle any separate routed scaling.
The runner uses ``num_experts = K`` so the routing tables match the slot ids.

Registered with paged experts when this module is imported (from the Ascend runner's
module), so the shared paged-experts code carries no Ascend imports.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.paged_experts.formats import UnquantizedFormat
from sglang.srt.layers.moe.paged_experts.runners import (
    RunnerContract,
    register_runner_contract,
    run_on_slots,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)


def _ascend_host_alloc(shape, dtype: torch.dtype) -> torch.Tensor:
    """Allocate mapped rows for graph page-in, or torch-pinned rows for eager copies."""
    if not cuda_graph_fully_disabled():
        from sglang.srt.hardware_backend.npu.moe.paged_experts_graph import (
            mapped_host_alloc,
        )

        return mapped_host_alloc(shape, dtype)

    from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory

    return alloc_with_pin_memory(
        dims=tuple(shape), dtype=dtype, device="cpu", pin_memory=True, allocator=None
    )


class AscendRunnerContract(RunnerContract):
    #: Pinned host rows through the NPU allocator, not the store's CUDA default.
    host_alloc = staticmethod(_ascend_host_alloc)

    def runner_config(self, moe_runner_config, num_slots: int):
        # Slot ids are the runner's expert ids: its routing tables and grouped GEMM
        # group lists must be K wide, and its own combine (the finalize op) stays on.
        return replace(
            moe_runner_config,
            num_experts=num_slots,
            num_local_experts=num_slots,
            no_combine=False,
            no_combine_keep_router_weight=False,
        )

    def make_device_executor(self, **kwargs):
        from sglang.srt.hardware_backend.npu.moe.paged_experts_graph import (
            AscendDeviceExecutor,
        )

        return AscendDeviceExecutor(contract=self, **kwargs)

    def validate(self, method) -> None:
        if method.graphs and not (
            check_cuda_graph_backend(Phase.DECODE, Backend.FULL)
            and check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
            and not method.auto_size
            and isinstance(method.format, UnquantizedFormat)
            and method.params_dtype == torch.bfloat16
        ):
            raise RuntimeError(
                "Paged experts on Ascend: graphs require full decode, disabled prefill graphs, "
                "an explicit --paged-experts-num-resident, and unquantized BF16 weights"
            )
        if not envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get():
            raise RuntimeError(
                "Paged experts on Ascend: set SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1 so "
                "the expert weights stay ND: the host store must equal the GPU rows "
                "verbatim, which the NZ format cast breaks"
            )

    def run_wave(
        self, base_method, layer, dispatch_output, slots, merged, num_entries=None
    ):
        # Routes outside the wave run as slot 0 with weight 0: they compute, and the
        # finalize op zeroes them, so waves add cleanly.
        partial = run_on_slots(base_method, layer, dispatch_output, slots)
        if merged is None:
            return partial
        return merged + partial

    def finish(self, merged, hidden_states, routed_scaling_factor: float):
        # Match the unpaged runner's weighted sum. The model owns routed scaling.
        out = hidden_states.new_empty(merged.shape[0], merged.shape[-1])
        out.copy_(merged)
        return out


register_runner_contract(MoeRunnerBackend.ASCEND.value, AscendRunnerContract())
