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

from sglang.srt.environ import envs
from sglang.srt.layers.moe.paged_experts.runners import (
    CombiningContract,
    register_runner_contract,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled


class AscendRunnerContract(CombiningContract):
    drops_outside = False

    def runner_config(self, moe_runner_config, num_slots: int):
        config = self.combining_config(moe_runner_config, num_slots)
        return replace(config, num_experts=num_slots)

    def validate(self) -> None:
        if not cuda_graph_fully_disabled():
            raise RuntimeError(
                "Paged experts on Ascend: CUDA graphs are not supported yet; run with "
                "--disable-cuda-graph (every step takes the host-planned eager path)"
            )
        if not envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.get():
            raise RuntimeError(
                "Paged experts on Ascend: set SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1 so "
                "the expert weights stay ND: the host store must equal the GPU rows "
                "verbatim, which the NZ format cast breaks"
            )


register_runner_contract(MoeRunnerBackend.ASCEND.value, AscendRunnerContract())
