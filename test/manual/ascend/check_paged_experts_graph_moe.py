"""Check synthetic BF16 MoE graph replay and eager handoff on Ascend.

Run from the repo with torch-npu, Triton-Ascend and CANN available:
    PYTHONPATH=python python test/manual/ascend/check_paged_experts_graph_moe.py
"""

import argparse
import math
from contextlib import ExitStack
from types import SimpleNamespace

import torch
import triton
import triton.language as tl
from bench_paged_experts_graph_transfer import (
    _BLOCK_WORDS,
    _WORD_BYTES,
    gather_rows,
    load_acl,
    mapped_host_buffer,
)

_EXPERTS = 6
_HIDDEN = 2048
_INTERMEDIATE = 512
_TOP_K = 2
_ROUTED_SCALE = 1.25


@triton.jit
def plan_residency(
    ids,
    weights,
    live_tokens,
    slot_expert,
    expert_slot,
    last_use,
    step,
    src,
    dst,
    count,
    run_ids,
    run_weights,
    routes: tl.constexpr,
    slots: tl.constexpr,
    experts: tl.constexpr,
    top_k: tl.constexpr,
):
    slot_ids = tl.arange(0, slots)
    resident = tl.load(slot_expert + slot_ids)
    ages = tl.load(last_use + slot_ids)
    now = tl.load(step) + 1
    live_routes = tl.load(live_tokens) * top_k
    # Protect every requested resident before assigning any misses.
    for route in range(routes):
        expert = tl.load(ids + route)
        valid = (route < live_routes) & (expert >= 0) & (expert < experts)
        ages = tl.where(valid & (resident == expert), now, ages)
    tl.store(last_use + slot_ids, ages)

    misses = 0
    for route in range(routes):
        expert = tl.load(ids + route)
        valid = (route < live_routes) & (expert >= 0) & (expert < experts)
        slot = 0
        weight = 0.0
        if valid:
            resident = tl.load(slot_expert + slot_ids)
            slot = tl.min(tl.where(resident == expert, slot_ids, slots), 0)
            if slot == slots:
                ages = tl.load(last_use + slot_ids)
                oldest = tl.min(tl.where(ages != now, ages, 2147483647), 0)
                slot = tl.min(tl.where(ages == oldest, slot_ids, slots), 0)
                evicted = tl.min(tl.where(slot_ids == slot, resident, 2147483647), 0)
                resident = tl.where(slot_ids == slot, expert, resident)
                ages = tl.where(slot_ids == slot, now, ages)
                tl.store(slot_expert + slot_ids, resident)
                tl.store(last_use + slot_ids, ages)
                tl.store(expert_slot + evicted, -1)
                tl.store(expert_slot + expert, slot)
                tl.store(src + misses, expert)
                tl.store(dst + misses, slot)
                misses += 1
            weight = tl.load(weights + route)
        tl.store(run_ids + route, slot)
        tl.store(run_weights + route, weight)
    tl.store(step, now)
    tl.store(count, misses)


def run_moe(layer, hidden, ids, weights):
    from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul
    from sglang.srt.hardware_backend.npu.moe.routing import (
        npu_moe_finalize_routing,
        npu_moe_init_routing,
    )
    from sglang.srt.hardware_backend.npu.moe.standard_dispatch import GROUP_LIST_TYPE

    routed = npu_moe_init_routing(
        hidden, ids, layer.w13_weight.shape[0], active_num=ids.numel()
    )
    gmm = GroupedMatmul()
    values = routed.hidden_states
    for prefix in ("w13", "w2"):
        values = gmm.forward(
            layer,
            prefix,
            values,
            routed.expert_tokens,
            hidden.dtype,
            group_list_type=GROUP_LIST_TYPE,
            transposed=False,
        )
        if prefix == "w13":
            values = torch.ops.npu.npu_swiglu(values)
    return npu_moe_finalize_routing(
        values, routed.expanded_row_idx, weights.to(hidden.dtype), ids
    )


class SyntheticMoE:
    def apply(self, layer, dispatch_output):
        topk = dispatch_output.topk_output
        return SimpleNamespace(
            hidden_states=run_moe(
                layer, dispatch_output.hidden_states, topk.topk_ids, topk.topk_weights
            )
        )


def finish_graph(graph):
    torch.npu.synchronize()
    graph.reset()


def capture_decode(resources, layer, hidden, weights, residency, transfers):
    device = hidden.device
    ids = torch.tensor([[0, 1]] * len(hidden), dtype=torch.int32, device=device)
    live = torch.full((1,), len(hidden), dtype=torch.int32, device=device)
    r = residency
    num_slots = r.slot_expert.numel()
    assert ids.numel() <= num_slots

    def forward():
        slot_ids = torch.empty_like(ids)
        slot_weights = torch.empty_like(weights)
        plan_residency[(1,)](
            ids,
            weights,
            live,
            r.slot_expert,
            r.expert_slot,
            r.slot_lastuse,
            r.step,
            r.src,
            r.dst,
            r.count,
            slot_ids,
            slot_weights,
            ids.numel(),
            num_slots,
            _EXPERTS,
            _TOP_K,
        )
        for address, tensor in transfers:
            row_words = tensor[0].numel() * tensor.element_size() // _WORD_BYTES
            gather_rows[(row_words // _BLOCK_WORDS, num_slots)](
                address,
                r.src,
                r.dst,
                r.count,
                tensor.view(torch.int32),
                row_words,
                _BLOCK_WORDS,
            )
        return run_moe(layer, hidden, slot_ids, slot_weights) * _ROUTED_SCALE

    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(2):
            forward()
    stream.synchronize()
    graph = torch.npu.NPUGraph()
    resources.callback(finish_graph, graph)
    with torch.npu.graph(graph, stream=stream, auto_dispatch_capture=True):
        output = forward()
    torch.npu.current_stream().wait_stream(stream)
    return graph, ids, live, output


def check_output(label, actual, expected):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    max_error = (actual - expected).abs().max().item()
    torch.testing.assert_close(
        actual, expected, atol=0.008, rtol=0.02, msg=lambda msg: f"{label}: {msg}"
    )
    return max_error


def check_residency(label, residency, layer, store):
    resident = residency.slot_expert.cpu().long()
    assert ((resident >= 0) & (resident < _EXPERTS)).all(), label
    assert resident.unique().numel() == len(resident), label
    inverse = torch.full((_EXPERTS,), -1, dtype=torch.int32)
    inverse[resident] = torch.arange(len(resident), dtype=torch.int32)
    assert torch.equal(residency.expert_slot.cpu(), inverse), label
    for name, tensor in store.host.items():
        actual = getattr(layer, name).cpu().view(torch.int32)
        assert torch.equal(actual, tensor[resident].view(torch.int32)), (
            f"{label}: {name}"
        )
    return resident.tolist()


def run_cases(device, slots, acl):
    from sglang.srt.hardware_backend.npu.moe.paged_experts import AscendRunnerContract
    from sglang.srt.layers.moe.paged_experts.executor import EagerExecutor
    from sglang.srt.layers.moe.paged_experts.residency import DeviceResidency, LRUPolicy
    from sglang.srt.layers.moe.paged_experts.store import HostExpertStore
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    dtype = torch.bfloat16
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn((3, _HIDDEN), generator=generator).to(device, dtype)
    weights = torch.tensor([[0.75, 0.25], [0.25, 0.75], [0.625, 0.375]], device=device)
    row_dims = {
        "w13_weight": ((2 * _INTERMEDIATE, _HIDDEN), dtype),
        "w2_weight": ((_HIDDEN, _INTERMEDIATE), dtype),
    }
    with ExitStack() as resources:
        mapped = {}

        def alloc(shape, dtype):
            size = math.prod(shape) * dtype.itemsize
            host, address = resources.enter_context(mapped_host_buffer(acl, size))
            tensor = host.view(dtype).reshape(shape)
            tensor.copy_(torch.randn(shape, generator=generator) / math.sqrt(shape[-1]))
            mapped[shape] = torch.tensor([address], dtype=torch.int64, device=device)
            return tensor

        store = HostExpertStore(row_dims, _EXPERTS, alloc=alloc)
        unpaged = SimpleNamespace(
            **{name: tensor.to(device) for name, tensor in store.host.items()}
        )
        layer = SimpleNamespace(
            **{name: getattr(unpaged, name)[:slots].clone() for name in store.host}
        )
        transfers = [
            (mapped[tensor.shape], getattr(layer, name))
            for name, tensor in store.host.items()
        ]
        state = DeviceResidency(_EXPERTS, slots, device)
        graphs = {
            bs: capture_decode(
                resources, layer, hidden[:bs], weights[:bs], state, transfers
            )
            for bs in range(1, slots // _TOP_K + 1)
        }

        def replay(case, routes, misses, live=None):
            bs = len(routes)
            live = bs if live is None else live
            graph, ids, live_tokens, output = graphs[bs]
            ids.copy_(torch.tensor(routes, dtype=ids.dtype))
            live_tokens.fill_(live)
            torch.npu.synchronize()
            graph.replay()
            actual_misses = state.count.item()
            assert actual_misses == misses, (
                f"{case}: expected {misses} misses, got {actual_misses}"
            )
            expected = run_moe(unpaged, hidden[:live], ids[:live], weights[:live])
            error = check_output(case, output[:live], expected * _ROUTED_SCALE)
            if live < bs:
                assert output[live:].cpu().count_nonzero().item() == 0, case
            resident = check_residency(case, state, layer, store)
            print(
                f"{case}: misses={misses}, slots={resident}, max_abs_error={error:g}",
                flush=True,
            )

        replay(f"K={slots} hit", [[0, 1]], 0)
        if slots == 2:
            replay("K=2 first miss", [[1, 3]], 1)
            replay("K=2 next miss", [[3, 2]], 1)
            return

        replay("K=4 shared miss", [[0, 4], [2, 4]], 1)
        replay("K=4 shape switch", [[2, 3], [4, 5]], 2)
        replay("K=4 padded row", [[2, 3], [0, 1]], 0, live=1)
        ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device=device)
        eager = EagerExecutor(
            base_method=SyntheticMoE(),
            store=store,
            policy=LRUPolicy(slots),
            contract=AscendRunnerContract(),
            num_experts=_EXPERTS,
            routed_scaling_factor=1.0,
        )
        dispatch = StandardDispatchOutput(
            hidden,
            None,
            StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=None),
        )
        # Exercise the existing eager planner and the device/host residency handoff.
        state.load_into(eager.policy, state.count[:0])
        actual = eager.run(layer, dispatch).hidden_states * _ROUTED_SCALE
        state.store_from(eager.policy)
        expected = run_moe(unpaged, hidden, ids, weights) * _ROUTED_SCALE
        error = check_output("K=4 eager overflow", actual, expected)
        resident = check_residency("K=4 eager overflow", state, layer, store)
        print(
            f"K=4 eager overflow: slots={resident}, max_abs_error={error:g}", flush=True
        )
        replay("K=4 replay after eager", [[0, 1]], 0)
        replay("K=4 second shape after eager", [[2, 3], [4, 5]], 2)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    torch.npu.set_device(args.device)
    acl = load_acl()
    for slots in (2, 4):
        run_cases(args.device, slots, acl)
    print("All synthetic Ascend MoE graph cases passed.")


if __name__ == "__main__":
    main()
