"""Mapped expert weights and single-wave decode on Ascend A2/A3."""

import ctypes
import math
import weakref
from contextlib import contextmanager
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe.paged_experts.executor import DeviceExecutor
from sglang.srt.runtime_context import get_forward

_BLOCK_WORDS = 4096
_HOST_ALIGNMENT = 64 * 1024
_ACL_HOST_REG_MAPPED = 0x2
_ACL_HOST_REG_PINNED = 0x10000000


@triton.jit
def gather_rows(
    host_address,
    src,
    dst,
    count,
    slots,
    row_words: tl.constexpr,
    block_words: tl.constexpr,
):
    copy = tl.program_id(1)
    # A2 mapped loads can fault when a load mask is used in place of this branch.
    if copy < tl.load(count):
        expert = tl.load(src + copy)
        slot = tl.load(dst + copy)
        host = tl.load(host_address).to(tl.pointer_type(tl.int32))
        slots = slots.to(tl.pointer_type(tl.int32))
        words = tl.program_id(0) * block_words + tl.arange(0, block_words)
        data = tl.load(host + expert.to(tl.int64) * row_words + words)
        tl.store(slots + slot.to(tl.int64) * row_words + words, data)


@lru_cache(maxsize=1)
def load_acl():
    acl = ctypes.CDLL("libascendcl.so")
    pointer = ctypes.c_void_p
    size = ctypes.c_size_t
    signatures = {
        "aclrtMallocHost": [ctypes.POINTER(pointer), size],
        "aclrtFreeHost": [pointer],
        "aclrtHostRegisterV2": [pointer, ctypes.c_uint64, ctypes.c_uint32],
        "aclrtHostGetDevicePointer": [
            pointer,
            ctypes.POINTER(pointer),
            ctypes.c_uint32,
        ],
        "aclrtHostUnregister": [pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(acl, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    return acl


def acl_call(function, *arguments):
    result = function(*arguments)
    if result:
        raise RuntimeError(f"{function.__name__} failed with ACL error {result}")


@contextmanager
def mapped_host_range(acl, byte_count):
    device = torch.npu.current_device()
    allocation = ctypes.c_void_p()
    acl_call(
        acl.aclrtMallocHost, ctypes.byref(allocation), byte_count + _HOST_ALIGNMENT - 1
    )
    if allocation.value is None:
        raise RuntimeError("aclrtMallocHost returned a null pointer")
    # vLLM-Ascend's Engram path uses 64 KiB alignment for mapped reads on A3.
    host = -(-allocation.value // _HOST_ALIGNMENT) * _HOST_ALIGNMENT
    mapped = ctypes.c_void_p()
    registered = False
    try:
        acl_call(
            acl.aclrtHostRegisterV2,
            host,
            byte_count,
            _ACL_HOST_REG_MAPPED | _ACL_HOST_REG_PINNED,
        )
        registered = True
        acl_call(acl.aclrtHostGetDevicePointer, host, ctypes.byref(mapped), 0)
        if mapped.value is None:
            raise RuntimeError("aclrtHostGetDevicePointer returned a null pointer")
        yield host, mapped.value
    finally:
        if registered:
            torch.npu.synchronize(device)
            acl_call(acl.aclrtHostUnregister, host)
        acl_call(acl.aclrtFreeHost, allocation)


def mapped_host_alloc(shape, dtype):
    size = math.prod(shape) * dtype.itemsize
    mapping = mapped_host_range(load_acl(), size)
    host, mapped = mapping.__enter__()
    try:
        buffer = (ctypes.c_byte * size).from_address(host)
        # All torch views share the frombuffer storage, which owns this buffer.
        owner = weakref.finalize(buffer, mapping.__exit__, None, None, None)
    except BaseException:
        mapping.__exit__(None, None, None)
        raise
    owner.atexit = False
    tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
    tensor._ascend_mapped_address = mapped
    return tensor


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
    slot_block: tl.constexpr,
):
    slot_ids = tl.arange(0, slot_block)
    active = slot_ids < slots
    resident = tl.load(slot_expert + slot_ids, mask=active, other=-1)
    ages = tl.load(last_use + slot_ids, mask=active, other=2147483647)
    now = tl.load(step) + 1
    live_routes = tl.load(live_tokens) * top_k
    # Protect every requested resident before assigning any misses.
    for route in range(routes):
        expert = tl.load(ids + route)
        valid = (route < live_routes) & (expert >= 0) & (expert < experts)
        ages = tl.where(valid & (resident == expert), now, ages)
    tl.store(last_use + slot_ids, ages, mask=active)

    misses = 0
    for route in range(routes):
        expert = tl.load(ids + route)
        valid = (route < live_routes) & (expert >= 0) & (expert < experts)
        slot = 0
        weight = 0.0
        if valid:
            resident = tl.load(slot_expert + slot_ids, mask=active, other=-1)
            slot = tl.min(tl.where(resident == expert, slot_ids, slots), 0)
            if slot == slots:
                ages = tl.load(last_use + slot_ids, mask=active, other=2147483647)
                oldest = tl.min(tl.where(ages != now, ages, 2147483647), 0)
                slot = tl.min(tl.where(ages == oldest, slot_ids, slots), 0)
                evicted = tl.min(tl.where(slot_ids == slot, resident, 2147483647), 0)
                resident = tl.where(slot_ids == slot, expert, resident)
                ages = tl.where(slot_ids == slot, now, ages)
                tl.store(slot_expert + slot_ids, resident, mask=active)
                tl.store(last_use + slot_ids, ages, mask=active)
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


class AscendDeviceExecutor(DeviceExecutor):
    def bind(self, layer):
        self.transfers = []
        for name, host in self.store.host.items():
            slots = getattr(layer, name).data
            if not host.numel():
                continue
            if (
                slots.dtype != torch.bfloat16
                or not slots.is_contiguous()
                or slots.shape[-1] % 2
            ):
                raise ValueError(
                    f"Ascend graph page-in requires contiguous BF16 rows of int32 words: {name}"
                )
            words = slots.view(torch.int32)
            row_words = words[0].numel()
            # Tile each row exactly. Masked mapped-host loads can fault on A2.
            block = min(_BLOCK_WORDS, row_words & -row_words)
            address = torch.tensor(
                [host._ascend_mapped_address], dtype=torch.int64, device=slots.device
            )
            self.transfers.append((address, words, row_words, block))

    def can_run(self):
        return get_forward().paged_experts_live_tokens is not None

    def prepare_routes(self, topk_ids, topk_weights):
        r = self.residency
        slots = r.slot_expert.numel()
        run_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=topk_ids.device)
        run_weights = torch.empty_like(
            topk_weights, memory_format=torch.contiguous_format
        )
        plan_residency[(1,)](
            topk_ids.int().contiguous(),
            topk_weights.contiguous(),
            get_forward().paged_experts_live_tokens,
            r.slot_expert,
            r.expert_slot,
            r.slot_lastuse,
            r.step,
            r.src,
            r.dst,
            r.count,
            run_ids,
            run_weights,
            topk_ids.numel(),
            slots,
            r.expert_slot.numel(),
            topk_ids.shape[-1],
            triton.next_power_of_2(slots),
        )
        for address, words, row_words, block in self.transfers:
            gather_rows[(row_words // block, slots)](
                address, r.src, r.dst, r.count, words, row_words, block
            )
        return run_ids, run_weights
