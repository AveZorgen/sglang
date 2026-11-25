"""Check captured expert page-in against ACL DMA on one Ascend A2/A3.

Run with torch-npu, Triton-Ascend, and CANN sourced, for example:
    python test/manual/ascend/bench_paged_experts_graph_transfer.py --slots 64

The two row sizes match Qwen3.5-35B-A3B BF16 ND. Timings are average wall time
for a fixed plan, including submission and device completion. Plan updates
and correctness checks are outside timing.
"""

import argparse
import ctypes
import json
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
import triton
import triton.language as tl

_WORD_BYTES = 4
_BLOCK_WORDS = 4096
_ROW_BYTES = {"w13": 4 * 1024**2, "w2": 2 * 1024**2}
_HOST_ALIGNMENT = 64 * 1024
_ACL_HOST_REG_MAPPED = 0x2
_ACL_HOST_REG_PINNED = 0x10000000
_ACL_MEMCPY_HOST_TO_DEVICE = 1


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
        "aclrtMemcpyAsync": [pointer, size, pointer, size, ctypes.c_int, pointer],
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
def mapped_host_buffer(acl, byte_count):
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
        data = (ctypes.c_int32 * (byte_count // _WORD_BYTES)).from_address(host)
        yield torch.frombuffer(data, dtype=torch.int32), mapped.value
    finally:
        if registered:
            torch.npu.synchronize()
            acl_call(acl.aclrtHostUnregister, host)
        acl_call(acl.aclrtFreeHost, allocation)


def set_plan(src, dst, count, pairs):
    for tensor, indices in ((src, [e for e, _ in pairs]), (dst, [s for _, s in pairs])):
        if indices:
            tensor[: len(indices)].copy_(torch.tensor(indices, dtype=tensor.dtype))
    count.fill_(len(pairs))


def elapsed_ms(run, iterations):
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        run()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def process_memory_kib():
    counters = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"VmRSS", "VmLck", "VmPin"}:
            counters[name] = int(value.split()[0])
    return counters


def probe(args, acl):
    slots = args.slots
    expert_count = slots + 1
    src = torch.zeros(slots, dtype=torch.int32, device=args.device)
    dst = torch.zeros_like(src)
    count = torch.zeros(1, dtype=torch.int32, device=args.device)
    generator = torch.Generator().manual_seed(0)
    memory_before = process_memory_kib()

    with ExitStack() as resources:
        host_addresses = {}
        mapped_addresses = {}
        for name, row_bytes in _ROW_BYTES.items():
            host, mapped = resources.enter_context(
                mapped_host_buffer(acl, expert_count * row_bytes)
            )
            host.random_(generator=generator)
            host_addresses[name] = host.data_ptr()
            mapped_addresses[name] = torch.tensor(
                [mapped], dtype=torch.int64, device=args.device
            )
        print(
            json.dumps(
                {
                    "host_payload_bytes": expert_count * sum(_ROW_BYTES.values()),
                    "acl_host_requested_bytes": expert_count * sum(_ROW_BYTES.values())
                    + len(_ROW_BYTES) * (_HOST_ALIGNMENT - 1),
                    "process_memory_before_kib": memory_before,
                    "process_memory_after_kib": process_memory_kib(),
                }
            ),
            flush=True,
        )

        actual = {
            name: torch.empty(
                (slots, row_bytes // _WORD_BYTES), dtype=torch.int32, device=args.device
            )
            for name, row_bytes in _ROW_BYTES.items()
        }
        reference = {name: torch.empty_like(tensor) for name, tensor in actual.items()}

        def gather():
            for name, tensor in actual.items():
                row_words = tensor.shape[1]
                gather_rows[(row_words // _BLOCK_WORDS, slots)](
                    mapped_addresses[name],
                    src,
                    dst,
                    count,
                    tensor,
                    row_words,
                    _BLOCK_WORDS,
                )

        def copy_with_acl(pairs):
            stream = torch.npu.current_stream().npu_stream
            for name, row_bytes in _ROW_BYTES.items():
                host = host_addresses[name]
                device = reference[name].data_ptr()
                for expert, slot in pairs:
                    acl_call(
                        acl.aclrtMemcpyAsync,
                        device + slot * row_bytes,
                        row_bytes,
                        host + expert * row_bytes,
                        row_bytes,
                        _ACL_MEMCPY_HOST_TO_DEVICE,
                        stream,
                    )

        capture_stream = torch.npu.Stream()
        capture_stream.wait_stream(torch.npu.current_stream())
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.stream(capture_stream):
                set_plan(src, dst, count, [(0, 0)])
                capture_stream.synchronize()
                for _ in range(2):
                    gather()
            capture_stream.synchronize()
            with torch.npu.graph(
                graph, stream=capture_stream, auto_dispatch_capture=True
            ):
                gather()
            torch.npu.current_stream().wait_stream(capture_stream)

            cases = {
                "zero": [],
                "one": [(slots, slots - 1)],
                "one_changed": [(1, 0)],
                "full": [(slots - slot, slot) for slot in range(slots)],
                "zero_after_full": [],
            }
            for case, pairs in cases.items():
                # Comparing the whole table also checks that inactive slots stay untouched.
                for tensor in (*actual.values(), *reference.values()):
                    tensor.fill_(-1)
                set_plan(src, dst, count, pairs)
                # Raw ACL submissions bypass the torch-npu task queue.
                torch.npu.synchronize()
                graph.replay()
                copy_with_acl(pairs)
                torch.npu.synchronize()
                for name in actual:
                    if not torch.equal(actual[name].cpu(), reference[name].cpu()):
                        raise AssertionError(
                            f"{case}: {name} graph output differs from ACL DMA"
                        )
                print(
                    f"{case}: byte-for-byte match ({len(pairs)} copies per tensor)",
                    flush=True,
                )

            for case in ("zero", "one", "full"):
                pairs = cases[case]
                set_plan(src, dst, count, pairs)
                print(
                    json.dumps(
                        {
                            "case": case,
                            "bytes_per_iteration": len(pairs)
                            * sum(_ROW_BYTES.values()),
                            "graph_wall_ms": elapsed_ms(graph.replay, args.iterations),
                            "acl_wall_ms": elapsed_ms(
                                lambda: copy_with_acl(pairs), args.iterations
                            )
                            if pairs
                            else 0.0,
                        }
                    ),
                    flush=True,
                )
        finally:
            torch.npu.synchronize()
            graph.reset()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if min(args.slots, args.iterations) < 1:
        parser.error("--slots and --iterations must be positive")

    import torch_npu

    torch.npu.set_device(args.device)
    print(
        json.dumps(
            {
                "device": torch.npu.get_device_name(),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "triton": triton.__version__,
                "row_bytes": _ROW_BYTES,
                "args": vars(args),
            }
        ),
        flush=True,
    )
    probe(args, load_acl())


if __name__ == "__main__":
    main()
