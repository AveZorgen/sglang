"""Measure host memory for the BF16 routed expert store from PR #51878.

Run each command in a fresh process on Ascend, from the repository root:

    PYTHONPATH=python python test/manual/ascend/check_paged_experts_host_rss.py --layers 40
    PYTHONPATH=python python test/manual/ascend/check_paged_experts_host_rss.py --experts 17

The first run reproduces the Qwen3.5-35B-A3B store at TP=1 (60 GiB).
The second requests 68 and 34 MiB. Per-allocation power-of-two rounding
would reserve 128 and 64 MiB. Compare the pinned allocator delta with both
predictions. RSS counts resident pages and may omit untouched block slack.
Linux VmLck/VmPin coverage depends on the driver.
"""

import argparse
import math
from pathlib import Path

import torch


def snapshot():
    metrics = {}
    for path, fields in (
        ("/proc/self/smaps_rollup", {"Rss"}),
        ("/proc/self/status", {"VmLck", "VmPin"}),
    ):
        for line in Path(path).read_text().splitlines():
            name, _, value = line.partition(":")
            if name in fields:
                metrics[name] = int(value.split()[0]) * 1024

    host_stats = getattr(torch.npu.memory, "host_memory_stats", None)
    if host_stats is not None:
        try:
            stats = host_stats()
        except (AttributeError, RuntimeError) as exc:
            print(f"Could not read NPU host allocator counters: {exc}")
            return metrics
        for field in ("allocated_bytes.current", "active_bytes.current"):
            if field in stats:
                metrics[field] = stats[field]
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="stores to hold simultaneously (full model: 40)",
    )
    parser.add_argument(
        "--experts",
        type=int,
        default=256,
        help="routed experts per layer (full model: 256)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print allocation sizes and exit"
    )
    args = parser.parse_args()
    if min(args.layers, args.experts) < 1:
        parser.error("--layers and --experts must be positive")

    # Qwen3.5-35B-A3B: hidden size 2048, expert intermediate size 512.
    rows = {
        "w13_weight": ((2 * 512, 2048), torch.bfloat16),
        "w2_weight": ((2048, 512), torch.bfloat16),
    }
    mib = 1024**2
    requested = rounded = 0
    for name, (shape, dtype) in rows.items():
        size = args.experts * math.prod(shape) * dtype.itemsize
        block = 1 << (size - 1).bit_length()
        requested += args.layers * size
        rounded += args.layers * block
        print(f"{name}: {args.layers} x {size / mib:g} MiB")
    print(f"Requested store: {requested / mib:g} MiB")
    print(
        f"If rounded per allocation: {rounded / mib:g} MiB ({rounded / requested:.3f}x)",
        flush=True,
    )
    if args.dry_run:
        return

    import torch_npu

    from sglang.srt.hardware_backend.npu.moe.paged_experts import AscendRunnerContract
    from sglang.srt.layers.moe.paged_experts.store import HostExpertStore

    torch.npu.set_device(0)
    torch.set_num_threads(1)
    alloc = AscendRunnerContract.host_alloc
    # Initialize the NPU context, lazy imports and allocator before measuring.
    warmup = alloc((1,), torch.bfloat16)
    warmup.fill_(1)
    torch.npu.synchronize()
    print(
        f"torch {torch.__version__}, torch_npu {torch_npu.__version__}, {torch.npu.get_device_name(0)}"
    )
    before = snapshot()
    stores = []
    for _ in range(args.layers):
        store = HostExpertStore(rows, args.experts, alloc=alloc)
        for tensor in store.host.values():
            if not tensor.is_pinned():
                raise RuntimeError("The Ascend store returned unpinned host memory")
            # Fault in the requested pages without a second weight copy.
            tensor.fill_(1)
        stores.append(store)
    after = snapshot()

    print(
        f"{'Metric':24} {'Before MiB':>14} {'After MiB':>14} {'Delta MiB':>14} {'Delta/store':>12}"
    )
    for name, start in before.items():
        if name not in after:
            continue
        end = after[name]
        delta = end - start
        print(
            f"{name:24} {start / mib:14.2f} {end / mib:14.2f} {delta / mib:14.2f} {delta / requested:11.3f}x"
        )
    if (
        "allocated_bytes.current" not in before
        or "allocated_bytes.current" not in after
    ):
        print("Pinned allocator block bytes unavailable in this torch_npu build.")


if __name__ == "__main__":
    main()
