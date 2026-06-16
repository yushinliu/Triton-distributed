################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""IBGDA bandwidth smoke test through Triton device APIs.

Example:
  TRITON_DIST_SHMEM_WRAPPER=1 NVSHMEM_IBGDA_SUPPORT=1 \
  torchrun --nproc_per_node=8 python/triton_dist/test/nvidia/test_ibgda_bw.py \
      --bytes 268435456 --iters 50 --aggregate
"""

import argparse
import os
import socket
import subprocess
from pathlib import Path
from typing import List, Tuple

# Must be set before Triton JIT functions are constructed.
os.environ.setdefault("TRITON_DIST_SHMEM_WRAPPER", "1")
os.environ.setdefault("NVSHMEM_IBGDA_SUPPORT", "1")

import torch
import torch.distributed
import triton
import triton.language as tl
import triton_dist

from triton_dist.language.extra.cuda import libibgda_device
from triton_dist.language.extra.cuda.language_extra import tid
from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, finalize_distributed, initialize_distributed,
                               nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_free_tensor_sync)


def _parse_ib_state(text: str) -> Tuple[bool, str]:
    normalized = text.strip()
    return normalized.startswith("4:") or normalized.upper() == "ACTIVE", normalized


def list_ib_ports() -> List[Tuple[str, str, str, str]]:
    base = Path("/sys/class/infiniband")
    ports = []
    if base.exists():
        for dev in sorted(base.iterdir()):
            ports_dir = dev / "ports"
            if not ports_dir.exists():
                continue
            for port in sorted(ports_dir.iterdir()):
                state = (port / "state").read_text().strip() if (port / "state").exists() else "unknown"
                phys_state = ((port / "phys_state").read_text().strip()
                              if (port / "phys_state").exists() else "unknown")
                rate = (port / "rate").read_text().strip() if (port / "rate").exists() else "unknown"
                ports.append((dev.name, port.name, state, f"phys={phys_state}, rate={rate}"))
        return ports

    try:
        output = subprocess.check_output(["ibstat"], text=True, stderr=subprocess.STDOUT)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    current_ca = "unknown"
    current_port = "unknown"
    current_state = "unknown"
    current_phys = "unknown"
    current_rate = "unknown"
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("CA '"):
            current_ca = line.split("'", 2)[1]
        elif line.startswith("Port "):
            current_port = line.split()[1].rstrip(":")
        elif line.startswith("State:"):
            current_state = line.split(":", 1)[1].strip()
        elif line.startswith("Physical state:"):
            current_phys = line.split(":", 1)[1].strip()
        elif line.startswith("Rate:"):
            current_rate = line.split(":", 1)[1].strip()
            ports.append((current_ca, current_port, current_state, f"phys={current_phys}, rate={current_rate}"))
    return ports


def active_ib_ports() -> List[Tuple[str, str, str, str]]:
    return [port for port in list_ib_ports() if _parse_ib_state(port[2])[0]]


def print_ib_ports(rank: int):
    if rank != 0:
        return
    ports = list_ib_ports()
    if not ports:
        print("No InfiniBand ports found from /sys/class/infiniband or ibstat")
        return
    print("InfiniBand ports:")
    for ca, port, state, details in ports:
        print(f"  {ca}/port{port}: state={state}, {details}")


@triton_dist.jit
def _ibgda_put_bw_kernel(src, dst, signal, target_rank: tl.constexpr, rank: tl.constexpr, nbytes: tl.constexpr,
                         chunk_bytes: tl.constexpr, use_signal: tl.constexpr, NUM_WARPS: tl.constexpr):
    thread_idx = tid(axis=0)
    warp_id = thread_idx // 32
    warp_count = tl.num_programs(axis=0) * NUM_WARPS
    global_warp_id = tl.program_id(axis=0) * NUM_WARPS + warp_id
    offset = global_warp_id * chunk_bytes
    stride = warp_count * chunk_bytes

    while offset < nbytes:
        bytes_this = tl.minimum(nbytes - offset, chunk_bytes)
        libibgda_device.putmem_nbi_warp(dst + rank * nbytes + offset, src + offset, bytes_this, target_rank)
        offset += stride

    if global_warp_id == 0:
        if use_signal:
            libibgda_device.signal_op(signal + rank, 1, libibgda_device.NVSHMEM_SIGNAL_SET, target_rank)
        libibgda_device.quiet()


@triton_dist.jit
def _ibgda_wait_signal_kernel(signal, source_rank: tl.constexpr):
    if tl.program_id(axis=0) == 0 and tid(axis=0) == 0:
        libibgda_device.signal_wait_until(signal + source_rank, libibgda_device.NVSHMEM_CMP_EQ, 1)


def _rank_bytes(args) -> int:
    if args.bytes <= 0:
        raise ValueError("--bytes must be positive")
    return int(args.bytes)


def _make_source(nbytes: int) -> torch.Tensor:
    return torch.empty((nbytes, ), dtype=torch.int8, device="cuda")


def run_pair(args, rank: int, world_size: int, aggregate: bool) -> float:
    nbytes = _rank_bytes(args)
    chunk_bytes = min(args.chunk_bytes, nbytes)
    chunk_bytes = max(32, (chunk_bytes // 32) * 32)
    num_warps = args.num_warps
    total_warps = max(1, min(args.blocks * num_warps, triton.cdiv(nbytes, chunk_bytes)))
    grid = (triton.cdiv(total_warps, num_warps), )

    dst = None
    signal = None
    try:
        src = _make_source(nbytes)
        dst = nvshmem_create_tensor((world_size * nbytes, ), torch.int8)
        signal = nvshmem_create_tensor((world_size, ), NVSHMEM_SIGNAL_DTYPE)
        dst.fill_(0)
        signal.fill_(0)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        if aggregate:
            targets = (rank + 1) % world_size
            sources_for_rank = (rank - 1 + world_size) % world_size
            active_sender = True
        else:
            targets = args.dst_rank
            sources_for_rank = args.src_rank
            active_sender = rank == args.src_rank

        for _ in range(args.warmup):
            if active_sender:
                _ibgda_put_bw_kernel[grid](src, dst, signal, targets, rank, nbytes, chunk_bytes, args.use_signal,
                                           num_warps, num_warps=num_warps)
            if args.use_signal and (aggregate or rank == args.dst_rank):
                _ibgda_wait_signal_kernel[(1, )](signal, sources_for_rank, num_warps=1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            signal.fill_(0)
            torch.cuda.synchronize()

        torch.distributed.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            if active_sender:
                _ibgda_put_bw_kernel[grid](src, dst, signal, targets, rank, nbytes, chunk_bytes, args.use_signal,
                                           num_warps, num_warps=num_warps)
            if args.use_signal and (aggregate or rank == args.dst_rank):
                _ibgda_wait_signal_kernel[(1, )](signal, sources_for_rank, num_warps=1)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            signal.fill_(0)
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end) / args.iters
        torch.distributed.barrier()

        moved_bytes = nbytes * (world_size if aggregate else 1)
        bw_gbs = moved_bytes / elapsed_ms / 1e6
        if rank == 0:
            mode = "all-pairs aggregate" if aggregate else f"rank {args.src_rank} -> rank {args.dst_rank}"
            print(f"IBGDA put_nbi_warp {mode}: {nbytes} bytes/rank, {elapsed_ms:.3f} ms, {bw_gbs:.2f} GB/s")
            print(f"  grid={grid}, num_warps={num_warps}, chunk_bytes={chunk_bytes}, use_signal={args.use_signal}")
        return bw_gbs
    finally:
        if signal is not None:
            nvshmem_free_tensor_sync(signal)
        if dst is not None:
            nvshmem_free_tensor_sync(dst)


def parse_args():
    parser = argparse.ArgumentParser(description="Triton IBGDA bandwidth test")
    parser.add_argument("--bytes", type=int, default=256 * 1024 * 1024, help="bytes sent by each active source rank")
    parser.add_argument("--chunk_bytes", type=int, default=64 * 1024, help="bytes transferred by one warp per RMA call")
    parser.add_argument("--blocks", type=int, default=256, help="maximum Triton CTAs used by the copy kernel")
    parser.add_argument("--num_warps", type=int, default=8, help="warps per Triton CTA")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--src_rank", type=int, default=0)
    parser.add_argument("--dst_rank", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true", help="run ring all-pairs traffic and report aggregate GB/s")
    parser.add_argument("--use_signal", action="store_true", help="wait for an IBGDA signal on the destination PE")
    parser.add_argument("--allow_inactive_ib", action="store_true", help="run even when no ACTIVE IB port is detected")
    parser.add_argument("--disable_p2p", action="store_true", help="set NVSHMEM_DISABLE_P2P=1 before NVSHMEM init")
    parser.add_argument("--check_ib_only", action="store_true", help="print IB port state and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.disable_p2p:
        os.environ.setdefault("NVSHMEM_DISABLE_P2P", "1")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.check_ib_only:
        print_ib_ports(rank)
        active = active_ib_ports()
        if rank == 0:
            print(f"ACTIVE IB ports: {len(active)} on {socket.gethostname()}")
        return

    has_active_ib = bool(active_ib_ports())
    if not has_active_ib and not args.allow_inactive_ib:
        if rank == 0:
            print_ib_ports(rank)
            print("Skip IBGDA bandwidth run: no ACTIVE InfiniBand port detected. "
                  "Enable IB ports or pass --allow_inactive_ib for a compile/runtime smoke test.")
        return

    if world_size < 2:
        raise RuntimeError("IBGDA bandwidth test needs at least 2 ranks")
    if not (0 <= args.src_rank < world_size and 0 <= args.dst_rank < world_size):
        raise RuntimeError("--src_rank/--dst_rank must be in [0, WORLD_SIZE)")
    if args.src_rank == args.dst_rank:
        raise RuntimeError("--src_rank and --dst_rank must be different")

    torch.cuda.set_device(local_rank)
    pg = initialize_distributed()
    try:
        if rank == 0:
            print(f"TRITON_DIST_SHMEM_WRAPPER={os.environ.get('TRITON_DIST_SHMEM_WRAPPER')}")
            print(f"NVSHMEM_IBGDA_SUPPORT={os.environ.get('NVSHMEM_IBGDA_SUPPORT')}")
            print(f"NVSHMEM_DISABLE_P2P={os.environ.get('NVSHMEM_DISABLE_P2P', '')}")
        run_pair(args, rank, world_size, aggregate=args.aggregate)
    finally:
        finalize_distributed()
        del pg


if __name__ == "__main__":
    main()
