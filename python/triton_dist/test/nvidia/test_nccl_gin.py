################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import argparse
import datetime
import os

import torch
import torch.distributed as dist
import triton.language as tl

import triton_dist
import triton_dist.language as dl
from triton_dist import nccl_gin
from triton_dist.language.extra.cuda import libnccl_device
from triton_dist.utils import finalize_distributed, init_nccl_gin_by_torch_process_group


@triton_dist.jit
def _gin_alltoall_put_i32(dev_comm, send_win, recv_win, signal_win, rank_arg, nelems: tl.constexpr):
    peer = tl.program_id(0)
    rank = rank_arg
    nbytes: tl.constexpr = nelems * 4
    libnccl_device.gin_put_va_signal_inc_cta(
        dev_comm,
        recv_win,
        rank.to(tl.uint64) * nbytes,
        send_win,
        peer.to(tl.uint64) * nbytes,
        nbytes,
        signal_win,
        0,
        peer,
        0,
    )
    libnccl_device.gin_flush_cta(dev_comm, 0)


@triton_dist.jit
def _gin_wait_signal(dev_comm, signal_win, least):
    libnccl_device.gin_wait_va_signal_cta(dev_comm, signal_win, 0, least, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="NCCL GIN Triton smoke test")
    parser.add_argument("--nelems", type=int, default=1024)
    parser.add_argument("--skip-wait", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=1800),
    )
    ep_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")

    def mark(stage):
        print(f"[rank {rank}] {stage}", flush=True)

    send_win = recv_win = signal_win = None
    try:
        mark("probe")
        props = nccl_gin.probe_by_torch_process_group(ep_group)
        if rank == 0:
            print(f"NCCL GIN probe properties before dev-comm init: {props}")
        mark("init_nccl_gin")
        init_nccl_gin_by_torch_process_group(ep_group, barrier_count=0, gin_signal_count=0, gin_context_count=4)
        mark("alloc")

        send = nccl_gin.empty((world_size, args.nelems), dtype=torch.int32)
        recv = nccl_gin.empty((world_size, args.nelems), dtype=torch.int32)
        signal = nccl_gin.empty((1, ), dtype=torch.int64)
        recv.fill_(-1)
        signal.zero_()
        for peer in range(world_size):
            send[peer].fill_(rank * 1000 + peer)

        mark("register_send")
        send_win = nccl_gin.register_window(send, collective_symmetric=False)
        torch.cuda.synchronize()
        mark("register_recv")
        recv_win = nccl_gin.register_window(recv, collective_symmetric=False)
        torch.cuda.synchronize()
        mark("register_signal")
        signal_win = nccl_gin.register_window(signal, collective_symmetric=False, strict_ordering=True)
        torch.cuda.synchronize()
        mark("get_dev_comm")

        dev_comm = nccl_gin.get_dev_comm_tensor()
        mark("warmup_wait")
        _gin_wait_signal[(1, )](dev_comm, int(signal_win), 0, num_warps=8)
        torch.cuda.synchronize()
        mark("launch_put")
        _gin_alltoall_put_i32[(world_size, )](
            dev_comm,
            int(send_win),
            int(recv_win),
            int(signal_win),
            rank,
            args.nelems,
            num_warps=8,
        )
        mark("sync_put")
        torch.cuda.synchronize()
        if not args.skip_wait:
            mark("launch_wait")
            _gin_wait_signal[(1, )](dev_comm, int(signal_win), world_size, num_warps=8)
        mark("sync")
        torch.cuda.synchronize()
        mark("check")
        if rank == 0:
            print(f"[rank {rank}] signal_after_sync={int(signal.item())}", flush=True)

        got = recv.cpu()
        expected = torch.empty(got.shape, dtype=got.dtype)
        for src_rank in range(world_size):
            expected[src_rank].fill_(src_rank * 1000 + rank)
        if not torch.equal(got, expected):
            print(f"rank {rank} recv={got} expected={expected}")
            raise AssertionError("NCCL GIN all-to-all smoke test failed")
        if rank == 0:
            print(f"NCCL GIN all-to-all smoke test passed: world_size={world_size}, nelems={args.nelems}")
    finally:
        for win in [signal_win, recv_win, send_win]:
            if win is not None:
                win.close()
        finalize_distributed()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
