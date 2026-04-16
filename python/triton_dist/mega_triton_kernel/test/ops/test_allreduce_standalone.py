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
import argparse
import os

import torch
import torch.distributed as dist

from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.profiler_utils import get_torch_prof_ctx
from triton_dist.utils import (
    finalize_distributed,
    initialize_distributed,
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--allreduce_impl", type=str, default="nvshmem", choices=["multimem", "nvshmem"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--num_tokens", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def torch_all_reduce(local_input: torch.Tensor, pg: torch.distributed.ProcessGroup):
    output = torch.clone(local_input)
    dist.all_reduce(output, group=pg)
    return output


def check_args(args):
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    if args.num_tokens <= 0:
        raise ValueError(f"--num_tokens must be positive, got {args.num_tokens}")
    if args.hidden_size <= 0:
        raise ValueError(f"--hidden_size must be positive, got {args.hidden_size}")


if __name__ == "__main__":
    args = parse_args()
    check_args(args)

    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    if WORLD_SIZE <= 1:
        raise ValueError("Mega allreduce standalone test requires at least 2 ranks")
    if WORLD_SIZE != LOCAL_WORLD_SIZE:
        raise ValueError(
            "Mega allreduce standalone test only supports intra-node runs, "
            f"got WORLD_SIZE={WORLD_SIZE}, LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE}"
        )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed + RANK)

    dtype = torch.bfloat16
    shape = (args.num_tokens, args.hidden_size)

    builder = ModelBuilder(rank=RANK, world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE)
    x = nvshmem_create_tensor(shape, dtype=dtype)
    out = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())

    builder.make_allreduce(x, out, double_input_buffer=False, implementation=args.allreduce_impl)
    builder.compile()

    ctx = get_torch_prof_ctx(args.profile)
    with ctx:
        for iter_idx in range(args.iters):
            local_input = torch.rand(shape, dtype=dtype, device=torch.cuda.current_device())
            x.copy_(local_input)
            builder.run()
            mega_out = out.clone()

            torch_out = torch_all_reduce(local_input, pg=TP_GROUP)
            try:
                torch.testing.assert_close(torch_out, mega_out, atol=3e-2, rtol=3e-2)
            except Exception as e:
                print(f"RANK = {RANK}, iteration {iter_idx} failed with {e}")
                raise

    if RANK == 0:
        print(
            f"Mega allreduce standalone {args.allreduce_impl} passed: "
            f"world_size={WORLD_SIZE}, shape={shape}, iters={args.iters}"
        )

    if args.profile:
        prof_dir = "prof/AR/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/standalone_rank_{RANK}.json.gz")

    builder.finalize()
    nvshmem_free_tensor_sync(x)
    finalize_distributed()
