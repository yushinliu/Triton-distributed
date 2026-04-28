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
from collections import Counter

import torch
import torch.distributed as dist

from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.profiler_utils import get_torch_prof_ctx
from triton_dist.utils import finalize_distributed, initialize_distributed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--allreduce_impl", type=str, default="nvshmem", choices=["multimem", "nvshmem"])
    parser.add_argument("--input_mode", type=str, default="direct", choices=["direct", "producer"])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--num_tokens", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_atol", type=float, default=5e-2)
    parser.add_argument("--check_rtol", type=float, default=5e-2)
    return parser.parse_args()


def torch_all_reduce(local_input: torch.Tensor, pg: torch.distributed.ProcessGroup):
    output = torch.clone(local_input)
    dist.all_reduce(output, group=pg)
    return output


def check_args(args, world_size, local_world_size):
    if world_size <= 1:
        raise ValueError("Mega allreduce standalone test requires at least 2 ranks")
    if world_size != local_world_size:
        raise ValueError(
            "Mega allreduce standalone test only supports intra-node runs, "
            f"got WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
        )
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    if args.num_tokens <= 0:
        raise ValueError(f"--num_tokens must be positive, got {args.num_tokens}")
    if args.hidden_size <= 0:
        raise ValueError(f"--hidden_size must be positive, got {args.hidden_size}")
    if args.check_atol < 0:
        raise ValueError(f"--check_atol must be non-negative, got {args.check_atol}")
    if args.check_rtol < 0:
        raise ValueError(f"--check_rtol must be non-negative, got {args.check_rtol}")


def assert_uses_nvshmem_allreduce(builder):
    task_counts = Counter(type(task).__name__ for task in builder.megakernel_tasks)
    expected = ("AllReduceNVSHMEMTask", "AllReduceNVSHMEMPushTask")
    missing = [task_name for task_name in expected if task_counts[task_name] == 0]
    if missing:
        raise AssertionError(f"missing NVSHMEM allreduce tasks: {missing}")
    if task_counts["AllReduceNVSHMEMTask"] != task_counts["AllReduceNVSHMEMPushTask"]:
        raise AssertionError(
            "NVSHMEM allreduce task count mismatch: "
            f"reduce={task_counts['AllReduceNVSHMEMTask']}, "
            f"push={task_counts['AllReduceNVSHMEMPushTask']}"
        )


if __name__ == "__main__":
    args = parse_args()

    TP_GROUP = initialize_distributed(seed=args.seed)
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    check_args(args, WORLD_SIZE, LOCAL_WORLD_SIZE)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed + RANK)

    dtype = torch.bfloat16
    shape = (args.num_tokens, args.hidden_size)

    builder = None
    ctx = get_torch_prof_ctx(args.profile)
    try:
        builder = ModelBuilder(rank=RANK, world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE)
        x = builder.create_symm_tensor(shape, dtype=dtype)
        out = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())

        if args.input_mode == "producer":
            lhs = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            rhs = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            builder.make_add(lhs, rhs, x)

        builder.make_allreduce(x, out, double_input_buffer=False, implementation=args.allreduce_impl)
        if args.allreduce_impl == "nvshmem":
            assert_uses_nvshmem_allreduce(builder)
        builder.compile()

        with ctx:
            for iter_idx in range(args.iters):
                if args.input_mode == "producer":
                    lhs_input = torch.randn(shape, dtype=dtype, device=torch.cuda.current_device())
                    rhs_input = torch.randn(shape, dtype=dtype, device=torch.cuda.current_device())
                    lhs.copy_(lhs_input)
                    rhs.copy_(rhs_input)
                    local_input = lhs_input + rhs_input
                else:
                    local_input = torch.rand(shape, dtype=dtype, device=torch.cuda.current_device())
                    x.copy_(local_input)

                builder.run()
                torch.cuda.synchronize()
                mega_out = out.clone()

                torch_out = torch_all_reduce(local_input, pg=TP_GROUP)
                try:
                    torch.testing.assert_close(torch_out, mega_out, atol=args.check_atol, rtol=args.check_rtol)
                except Exception as e:
                    print(f"RANK = {RANK}, iteration {iter_idx} failed with {e}")
                    raise

        if RANK == 0:
            print(
                f"Mega allreduce standalone {args.allreduce_impl} passed: "
                f"world_size={WORLD_SIZE}, shape={shape}, iters={args.iters}, "
                f"input_mode={args.input_mode}, check_atol={args.check_atol}, check_rtol={args.check_rtol}"
            )

        if args.profile:
            prof_dir = "prof/AR/"
            os.makedirs(prof_dir, exist_ok=True)
            ctx.export_chrome_trace(f"{prof_dir}/standalone_rank_{RANK}.json.gz")
    finally:
        if builder is not None:
            builder.finalize()
        finalize_distributed()
