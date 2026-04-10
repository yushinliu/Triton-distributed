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
from triton_dist.mega_triton_kernel import ModelBuilder
import torch
import argparse
import itertools
from triton_dist.profiler_utils import get_torch_prof_ctx
from triton_dist.mega_triton_kernel.test.torch_impl_utils import torch_all_reduce
import os
from triton_dist.utils import (
    initialize_distributed,
    nvshmem_create_tensor,
    finalize_distributed,
    nvshmem_free_tensor_sync,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--allreduce_impl", type=str, default="multimem", choices=["multimem", "nvshmem"])
    parser.add_argument("--iters", type=int, default=30, help="number of correctness iterations")
    parser.add_argument("--verify_hang", type=int, default=0, help="extra stress iterations without correctness checks")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=5120)
    parser.add_argument("--intermediate_size", type=int, default=25600)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed + RANK)

    builder = ModelBuilder(rank=RANK, world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE,
                           allreduce_implementation=args.allreduce_impl)
    batch = args.batch
    seq_len = args.seq_len
    hidden_size = args.hidden_size
    intermidiate_size = args.intermediate_size
    dtype = torch.bfloat16

    x = nvshmem_create_tensor((batch * seq_len, hidden_size), dtype=dtype)
    out = torch.zeros((batch * seq_len, hidden_size), dtype=dtype, device=torch.cuda.current_device())
    gemm_weight = torch.rand((intermidiate_size, hidden_size), dtype=dtype, device=torch.cuda.current_device())
    gemm_out = torch.empty((batch * seq_len, gemm_weight.shape[0]), dtype=dtype, device=torch.cuda.current_device())
    builder.make_allreduce(x, out, double_input_buffer=True, implementation=args.allreduce_impl)
    builder.make_linear(out, gemm_weight, gemm_out)

    builder.compile()

    ctx = get_torch_prof_ctx(args.profile)
    with ctx:
        num_inputs = max(args.iters, 1 if args.verify_hang > 0 else 0)
        inputs = [torch.rand(x.shape, dtype=dtype).cuda() for i in range(num_inputs)]
        mega_outs = []
        torch_outs = []

        # mega impl
        for iter_idx, tmp_input in enumerate(inputs[:args.iters]):
            x.copy_(tmp_input)
            builder.run()
            mega_outs.append(gemm_out.clone())
            if RANK == 0:
                print(f"mega correctness iter {iter_idx + 1}/{args.iters} done")

        torch.cuda.synchronize()
        # torch impl
        for tmp_input in inputs[:args.iters]:
            out_ref = torch_all_reduce(tmp_input, pg=TP_GROUP)
            gemm_out_ref = torch.nn.functional.linear(out_ref, gemm_weight)
            torch_outs.append(gemm_out_ref)

        # verify
        for idx, (mega_out, torch_out) in enumerate(zip(mega_outs, torch_outs)):
            torch.testing.assert_close(torch_out, mega_out, atol=3e-2, rtol=3e-2)

        for iter_idx, tmp_input in enumerate(itertools.islice(itertools.cycle(inputs), args.verify_hang)):
            x.copy_(tmp_input)
            builder.run()
            if RANK == 0:
                print(f"hang verify iter {iter_idx + 1}/{args.verify_hang} done")

    if args.profile:
        import os
        prof_dir = "prof/AR/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/rank_{RANK}.json.gz")
    builder.finalize()
    nvshmem_free_tensor_sync(x)
    finalize_distributed()
