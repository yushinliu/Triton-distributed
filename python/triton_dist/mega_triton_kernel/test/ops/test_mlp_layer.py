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
import torch
import os
import argparse
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.profiler_utils import get_torch_prof_ctx
from triton_dist.mega_triton_kernel.test.torch_impl_utils import (
    torch_gate_silu_mul_up, )
import triton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--intra_kernel_profile", default=False, action="store_true",
                        help="enable intra kernel profiling")
    parser.add_argument("--enable_runtime_scheduler", default=False, action="store_true",
                        help="enable runtime scheduler")
    parser.add_argument("--fuse_fc1_silu", default=False, action="store_true",
                        help="fuse fc1 and silu_mul_up into one mega task")
    parser.add_argument("--bench_warmup", type=int, default=20, help="benchmark warmup iterations")
    parser.add_argument("--bench_iters", type=int, default=0, help="benchmark iterations")

    return parser.parse_args()


def randomize_input_and_flush_l2(l2_cache, mlp_layer_input, dtype):
    l2_cache.zero_()
    tmp_input = torch.randn(mlp_layer_input.shape, dtype=dtype).cuda()
    mlp_layer_input.copy_(tmp_input)


def benchmark(builder, l2_cache, mlp_layer_input, dtype, warmup, iters):
    for _ in range(warmup):
        randomize_input_and_flush_l2(l2_cache, mlp_layer_input, dtype)
        builder.run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        randomize_input_and_flush_l2(l2_cache, mlp_layer_input, dtype)
        start_event.record()
        builder.run()
        end_event.record()
        end_event.synchronize()
        elapsed_ms += start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / iters
    print(f"builder_avg_ms={avg_ms:.4f}")
    return avg_ms


def get_tol(dtype, output_name):
    if dtype == torch.bfloat16:
        if output_name == "fc2":
            return {"atol": 4.0, "rtol": 0.05}
        return {"atol": 0.0625, "rtol": 0.04}
    return {"atol": 0, "rtol": 0}


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(0)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    l2_cache = torch.randn((256, 1024, 1024)).cuda()
    builder = ModelBuilder(enable_profiling=args.intra_kernel_profile,
                           enable_runtime_scheduler=args.enable_runtime_scheduler,
                           enable_mlp_fc1_silu_fusion=args.fuse_fc1_silu)
    batch = 1
    seq_len = 1
    PAGE_SIZE = 1
    MAX_SEQ_LEN = 32 * 1024  # 32k
    MAX_NUM_KV_BLOCKS = 128 * 1024
    dtype = torch.bfloat16
    tp_size = 8
    hidden_size = 5120
    intermediate_size = 25600 // tp_size
    rope_theta = 1000000

    # weight
    fc1_weight = torch.randn((intermediate_size * 2, hidden_size), dtype=dtype).cuda() / 10
    fc2_weight = torch.randn((hidden_size, intermediate_size), dtype=dtype).cuda() / 10
    # mlp
    mlp_layer_input = torch.randn((batch * seq_len, hidden_size), dtype=dtype, device=torch.cuda.current_device())
    fc1_output = torch.zeros((batch * seq_len, intermediate_size * 2), dtype=dtype).cuda()
    act_out = torch.zeros((batch * seq_len, intermediate_size), dtype=dtype).cuda()
    fc2_out = torch.zeros((batch * seq_len, hidden_size), dtype=dtype).cuda()

    if args.fuse_fc1_silu:
        builder.make_fused_fc1_silu_mul_up(mlp_layer_input, fc1_weight, act_out)
    else:
        builder.make_fc1(mlp_layer_input, fc1_weight, fc1_output)
        builder.make_silu_mul_up(fc1_output, act_out)
    builder.make_fc2(act_out, fc2_weight, fc2_out)
    builder.compile()

    ctx = get_torch_prof_ctx(args.profile)

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    with ctx:
        for i in range(30):
            randomize_input_and_flush_l2(l2_cache, mlp_layer_input, dtype)
            builder.run()

            # torch impl
            l2_cache.zero_()
            fc1_output_ref = torch.nn.functional.linear(mlp_layer_input, fc1_weight)
            act_out_ref = torch_gate_silu_mul_up(fc1_output_ref)
            fc2_output_ref = torch.nn.functional.linear(act_out_ref, fc2_weight)
            if not args.fuse_fc1_silu:
                torch.testing.assert_close(fc1_output_ref, fc1_output, **get_tol(dtype, "fc1"))
            torch.testing.assert_close(act_out_ref, act_out, **get_tol(dtype, "act"))
            torch.testing.assert_close(fc2_output_ref, fc2_out, **get_tol(dtype, "fc2"))

    if args.bench_iters > 0:
        benchmark(builder, l2_cache, mlp_layer_input, dtype, args.bench_warmup, args.bench_iters)

    if args.intra_kernel_profile:
        builder.dump_trace()

    if args.profile:
        import os
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/mlp_layer.json.gz")
