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
import triton

from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.mega_triton_kernel.test.torch_impl_utils import (
    ref_paged_attn,
    rmsnorm_ref,
    torch_gate_silu_mul_up,
)
from triton_dist.profiler_utils import get_torch_prof_ctx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--fuse_o_proj_add", default=False, action="store_true",
                        help="fuse o_proj and residual add into one mega task")
    parser.add_argument("--fuse_post_rms_fc1_silu", default=False, action="store_true",
                        help="fuse post rms_norm, fc1 and silu into one mega task")
    parser.add_argument("--bench_warmup", type=int, default=20, help="benchmark warmup iterations")
    parser.add_argument("--bench_iters", type=int, default=0, help="benchmark iterations")
    return parser.parse_args()


def randomize_inputs_and_flush_l2(l2_cache, q_norm_rope, residual_input, dtype):
    l2_cache.zero_()
    q_norm_rope.copy_(torch.randn_like(q_norm_rope, dtype=dtype))
    residual_input.copy_(torch.randn_like(residual_input, dtype=dtype))


def benchmark(builder, l2_cache, q_norm_rope, residual_input, dtype, warmup, iters):
    for _ in range(warmup):
        randomize_inputs_and_flush_l2(l2_cache, q_norm_rope, residual_input, dtype)
        builder.run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0
    for _ in range(iters):
        randomize_inputs_and_flush_l2(l2_cache, q_norm_rope, residual_input, dtype)
        start_event.record()
        builder.run()
        end_event.record()
        end_event.synchronize()
        elapsed_ms += start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / iters
    print(f"builder_avg_ms={avg_ms:.4f}")
    return avg_ms


def get_tol(dtype, output_name):
    if dtype != torch.bfloat16:
        return {"atol": 0.0, "rtol": 0.0}
    tol = {
        "attn": {"atol": 2e-2, "rtol": 1e-2},
        "o_proj": {"atol": 0.125, "rtol": 0.02},
        "residual": {"atol": 0.125, "rtol": 0.02},
        "post_norm": {"atol": 0.125, "rtol": 0.03},
        "fc1": {"atol": 0.25, "rtol": 0.03},
        "act": {"atol": 1.5, "rtol": 0.05},
    }
    return tol[output_name]


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(0)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    l2_cache = torch.randn((256, 1024, 1024), device="cuda")
    builder = ModelBuilder()
    batch = 1
    seq_len = 1
    page_size = 1
    max_seq_len = 4096
    max_num_kv_blocks = max_seq_len
    max_num_blocks_per_seq = max_seq_len
    dtype = torch.bfloat16
    tp_size = 8
    hidden_size = 5120
    num_q_heads = 40 // tp_size
    num_kv_heads = 8 // tp_size
    head_dim = 128
    intermediate_size = 25600 // tp_size
    rms_eps = 1e-6

    sm_scale = head_dim**-0.5
    soft_cap = 0.0
    kv_lens = torch.tensor([2048], dtype=torch.int32, device="cuda")
    block_tables = torch.arange(max_num_blocks_per_seq, dtype=torch.int32, device="cuda").unsqueeze(0)

    q_norm_rope = torch.randn((batch, seq_len, num_q_heads, head_dim), dtype=dtype, device="cuda")
    key_cache = torch.randn((max_num_kv_blocks, page_size, num_kv_heads, head_dim), dtype=dtype, device="cuda")
    value_cache = torch.randn((max_num_kv_blocks, page_size, num_kv_heads, head_dim), dtype=dtype, device="cuda")
    attn_out = torch.empty_like(q_norm_rope)

    residual_input = torch.randn((batch * seq_len, hidden_size), dtype=dtype, device="cuda")
    o_proj_weight = torch.randn((hidden_size, num_q_heads * head_dim), dtype=dtype, device="cuda") / 10
    o_proj_out = torch.empty((batch * seq_len, hidden_size), dtype=dtype, device="cuda")
    attn_residual_out = torch.empty((batch * seq_len, hidden_size), dtype=dtype, device="cuda")

    post_norm_weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    post_norm_out = torch.empty_like(attn_residual_out)

    fc1_weight = torch.randn((intermediate_size * 2, hidden_size), dtype=dtype, device="cuda") / 10
    fc1_out = torch.empty((batch * seq_len, intermediate_size * 2), dtype=dtype, device="cuda")
    act_out = torch.empty((batch * seq_len, intermediate_size), dtype=dtype, device="cuda")

    builder.make_flash_decode(q_norm_rope, key_cache, value_cache, block_tables, kv_lens, attn_out, sm_scale, soft_cap)
    attn_out_2d = attn_out.reshape(batch * seq_len, num_q_heads * head_dim)
    if args.fuse_o_proj_add:
        builder.make_o_proj_add(attn_out_2d, o_proj_weight, residual_input, attn_residual_out)
    else:
        builder.make_o_proj(attn_out_2d, o_proj_weight, o_proj_out)
        builder.make_add(residual_input, o_proj_out, attn_residual_out)
    if args.fuse_post_rms_fc1_silu:
        builder.make_fused_rms_norm_fc1_silu_mul_up(attn_residual_out, post_norm_weight, fc1_weight, act_out, rms_eps)
    else:
        builder.make_rms_norm(attn_residual_out, post_norm_weight, post_norm_out, rms_eps)
        builder.make_fc1(post_norm_out, fc1_weight, fc1_out)
        builder.make_silu_mul_up(fc1_out, act_out)
    builder.compile()

    ctx = get_torch_prof_ctx(args.profile)

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    with ctx:
        for _ in range(30):
            randomize_inputs_and_flush_l2(l2_cache, q_norm_rope, residual_input, dtype)
            builder.run()

            l2_cache.zero_()
            attn_out_ref = ref_paged_attn(query=q_norm_rope.reshape(batch, num_q_heads, head_dim),
                                          key_cache=key_cache, value_cache=value_cache, query_lens=[1] * batch,
                                          kv_lens=kv_lens, block_tables=block_tables, scale=sm_scale,
                                          soft_cap=soft_cap).reshape_as(attn_out)
            o_proj_out_ref = torch.nn.functional.linear(attn_out_ref.reshape(-1, num_q_heads * head_dim), o_proj_weight)
            attn_residual_out_ref = o_proj_out_ref + residual_input
            post_norm_out_ref = rmsnorm_ref(attn_residual_out_ref, post_norm_weight, rms_eps)
            fc1_out_ref = torch.nn.functional.linear(post_norm_out_ref, fc1_weight)
            act_out_ref = torch_gate_silu_mul_up(fc1_out_ref)

            torch.testing.assert_close(attn_out_ref, attn_out, **get_tol(dtype, "attn"))
            if not args.fuse_o_proj_add:
                torch.testing.assert_close(o_proj_out_ref, o_proj_out, **get_tol(dtype, "o_proj"))
            torch.testing.assert_close(attn_residual_out_ref, attn_residual_out, **get_tol(dtype, "residual"))
            if not args.fuse_post_rms_fc1_silu:
                torch.testing.assert_close(post_norm_out_ref, post_norm_out, **get_tol(dtype, "post_norm"))
                torch.testing.assert_close(fc1_out_ref, fc1_out, **get_tol(dtype, "fc1"))
            torch.testing.assert_close(act_out_ref, act_out, **get_tol(dtype, "act"))

    if args.bench_iters > 0:
        benchmark(builder, l2_cache, q_norm_rope, residual_input, dtype, args.bench_warmup, args.bench_iters)

    if args.profile:
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/qwen3_decode_subgraph.json.gz")
