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
import triton
import triton.language as tl
from .task_context import TaskBaseInfo, Scoreboard, TensorDesc
from .linear import tile_wise_matmul_compute


@triton.jit
def fc1_task_compute(task_base_info: TaskBaseInfo, scoreboard: Scoreboard, BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, NUM_STAGES: tl.constexpr):

    input: TensorDesc = task_base_info.get_tensor(0)
    weight: TensorDesc = task_base_info.get_tensor(1)
    output: TensorDesc = task_base_info.get_tensor(2)

    M = input.size(0)
    K = input.size(1, 16)
    N = weight.size(0)

    a_ptr = input.data_ptr(tl.bfloat16)
    b_ptr = weight.data_ptr(tl.bfloat16)
    c_ptr = output.data_ptr(tl.bfloat16)

    tile_id = task_base_info.tile_id_or_start
    tile_wise_matmul_compute(tile_id, a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                             NUM_STAGES)
    scoreboard.release_tile(task_base_info, tile_id)


@triton.jit
def mlp_fc1_silu_mul_up_task_compute(task_base_info: TaskBaseInfo, scoreboard: Scoreboard, BLOCK_SIZE_M: tl.constexpr,
                                     BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                                     NUM_STAGES: tl.constexpr):
    input: TensorDesc = task_base_info.get_tensor(0)
    weight: TensorDesc = task_base_info.get_tensor(1)
    output: TensorDesc = task_base_info.get_tensor(2)

    M = input.size(0)
    K = input.size(1, 16)
    N = output.size(1)

    a_ptr = input.data_ptr(tl.bfloat16)
    w_ptr = weight.data_ptr(tl.bfloat16)
    out_ptr = output.data_ptr(tl.bfloat16)

    tile_id = task_base_info.tile_id_or_start
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for ki in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        gate_w_ptrs = w_ptr + (offs_bn[:, None] * K + offs_k[None, :])
        up_w_ptrs = w_ptr + ((offs_bn + N)[:, None] * K + offs_k[None, :])

        mask = offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        gate_w = tl.load(gate_w_ptrs, mask=mask, other=0.0)
        up_w = tl.load(up_w_ptrs, mask=mask, other=0.0)
        gate_acc = tl.dot(a, gate_w.T, gate_acc)
        up_acc = tl.dot(a, up_w.T, up_acc)

    gate = gate_acc.to(tl.bfloat16)
    up = up_acc.to(tl.bfloat16)
    gate_fp32 = gate.to(tl.float32)
    gate_fp32 = gate_fp32 * (1.0 / (1.0 + tl.exp(-gate_fp32)))
    gate = gate_fp32.to(tl.bfloat16)
    out = (gate * up).to(out_ptr.dtype.element_ty)

    offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)
    scoreboard.release_tile(task_base_info, tile_id)


@triton.jit
def rms_norm_mlp_fc1_silu_mul_up_task_compute(task_base_info: TaskBaseInfo, scoreboard: Scoreboard,
                                              BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                              BLOCK_SIZE_K: tl.constexpr, NUM_STAGES: tl.constexpr,
                                              RMS_EPS: tl.constexpr):
    input: TensorDesc = task_base_info.get_tensor(0)
    rms_weight: TensorDesc = task_base_info.get_tensor(1)
    weight: TensorDesc = task_base_info.get_tensor(2)
    output: TensorDesc = task_base_info.get_tensor(3)

    M = input.size(0)
    K = input.size(1, 16)
    N = output.size(1)

    a_ptr = input.data_ptr(tl.bfloat16)
    rms_w_ptr = rms_weight.data_ptr(tl.bfloat16)
    w_ptr = weight.data_ptr(tl.bfloat16)
    out_ptr = output.data_ptr(tl.bfloat16)

    tile_id = task_base_info.tile_id_or_start
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    row_mask = offs_am < M
    offs_am_safe = tl.where(row_mask, offs_am, 0)
    offs_bn_safe = tl.where(offs_bn < N, offs_bn, 0)
    offs_am_safe = tl.max_contiguous(tl.multiple_of(offs_am_safe, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn_safe = tl.max_contiguous(tl.multiple_of(offs_bn_safe, BLOCK_SIZE_N), BLOCK_SIZE_N)

    square_sum = tl.zeros((BLOCK_SIZE_M, ), dtype=tl.float32)
    for ki in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k_for_mask < K - ki * BLOCK_SIZE_K
        a_ptrs = a_ptr + (offs_am_safe[:, None] * K + offs_k[None, :])
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        square_sum += tl.sum(a * a, axis=1)
    row_rms = tl.rsqrt(square_sum / K + RMS_EPS)

    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for ki in tl.range(0, k_tiles, num_stages=NUM_STAGES):
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k_for_mask < K - ki * BLOCK_SIZE_K
        a_ptrs = a_ptr + (offs_am_safe[:, None] * K + offs_k[None, :])
        rms_w_ptrs = rms_w_ptr + offs_k
        gate_w_ptrs = w_ptr + (offs_bn_safe[:, None] * K + offs_k[None, :])
        up_w_ptrs = w_ptr + ((offs_bn_safe + N)[:, None] * K + offs_k[None, :])

        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        rms_w = tl.load(rms_w_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        a_norm = (a * row_rms[:, None] * rms_w[None, :]).to(tl.bfloat16)
        gate_w = tl.load(gate_w_ptrs, mask=k_mask[None, :], other=0.0)
        up_w = tl.load(up_w_ptrs, mask=k_mask[None, :], other=0.0)
        gate_acc = tl.dot(a_norm, gate_w.T, gate_acc)
        up_acc = tl.dot(a_norm, up_w.T, up_acc)

    gate = gate_acc.to(tl.bfloat16)
    up = up_acc.to(tl.bfloat16)
    gate_fp32 = gate.to(tl.float32)
    gate_fp32 = gate_fp32 * (1.0 / (1.0 + tl.exp(-gate_fp32)))
    gate = gate_fp32.to(tl.bfloat16)
    out = (gate * up).to(out_ptr.dtype.element_ty)

    offs_cm = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = start_n + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + N * offs_cm[:, None] + offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)
    scoreboard.release_tile(task_base_info, tile_id)
