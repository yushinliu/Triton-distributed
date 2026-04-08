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
from triton_dist.language.extra.language_extra import tid, st, __syncthreads, ld
from triton.language.extra.cuda.utils import num_warps


@triton.jit
def get_tensor_desc_ptr(io_tensors_ptr, idx, MAX_NUM_TENSOR_DIMS: tl.constexpr):
    INT_PER_TENSOR: tl.constexpr = MAX_NUM_TENSOR_DIMS + 2
    return io_tensors_ptr + idx * INT_PER_TENSOR


@triton.jit
def get_tensor_data_ptr(io_tensors_ptr, idx, dtype, MAX_NUM_TENSOR_DIMS: tl.constexpr):
    buf_ptr = get_tensor_desc_ptr(io_tensors_ptr, idx, MAX_NUM_TENSOR_DIMS).to(tl.pointer_type(tl.uint64))
    data_ptr = tl.load(buf_ptr).to(tl.pointer_type(dtype))
    data_ptr = tl.multiple_of(data_ptr, 16)
    return data_ptr


@triton.jit
def get_tensor_size(io_tensors_ptr, idx, dim, MAX_NUM_TENSOR_DIMS: tl.constexpr, multiple: tl.constexpr = 1):
    int_per_data_ptr: tl.constexpr = 2
    dim_ptr = get_tensor_desc_ptr(io_tensors_ptr, idx, MAX_NUM_TENSOR_DIMS) + dim + int_per_data_ptr
    value = tl.load(dim_ptr)
    value = tl.multiple_of(value, multiple)
    return value.to(tl.int32)


@triton.jit
def get_extra_params_ptr(io_tensors_ptr, num_io_tensors, MAX_NUM_TENSOR_DIMS: tl.constexpr):
    INT_PER_TENSOR: tl.constexpr = MAX_NUM_TENSOR_DIMS + 2
    return io_tensors_ptr + num_io_tensors * INT_PER_TENSOR


@triton.jit
def get_task_scoreboard_start(scoreboard_ptr, layer_id, task_id, MAX_TASK_ID: tl.constexpr,
                              MAX_NUM_TILES_PER_OP: tl.constexpr):
    sb_layer_offset = layer_id * MAX_TASK_ID * MAX_NUM_TILES_PER_OP
    return scoreboard_ptr + sb_layer_offset + task_id * MAX_NUM_TILES_PER_OP


@triton.jit
def wait_deps(task_deps_ptr, INT_PER_DEPS: tl.constexpr, scoreboard_ptr, depend_entry_start, depend_entry_end,
              TILE_READY_SIGNAL: tl.constexpr = 1):
    lane_id = tid(0) % 32
    warp_id = tid(0) // 32
    for t in range(depend_entry_start + warp_id, depend_entry_end, num_warps()):
        l = ld(task_deps_ptr + t * INT_PER_DEPS + 0)
        r = ld(task_deps_ptr + t * INT_PER_DEPS + 1)
        num_signals = r - l
        sb_wait_base_ptr = scoreboard_ptr + l
        for i in range(lane_id, num_signals, 32):
            while ld(sb_wait_base_ptr + i, scope="gpu", semantic="acquire") != TILE_READY_SIGNAL:
                pass
    __syncthreads()


@triton.jit
def release_tile(scoreboard_ptr, layer_id, task_id, tile_id, MAX_TASK_ID: tl.constexpr,
                 MAX_NUM_TILES_PER_OP: tl.constexpr, TILE_READY_SIGNAL: tl.constexpr = 1):
    sb_set_base_ptr = get_task_scoreboard_start(scoreboard_ptr, layer_id, task_id, MAX_TASK_ID, MAX_NUM_TILES_PER_OP)
    __syncthreads()
    if tid(0) == 0:
        st(sb_set_base_ptr + tile_id, TILE_READY_SIGNAL, "gpu", "release")
    __syncthreads()
