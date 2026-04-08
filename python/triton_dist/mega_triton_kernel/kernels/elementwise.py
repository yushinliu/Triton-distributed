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
from .task_context import get_tensor_data_ptr, get_tensor_size, release_tile


@triton.jit
def add_task_compute(
    io_tensors_ptr,
    layer_id,
    task_id,
    tile_id_or_start,
    scoreboard_ptr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    lhs_ptr = get_tensor_data_ptr(io_tensors_ptr, 0, tl.bfloat16, MAX_NUM_TENSOR_DIMS)
    rhs_ptr = get_tensor_data_ptr(io_tensors_ptr, 1, tl.bfloat16, MAX_NUM_TENSOR_DIMS)
    out_ptr = get_tensor_data_ptr(io_tensors_ptr, 2, tl.bfloat16, MAX_NUM_TENSOR_DIMS)

    n_elements = get_tensor_size(io_tensors_ptr, 2, 0, MAX_NUM_TENSOR_DIMS)
    block_start = tile_id_or_start * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(lhs_ptr + offsets, mask=mask)
    y = tl.load(rhs_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)
    release_tile(scoreboard_ptr, layer_id, task_id, tile_id_or_start, MAX_TASK_ID, MAX_NUM_TILES_PER_OP)
