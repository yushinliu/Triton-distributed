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
from triton_dist.language.extra.language_extra import tid, __syncthreads, atomic_cas
import triton_dist.language as dl
from .task_context import get_extra_params_ptr, get_tensor_data_ptr, release_tile


@triton.jit
def barrier_all_intra_node_atomic_cas_block(local_rank, local_world_size, symm_flag_ptr):
    """ NOTE: this function should only be called with atomic support. memory over PCI-e does not support atomic r/w. DON'T use this function on such platforms.
    """

    thread_idx = tid(0)
    if thread_idx < local_world_size:  # thread_idx => local_rank
        remote_ptr = dl.symm_at(symm_flag_ptr + local_rank, thread_idx)
        while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
            pass

    if thread_idx < local_world_size:  # thread_idx => local_rank
        while (atomic_cas(symm_flag_ptr + thread_idx, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


@triton.jit
def barrier_all_intra_node_task_compute(
    io_tensors_ptr,
    layer_id,
    task_id,
    tile_id_or_start,
    scoreboard_ptr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
):
    symm_flag_ptr = get_tensor_data_ptr(io_tensors_ptr, 0, tl.int32, MAX_NUM_TENSOR_DIMS)
    extra_params_ptr = get_extra_params_ptr(io_tensors_ptr, 1, MAX_NUM_TENSOR_DIMS)

    local_rank = tl.load(extra_params_ptr + 0).to(tl.int32)
    local_world_size = tl.load(extra_params_ptr + 1).to(tl.int32)
    barrier_all_intra_node_atomic_cas_block(local_rank, local_world_size, symm_flag_ptr)
    release_tile(scoreboard_ptr, layer_id, task_id, 0, MAX_TASK_ID, MAX_NUM_TILES_PER_OP)
