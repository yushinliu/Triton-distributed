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
import triton.language as tl
import triton_dist
from triton_dist.language.extra.language_extra import tid, __syncthreads
from .task_context import get_tensor_data_ptr, get_tensor_size, release_tile
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.cuda.language_extra import (st_v4_b32, multimem_ld_reduce_v4)
from triton.language.extra.cuda.utils import num_warps


@triton_dist.jit
def allreduce_one_shot_multimem_intra_node_kernel(pid, num_pid, symm_in_ptr, out_ptr, elems):
    symm_in_ptr = tl.cast(symm_in_ptr, out_ptr.dtype)

    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_in_ptr)
    VEC_SIZE = 128 // tl.constexpr(symm_in_ptr.dtype.element_ty.primitive_bitwidth)

    thread_idx = tid(axis=0)
    block_dim = num_warps() * 32
    for idx in range(thread_idx + block_dim * pid, elems // VEC_SIZE, num_pid * block_dim):
        val0, val1, val2, val3 = multimem_ld_reduce_v4(data_mc_ptr + idx * VEC_SIZE, acc_dtype=tl.float32)
        st_v4_b32(out_ptr + idx * VEC_SIZE, val0, val1, val2, val3)
    __syncthreads()


@triton_dist.jit
def allreduce_task_compute(
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
    input_ptr = get_tensor_data_ptr(io_tensors_ptr, 0, tl.bfloat16, MAX_NUM_TENSOR_DIMS)
    output_ptr = get_tensor_data_ptr(io_tensors_ptr, 1, tl.bfloat16, MAX_NUM_TENSOR_DIMS)

    n_elements = get_tensor_size(io_tensors_ptr, 1, 0, MAX_NUM_TENSOR_DIMS)
    tile_id = tile_id_or_start
    num_pid = tl.cdiv(n_elements, BLOCK_SIZE)
    allreduce_one_shot_multimem_intra_node_kernel(tile_id, num_pid, input_ptr, output_ptr, n_elements)
    release_tile(scoreboard_ptr, layer_id, task_id, tile_id, MAX_TASK_ID, MAX_NUM_TILES_PER_OP)


@triton_dist.jit
def allreduce_one_shot_nvshmem_remote_write_intra_node_kernel(pid, symm_in_ptr, symm_scratch_ptr, out_ptr, elems,
                                                              BLOCK_SIZE: tl.constexpr, NUM_LOCAL_PES: tl.constexpr):
    symm_in_ptr = tl.cast(symm_in_ptr, out_ptr.dtype)
    symm_scratch_ptr = tl.cast(symm_scratch_ptr, out_ptr.dtype)

    tile_start = pid * BLOCK_SIZE
    if tile_start >= elems:
        return

    valid_elems = tl.minimum(BLOCK_SIZE, elems - tile_start)
    elem_size = tl.constexpr(out_ptr.dtype.element_ty.primitive_bitwidth) // 8
    input_tile_ptr = symm_in_ptr + tile_start
    scratch_tile_ptr = symm_scratch_ptr + pid * BLOCK_SIZE * NUM_LOCAL_PES
    local_pe = libshmem_device.team_my_pe(libshmem_device.NVSHMEMX_TEAM_NODE)
    local_slot_ptr = scratch_tile_ptr + local_pe * BLOCK_SIZE

    __syncthreads()

    thread_idx = tid(axis=0)
    block_dim = num_warps() * 32
    # Remote-write / local-read protocol:
    # 1. Copy the local tile into the local scratch slot.
    # 2. Push the same tile to every peer's scratch slot assigned to this PE.
    # 3. Synchronize across the node team, then reduce from the local scratch buffer.
    for off in range(0, BLOCK_SIZE, block_dim):
        idx = thread_idx + off
        if idx < valid_elems:
            tl.store(local_slot_ptr + idx, tl.load(input_tile_ptr + idx))
    __syncthreads()

    nbytes = valid_elems * elem_size
    for pe in tl.static_range(0, NUM_LOCAL_PES):
        if pe != local_pe:
            libshmem_device.putmem_block(local_slot_ptr, input_tile_ptr, nbytes, pe)
    libshmem_device.barrier_block(libshmem_device.NVSHMEMX_TEAM_NODE)
    __syncthreads()

    for off in range(0, BLOCK_SIZE, block_dim):
        idx = thread_idx + off
        if idx < valid_elems:
            acc = tl.zeros((), dtype=tl.float32)
            for pe in tl.static_range(0, NUM_LOCAL_PES):
                acc += tl.load(scratch_tile_ptr + pe * BLOCK_SIZE + idx).to(tl.float32)
            tl.store(out_ptr + tile_start + idx, acc.to(out_ptr.dtype.element_ty))
    __syncthreads()


@triton_dist.jit
def allreduce_nvshmem_task_compute(
    io_tensors_ptr,
    layer_id,
    task_id,
    tile_id_or_start,
    scoreboard_ptr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_LOCAL_PES: tl.constexpr,
):
    input_ptr = get_tensor_data_ptr(io_tensors_ptr, 0, tl.bfloat16, MAX_NUM_TENSOR_DIMS)
    scratch_ptr = get_tensor_data_ptr(io_tensors_ptr, 1, tl.bfloat16, MAX_NUM_TENSOR_DIMS)
    output_ptr = get_tensor_data_ptr(io_tensors_ptr, 2, tl.bfloat16, MAX_NUM_TENSOR_DIMS)

    n_elements = get_tensor_size(io_tensors_ptr, 2, 0, MAX_NUM_TENSOR_DIMS)
    tile_id = tile_id_or_start
    allreduce_one_shot_nvshmem_remote_write_intra_node_kernel(tile_id, input_ptr, scratch_ptr, output_ptr, n_elements,
                                                              BLOCK_SIZE, NUM_LOCAL_PES)
    release_tile(scoreboard_ptr, layer_id, task_id, tile_id, MAX_TASK_ID, MAX_NUM_TILES_PER_OP)
