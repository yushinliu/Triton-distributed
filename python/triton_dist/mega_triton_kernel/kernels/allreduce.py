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
from .task_context import TaskBaseInfo, Scoreboard
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
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    BLOCK_SIZE: tl.constexpr,
):
    input_tensor = task_base_info.get_tensor(0)
    output_tensor = task_base_info.get_tensor(1)

    input_ptr = input_tensor.data_ptr(tl.bfloat16)
    output_ptr = output_tensor.data_ptr(tl.bfloat16)

    n_elements = output_tensor.size(0)
    tile_id = task_base_info.tile_id_or_start
    num_pid = tl.cdiv(n_elements, BLOCK_SIZE)
    allreduce_one_shot_multimem_intra_node_kernel(tile_id, num_pid, input_ptr, output_ptr, n_elements)
    scoreboard.release_tile(task_base_info, task_base_info.tile_id_or_start)


@triton_dist.jit
def allreduce_one_shot_nvshmem_issue_chunk(
    input_tile_ptr,
    scratch_tile_ptr,
    signal_tile_ptr,
    valid_elems,
    local_pe,
    chunk_id,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_LOCAL_PES: tl.constexpr,
):
    chunk_start = chunk_id * CHUNK_SIZE
    if chunk_start >= valid_elems:
        return

    valid_chunk_elems = tl.minimum(CHUNK_SIZE, valid_elems - chunk_start)
    local_chunk_slot_ptr = scratch_tile_ptr + local_pe * BLOCK_SIZE + chunk_start
    signal_chunk_ptr = signal_tile_ptr + chunk_id * NUM_LOCAL_PES
    elem_size = tl.constexpr(input_tile_ptr.dtype.element_ty.primitive_bitwidth) // 8

    thread_idx = tid(axis=0)
    block_dim = num_warps() * 32
    for off in range(0, CHUNK_SIZE, block_dim):
        idx = thread_idx + off
        if idx < valid_chunk_elems:
            tl.store(local_chunk_slot_ptr + idx, tl.load(input_tile_ptr + chunk_start + idx))
    __syncthreads()

    if thread_idx == 0:
        tl.store(signal_chunk_ptr + local_pe, 1)
    __syncthreads()

    nbytes = valid_chunk_elems * elem_size
    for pe in tl.static_range(0, NUM_LOCAL_PES):
        if pe != local_pe:
            libshmem_device.putmem_signal_nbi_block(local_chunk_slot_ptr, local_chunk_slot_ptr, nbytes,
                                                    signal_chunk_ptr + local_pe, 1,
                                                    libshmem_device.NVSHMEM_SIGNAL_SET, pe)


@triton_dist.jit
def allreduce_one_shot_nvshmem_reduce_chunk(
    scratch_tile_ptr,
    signal_tile_ptr,
    out_ptr,
    tile_start,
    valid_elems,
    chunk_id,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_LOCAL_PES: tl.constexpr,
):
    chunk_start = chunk_id * CHUNK_SIZE
    if chunk_start >= valid_elems:
        return

    valid_chunk_elems = tl.minimum(CHUNK_SIZE, valid_elems - chunk_start)
    signal_chunk_ptr = signal_tile_ptr + chunk_id * NUM_LOCAL_PES

    thread_idx = tid(axis=0)
    if thread_idx < NUM_LOCAL_PES:
        libshmem_device.signal_wait_until(signal_chunk_ptr + thread_idx, libshmem_device.NVSHMEM_CMP_EQ, 1)
    __syncthreads()

    block_dim = num_warps() * 32
    for off in range(0, CHUNK_SIZE, block_dim):
        idx = thread_idx + off
        if idx < valid_chunk_elems:
            acc = tl.zeros((), dtype=tl.float32)
            for pe in tl.static_range(0, NUM_LOCAL_PES):
                acc += tl.load(scratch_tile_ptr + pe * BLOCK_SIZE + chunk_start + idx).to(tl.float32)
            tl.store(out_ptr + tile_start + chunk_start + idx, acc.to(out_ptr.dtype.element_ty))
    __syncthreads()


@triton_dist.jit
def allreduce_one_shot_nvshmem_remote_write_intra_node_kernel(pid, symm_in_ptr, symm_scratch_ptr, symm_signal_ptr,
                                                              out_ptr, elems, BLOCK_SIZE: tl.constexpr,
                                                              CHUNK_SIZE: tl.constexpr, NUM_LOCAL_PES: tl.constexpr):
    symm_in_ptr = tl.cast(symm_in_ptr, out_ptr.dtype)
    symm_scratch_ptr = tl.cast(symm_scratch_ptr, out_ptr.dtype)

    tile_start = pid * BLOCK_SIZE
    if tile_start >= elems:
        return

    NUM_CHUNKS: tl.constexpr = tl.cdiv(BLOCK_SIZE, CHUNK_SIZE)
    valid_elems = tl.minimum(BLOCK_SIZE, elems - tile_start)
    input_tile_ptr = symm_in_ptr + tile_start
    scratch_tile_ptr = symm_scratch_ptr + pid * BLOCK_SIZE * NUM_LOCAL_PES
    signal_tile_ptr = symm_signal_ptr + pid * NUM_LOCAL_PES * NUM_CHUNKS
    local_pe = libshmem_device.team_my_pe(libshmem_device.NVSHMEMX_TEAM_NODE)

    thread_idx = tid(axis=0)
    block_dim = num_warps() * 32
    for off in range(0, NUM_LOCAL_PES * NUM_CHUNKS, block_dim):
        idx = thread_idx + off
        if idx < NUM_LOCAL_PES * NUM_CHUNKS:
            tl.store(signal_tile_ptr + idx, 0)
    __syncthreads()
    libshmem_device.barrier_block(libshmem_device.NVSHMEMX_TEAM_NODE)
    __syncthreads()

    # Pipeline remote writes for chunk N+1 while reducing chunk N.
    allreduce_one_shot_nvshmem_issue_chunk(input_tile_ptr, scratch_tile_ptr, signal_tile_ptr, valid_elems, local_pe, 0,
                                           BLOCK_SIZE, CHUNK_SIZE, NUM_LOCAL_PES)
    for chunk_id in tl.static_range(1, NUM_CHUNKS):
        allreduce_one_shot_nvshmem_issue_chunk(input_tile_ptr, scratch_tile_ptr, signal_tile_ptr, valid_elems, local_pe,
                                               chunk_id, BLOCK_SIZE, CHUNK_SIZE, NUM_LOCAL_PES)
        allreduce_one_shot_nvshmem_reduce_chunk(scratch_tile_ptr, signal_tile_ptr, out_ptr, tile_start, valid_elems,
                                                chunk_id - 1, BLOCK_SIZE, CHUNK_SIZE, NUM_LOCAL_PES)
    allreduce_one_shot_nvshmem_reduce_chunk(scratch_tile_ptr, signal_tile_ptr, out_ptr, tile_start, valid_elems,
                                            NUM_CHUNKS - 1, BLOCK_SIZE, CHUNK_SIZE, NUM_LOCAL_PES)


@triton_dist.jit
def allreduce_nvshmem_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_LOCAL_PES: tl.constexpr,
):
    input_tensor = task_base_info.get_tensor(0)
    scratch_tensor = task_base_info.get_tensor(1)
    signal_tensor = task_base_info.get_tensor(2)
    output_tensor = task_base_info.get_tensor(3)

    input_ptr = input_tensor.data_ptr(tl.bfloat16)
    scratch_ptr = scratch_tensor.data_ptr(tl.bfloat16)
    signal_ptr = signal_tensor.data_ptr(tl.int64)
    output_ptr = output_tensor.data_ptr(tl.bfloat16)

    n_elements = output_tensor.size(0)
    tile_id = task_base_info.tile_id_or_start
    allreduce_one_shot_nvshmem_remote_write_intra_node_kernel(tile_id, input_ptr, scratch_ptr, signal_ptr, output_ptr,
                                                              n_elements, BLOCK_SIZE, CHUNK_SIZE, NUM_LOCAL_PES)
    scoreboard.release_tile(task_base_info, task_base_info.tile_id_or_start)
