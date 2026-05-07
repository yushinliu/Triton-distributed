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
import importlib
import tempfile
import os
import json
import copy
import nvshmem
import nvshmem.core

from ..core.code_generator import CodeGenerator, CodeGenOptions
from ..core.registry import registry
from ..core.task_base import TaskBase, DeviceProp, TaskDependency, TaskIDManager, MAX_NUM_TENSOR_DIMS
from ..core.builder import TaskBuilderBase
from ..core.graph import Graph
from typing import List, Dict, Any
from triton_dist.utils import NVSHMEM_SIGNAL_DTYPE, nvshmem_create_tensor, nvshmem_free_tensor_sync
from ..core.scheduler import enque_tasks
from triton_dist.models.utils import logger


def is_multicast_ptr(tensor):
    """
    On unsupported platforms, `mc_ptr` return nullptr
    """
    return nvshmem.bindings.mc_ptr(nvshmem.core.Teams.TEAM_NODE, tensor.data_ptr()) != 0


def check_tensor_shape(tensor, shape):
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(shape, (list, tuple))
    tensor_shape = list(tensor.shape)
    assert len(tensor_shape) == len(shape), f"tensor shape mismatch, tensor shape = {tensor.shape}, shape = {shape}"
    for (x, y) in zip(tensor_shape, shape):
        assert x == y, f"tensor shape mismatch, tensor shape = {tensor.shape}, shape = {shape}"


def check_tensor_dim(tensor, ndim):
    assert isinstance(tensor, torch.Tensor)
    assert len(
        tensor.shape
    ) == ndim, f"tensor dim mismatch, tensor dim = {len(tensor.shape)}, ndim = {ndim}, shape = {tensor.shape}"


def check_tensor_dtype(tensor, dtype):
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == dtype, f"tensor dtype mismatch, expect {dtype}, but got {tensor.dtype}"


def check_contiguous(tensors):
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    for t in tensors:
        assert t.is_contiguous()


def check_alignment(tensors):
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    for t in tensors:
        assert t.data_ptr() % 16 == 0, f"data_ptr = {t.data_ptr()}"


class ModelBuilder:

    def __init__(self, rank=0, world_size=1, local_world_size=1, num_warps=4, enable_profiling=False,
                 enable_dep_opt=True, enable_runtime_scheduler=False, enable_mlp_fc1_silu_fusion=False):
        self.reset()
        self._registry = registry
        self._code_generator = CodeGenerator()
        self._max_tensor_dim = MAX_NUM_TENSOR_DIMS
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        self.device_prop = DeviceProp(NUM_SMS=NUM_SMS)
        self.megakernel_tasks = []
        self.scoreboard = None
        self.wq_tensor = None  # work queue
        self.num_task_tensor = None  # num task in each work queue
        self.MAX_NUM_TILES_PER_OP = 1
        self.last_dependency = TaskDependency()
        self.max_layer_id = 0
        self.max_task_id = 0
        self.num_warps = num_warps
        self._metrics = {"memory": 0}
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.rank = rank
        self.local_rank = self.rank % self.local_world_size
        assert self.world_size % self.local_world_size == 0
        assert self.world_size > 0 and self.local_world_size > 0
        self.all_symm_tensors = []
        self.allreduce_phase_tensors = []
        if self.world_size > 1:
            self.barrier_all_intra_node_buf = self.create_symm_tensor([
                world_size,
            ], torch.int32)
            self.barrier_all_intra_node_buf.zero_()
            torch.distributed.barrier()
        else:
            self.barrier_all_intra_node_buf = None
        self.logger = logger
        self._enable_profiling = enable_profiling
        self._enable_dep_opt = enable_dep_opt
        self._enable_runtime_scheduler = enable_runtime_scheduler
        self._enable_mlp_fc1_silu_fusion = enable_mlp_fc1_silu_fusion
        self._codegen_options = CodeGenOptions(enable_profiling=enable_profiling,
                                               enable_runtime_scheduler=enable_runtime_scheduler)
        self.task_types_to_str = None
        self.trace_dependency_metadata = None
        self._graph = Graph()

    def create_symm_tensor(self, shape, dtype) -> torch.Tensor:
        tensor = nvshmem_create_tensor(shape, dtype)
        self.all_symm_tensors.append(tensor)
        return tensor

    def _update_metrics(self, op_type: str, io_tensors: List[List[torch.Tensor]], extra_params: Dict[str, Any] = {}):
        # avoid kv cache tensor being counted
        if "attn" in op_type or "kvcache" in op_type:
            return

        nbytes = 0
        all_tensors = io_tensors[0] + io_tensors[1]
        for ten in all_tensors:
            nbytes += ten.numel() * ten.element_size()

        if "memory" not in self._metrics:
            self._metrics["memory"] = 0
        self._metrics["memory"] += nbytes

    def get_memory_size(self):
        return self._metrics["memory"]

    def _update_tasks(self, tasks: List[TaskBase], do_not_update_dependency=False):
        assert len(tasks) > 0
        last_task = tasks[-1]
        if not do_not_update_dependency:
            self.last_dependency = TaskDependency(layer_id=last_task.layer_id, task_id=last_task.task_id, start_tiles=0,
                                                  end_tiles=last_task.num_tiles)
        self.megakernel_tasks += tasks
        for task in tasks:
            self.MAX_NUM_TILES_PER_OP = max(self.MAX_NUM_TILES_PER_OP, task.num_tiles)
            self.max_layer_id = max(task.layer_id, self.max_layer_id)
            self.max_task_id = max(task.task_id, self.max_task_id)

    @staticmethod
    def _interleave_tasks(first_tasks: List[TaskBase], second_tasks: List[TaskBase]) -> List[TaskBase]:
        interleaved = []
        max_len = max(len(first_tasks), len(second_tasks))
        for idx in range(max_len):
            if idx < len(first_tasks):
                interleaved.append(first_tasks[idx])
            if idx < len(second_tasks):
                interleaved.append(second_tasks[idx])
        return interleaved

    @staticmethod
    def _has_task_data_dependency(consumer_task: TaskBase, producer_task: TaskBase, producer_out_idx: int = 0,
                                  consumer_input_idx: int = 0) -> bool:
        input_dep_desc = consumer_task.get_input_dep_desc(consumer_input_idx)
        out_tiling_desc = producer_task.get_out_tiling_desc(producer_out_idx)
        if input_dep_desc is None or out_tiling_desc is None:
            return True
        if input_dep_desc.require_full:
            return True

        consumer_shape = input_dep_desc.input.shape
        producer_shape = producer_task.io_tensors[1][producer_out_idx].shape
        if (len(consumer_shape) == 1 and len(producer_shape) == 2 and len(input_dep_desc.start_indices) == 1
                and len(out_tiling_desc.start_indices) == 2 and out_tiling_desc.tile_sizes is not None):
            flat_start = input_dep_desc.start_indices[0]
            flat_end = min(flat_start + input_dep_desc.data_sizes[0], consumer_shape[0])
            if flat_start >= flat_end:
                return False

            rows, cols = producer_shape
            row_start, col_start = out_tiling_desc.start_indices
            tile_rows, tile_cols = out_tiling_desc.tile_sizes
            row_end = min(row_start + tile_rows, rows)
            col_end = min(col_start + tile_cols, cols)
            if row_start >= row_end or col_start >= col_end:
                return False

            flat_row_start = flat_start // cols
            flat_row_end = (flat_end - 1) // cols + 1
            for row in range(max(row_start, flat_row_start), min(row_end, flat_row_end)):
                row_flat_start = max(flat_start, row * cols)
                row_flat_end = min(flat_end, (row + 1) * cols)
                flat_col_start = row_flat_start - row * cols
                flat_col_end = row_flat_end - row * cols
                if max(flat_col_start, col_start) < min(flat_col_end, col_end):
                    return True
            return False

        return consumer_task.has_data_dependency(producer_task, producer_out_idx, consumer_input_idx)

    def _interleave_producer_with_allreduce_tasks(self, producer_tasks: List[TaskBase], local_tasks: List[TaskBase],
                                                  push_tasks: List[TaskBase]) -> List[TaskBase]:
        if len(producer_tasks) == 0 or len(local_tasks) == 0:
            return self._interleave_tasks(local_tasks, push_tasks)

        ready_items = []
        for local_task in local_tasks:
            max_producer_idx = -1
            has_dependency = False
            for idx, producer_task in enumerate(producer_tasks):
                if self._has_task_data_dependency(local_task, producer_task):
                    has_dependency = True
                    max_producer_idx = max(max_producer_idx, idx)
            if not has_dependency:
                max_producer_idx = len(producer_tasks) - 1
            ready_items.append((max_producer_idx, local_task.tile_id_or_start, local_task))
        ready_items.sort(key=lambda item: (item[0], item[1]))

        def read_positive_int_env(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default))))
            except ValueError:
                return default

        producer_chunk = read_positive_int_env("MEGA_KERNEL_NVSHMEM_INTERLEAVE_PRODUCER_CHUNK", 16)
        ar_tiles_per_flush = read_positive_int_env("MEGA_KERNEL_NVSHMEM_INTERLEAVE_AR_TILES", 1)
        push_by_tile = {task.tile_id_or_start: task for task in push_tasks}
        emitted_push_ids = set()
        interleaved = []
        ready_idx = 0
        for producer_idx, producer_task in enumerate(producer_tasks):
            interleaved.append(producer_task)
            should_flush = ((producer_idx + 1) % producer_chunk == 0 or producer_idx == len(producer_tasks) - 1)
            if not should_flush:
                continue

            emitted_ar_tiles = 0
            while (ready_idx < len(ready_items) and ready_items[ready_idx][0] <= producer_idx
                   and emitted_ar_tiles < ar_tiles_per_flush):
                _, tile_id, local_task = ready_items[ready_idx]
                interleaved.append(local_task)
                push_task = push_by_tile.get(tile_id)
                if push_task is not None:
                    interleaved.append(push_task)
                    emitted_push_ids.add(id(push_task))
                ready_idx += 1
                emitted_ar_tiles += 1

        while ready_idx < len(ready_items):
            _, tile_id, local_task = ready_items[ready_idx]
            interleaved.append(local_task)
            push_task = push_by_tile.get(tile_id)
            if push_task is not None:
                interleaved.append(push_task)
                emitted_push_ids.add(id(push_task))
            ready_idx += 1
        for push_task in push_tasks:
            if id(push_task) not in emitted_push_ids:
                interleaved.append(push_task)
        return interleaved

    def _interleave_recent_producer_with_allreduce_tasks(self, producer_dependency: TaskDependency,
                                                         local_tasks: List[TaskBase], push_tasks: List[TaskBase]):
        num_allreduce_tasks = len(local_tasks) + len(push_tasks)
        allreduce_start = len(self.megakernel_tasks) - num_allreduce_tasks
        if producer_dependency.layer_id < 0 or producer_dependency.task_id < 0 or allreduce_start <= 0:
            self.megakernel_tasks[-num_allreduce_tasks:] = self._interleave_tasks(local_tasks, push_tasks)
            return

        producer_indices = [
            idx for idx, task in enumerate(self.megakernel_tasks[:allreduce_start])
            if task.layer_id == producer_dependency.layer_id and task.task_id == producer_dependency.task_id
        ]
        if len(producer_indices) == 0:
            self.megakernel_tasks[-num_allreduce_tasks:] = self._interleave_tasks(local_tasks, push_tasks)
            return

        producer_start = producer_indices[0]
        producer_end = producer_indices[-1] + 1
        is_contiguous = producer_indices == list(range(producer_start, producer_end))
        if not is_contiguous or producer_end != allreduce_start:
            self.megakernel_tasks[-num_allreduce_tasks:] = self._interleave_tasks(local_tasks, push_tasks)
            return

        producer_tasks = self.megakernel_tasks[producer_start:producer_end]
        interleaved = self._interleave_producer_with_allreduce_tasks(producer_tasks, local_tasks, push_tasks)
        self.megakernel_tasks = self.megakernel_tasks[:producer_start] + interleaved
        self.logger.log(
            f"interleaved {len(producer_tasks)} producer tasks with {len(local_tasks)} local and "
            f"{len(push_tasks)} push NVSHMEM allreduce tasks",
            level="debug")

    def get_sm_activity(self):
        assert self._enable_profiling
        from triton_dist.tools.profiler import parse_to_tracks

        block_idx_to_tracks = parse_to_tracks(self.profile_buf)
        sb_wait_deps_task_type = None
        for k, v in self.task_types_to_str.items():
            if v == "scoreboard_wait_deps":
                sb_wait_deps_task_type = k
        assert sb_wait_deps_task_type is not None
        wait_deps_time = 0
        e2e_time = 0
        NUM_SMS = self.device_prop.NUM_SMS
        act_time = 0
        for i in range(NUM_SMS):
            assert i in block_idx_to_tracks.keys()
            tracks_list = block_idx_to_tracks[i]
            block_time = 0
            for track in tracks_list:
                if track.task_type == sb_wait_deps_task_type:
                    wait_deps_time += track.duration
                else:
                    act_time += track.duration
                block_time += track.duration
            e2e_time = max(e2e_time, block_time)
        self.logger.log(
            f"e2e_time = {e2e_time * 1.0 / 1e6} ms, avg_time = {act_time * 1.0 / NUM_SMS  / 1e6} ms, avg_wait_deps_per_block_time = {wait_deps_time * 1.0 / NUM_SMS  / 1e6} ms",
            level="debug")
        return act_time * 1.0 / (e2e_time * NUM_SMS)

    def get_task_builder(self, op_type: str) -> 'TaskBuilderBase':
        task_type = self._registry.get_op_mapping(op_type)
        if not task_type:
            raise ValueError(f"Unsupported op type: {op_type}")
        builder_cls = registry.get_builder(task_type)
        return builder_cls

    def _convert_op(self, op_type: str, layer_id: int, io_tensors: List[List[torch.Tensor]],
                    extra_params: Dict[str, Any] = {}) -> List[TaskBase]:
        assert len(io_tensors) == 2
        check_contiguous(io_tensors[0])
        check_contiguous(io_tensors[1])
        check_alignment(io_tensors[0])
        check_alignment(io_tensors[1])

        builder_cls = self.get_task_builder(op_type)
        tasks = builder_cls.build_tasks(device_prop=self.device_prop, layer_id=layer_id,
                                        dependency=self.last_dependency, io_tensors=io_tensors,
                                        extra_params=extra_params)
        self._update_tasks(tasks)
        self._update_metrics(op_type, io_tensors, extra_params)
        self._graph.new_node(tasks=tasks, op_type=op_type, io_tensors=io_tensors, extra_params=extra_params)
        return tasks

    def _make_fc(self, op_type: str, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
                 layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM == M and oN == N
        assert K % 32 == 0

        self._convert_op(op_type, layer_id, [[input, weight], [output]])

    def make_fc1(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc1", input, weight, output, layer_id)

    def make_fused_fc1_silu_mul_up(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
                                   layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        check_tensor_dim(output, 2)
        M, K = input.shape
        N2, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert N2 == oN * 2
        assert oM == M
        assert K % 32 == 0
        self._convert_op("mlp_fc1_silu_mul_up", layer_id, [[input, weight], [output]])

    def make_fused_rms_norm_fc1_silu_mul_up(self, input: torch.Tensor, rms_weight: torch.Tensor, weight: torch.Tensor,
                                            output: torch.Tensor, rms_eps: float = 1e-6, layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(rms_weight, 1)
        check_tensor_dim(weight, 2)
        check_tensor_dim(output, 2)
        M, K = input.shape
        N2, wK = weight.shape
        oM, oN = output.shape
        assert rms_weight.shape[0] == K
        assert K == wK
        assert N2 == oN * 2
        assert oM == M
        assert K % 32 == 0
        self._convert_op("rms_norm_mlp_fc1_silu_mul_up", layer_id, [[input, rms_weight, weight], [output]],
                         {"rms_eps": rms_eps})

    def make_fc2(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc2", input, weight, output, layer_id)

    def make_qkv_proj(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("qkv_proj", input, weight, output, layer_id)

    def make_o_proj(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("o_proj", input, weight, output, layer_id)

    def make_o_proj_add(self, input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, output: torch.Tensor,
                        layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        check_tensor_dim(residual, 2)
        check_tensor_dim(output, 2)
        M, K = input.shape
        N, wK = weight.shape
        rM, rN = residual.shape
        oM, oN = output.shape
        assert K == wK
        assert (rM, rN) == (M, N)
        assert (oM, oN) == (M, N)
        assert K % 32 == 0
        self._convert_op("o_proj_add", layer_id, [[input, weight, residual], [output]])

    def make_linear(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM == M and oN == N
        self._convert_op("linear", layer_id, [[input, weight], [output]])

    def make_flash_decode(self, query, key_cache: torch.Tensor, value_cache: torch.Tensor, block_tables: torch.Tensor,
                          kv_lens: torch.Tensor, output: torch.Tensor, sm_scale=None, soft_cap=0.0, layer_id: int = 0):
        """
            query: (batch, seq_len, num_q_heads, q_head_dim)
            key_cache: (MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, q_head_dim)
            value_cache: (MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, v_head_dim)
            block_tables: (batch, MAX_NUM_BLOCKS_PER_SEQ)
            kv_lens: (batch)
            output: (batch, seq_len, num_q_heads, v_head_dim)
        """
        check_tensor_dim(query, 4)
        check_tensor_dim(key_cache, 4)
        check_tensor_dim(value_cache, 4)
        check_tensor_dim(block_tables, 2)
        check_tensor_dim(kv_lens, 1)
        check_tensor_dim(output, 4)
        check_tensor_dtype(query, torch.bfloat16)
        check_tensor_dtype(key_cache, torch.bfloat16)
        check_tensor_dtype(value_cache, torch.bfloat16)
        check_tensor_dtype(block_tables, torch.int32)
        check_tensor_dtype(kv_lens, torch.int32)
        check_tensor_dtype(output, torch.bfloat16)
        assert query.shape[0] == kv_lens.shape[0]

        batch, q_seq_len, num_q_heads, q_head_dim = query.shape
        v_head_dim = value_cache.shape[-1]
        assert q_seq_len == 1, "currently flash decoding only support q_seq_len == 1"
        if sm_scale is None:
            sm_scale = q_head_dim**-0.5
        soft_cap = 0.0
        NUM_KV_SPLITS = 32
        extra_params = {"sm_scale": sm_scale, "soft_cap": soft_cap, "NUM_KV_SPLITS": NUM_KV_SPLITS}
        # Assuming q_seq_len is 1, ignore the seq_len dimension to reduce the number of tensor dimensions (MAX_TENSOR_DIM = 4)
        partial_out = torch.empty([batch, num_q_heads, NUM_KV_SPLITS, v_head_dim], dtype=torch.float32,
                                  device=query.device)
        lse = torch.empty([batch, num_q_heads, NUM_KV_SPLITS], dtype=torch.float32, device=query.device)
        self._convert_op("attn_split", layer_id,
                         [[query, key_cache, value_cache, block_tables, kv_lens], [partial_out, lse]], extra_params)
        self._convert_op("attn_combine", layer_id, [[kv_lens, partial_out, lse], [output]], extra_params)

    def make_qkv_pack_flash_attn(self, qkv: torch.Tensor, output: torch.Tensor, sm_scale=None, soft_cap=0.0,
                                 is_causal=True, layer_id: int = 0):
        """
            Args:
                qkv: (bs, seq, nheads_q + 2 * nheads_kv, head_dim)
                out: (bs, seq, nheads_q, head_dim)
        """
        check_tensor_dim(qkv, 4)
        check_tensor_dim(output, 4)
        check_tensor_dtype(qkv, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)
        assert qkv.shape[0] == output.shape[0] and qkv.shape[1] == output.shape[1] and qkv.shape[3] == output.shape[3]
        q_head_dim = qkv.shape[-1]

        if sm_scale is None:
            sm_scale = q_head_dim**-0.5
        extra_params = {"sm_scale": sm_scale, "soft_cap": soft_cap, "is_causal": is_causal}
        self._convert_op("qkv_pack_flash_attn", layer_id, [[qkv], [output]], extra_params)

    def make_flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, output: torch.Tensor, sm_scale=None,
                        soft_cap=0.0, is_causal=True, layer_id: int = 0):
        """
            Args:
                q: (bs, seq, nheads_q, head_dim)
                k: (bs, seq, nheads_kv, head_dim)
                v: (bs, seq, nheads_kv, head_dim)
                out: (bs, seq, nheads_q, head_dim)
        """
        check_tensor_dim(q, 4)
        check_tensor_dim(k, 4)
        check_tensor_dim(v, 4)

        check_tensor_dim(output, 4)
        check_tensor_dtype(q, torch.bfloat16)
        check_tensor_dtype(k, torch.bfloat16)
        check_tensor_dtype(v, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)
        assert q.shape[0] == output.shape[0] and q.shape[1] == output.shape[1] and q.shape[3] == output.shape[3]
        q_head_dim = q.shape[-1]

        if sm_scale is None:
            sm_scale = q_head_dim**-0.5
        extra_params = {"sm_scale": sm_scale, "soft_cap": soft_cap, "is_causal": is_causal}
        self._convert_op("flash_attn", layer_id, [[q, k, v], [output]], extra_params)

    def make_silu_mul_up(self, fc1_out, act_out, layer_id=0):
        check_tensor_dim(fc1_out, 2)
        check_tensor_dim(act_out, 2)
        M, N = fc1_out.shape
        assert act_out.shape[0] == M
        assert act_out.shape[1] * 2 == N
        self._convert_op("silu_mul_up", layer_id, [[fc1_out], [act_out]])

    def make_qk_norm_rope_update_kvcache(self, qkv: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                                         block_tables: torch.Tensor, kv_lens: torch.Tensor, q_rms_weight: torch.Tensor,
                                         k_rms_weight: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                                         q_norm_rope: torch.Tensor, q_rms_eps: float = 1e-6, k_rms_eps: float = 1e-6,
                                         rope_theta: int = 1000000, skip_q_norm=False, skip_k_norm=False, layer_id=0):
        """
            this op assume that kv_lens has been update (kv_lens = history_kv_len + seq_len(qkv.shape[1]))
            inplace update new kv to key_cache/value_cache
            cos_cache/sin_cache: [batch, seq_len, head_dim]
            rms_weight: [head_dim]
        """
        check_tensor_dim(qkv, 4)
        check_tensor_dim(key_cache, 4)
        check_tensor_dim(value_cache, 4)
        check_tensor_dim(block_tables, 2)
        check_tensor_dim(kv_lens, 1)
        check_tensor_dim(q_norm_rope, 4)
        check_tensor_dim(cos_cache, 3)
        check_tensor_dim(sin_cache, 3)
        check_tensor_dim(q_rms_weight, 1)
        check_tensor_dim(k_rms_weight, 1)
        check_tensor_dtype(qkv, torch.bfloat16)
        check_tensor_dtype(key_cache, torch.bfloat16)
        check_tensor_dtype(value_cache, torch.bfloat16)
        check_tensor_dtype(block_tables, torch.int32)
        check_tensor_dtype(kv_lens, torch.int32)
        check_tensor_dtype(q_norm_rope, torch.bfloat16)
        check_tensor_dtype(cos_cache, torch.float32)
        check_tensor_dtype(sin_cache, torch.float32)
        check_tensor_dtype(q_rms_weight, torch.bfloat16)
        check_tensor_dtype(k_rms_weight, torch.bfloat16)

        assert qkv.shape[0] == kv_lens.shape[0]
        assert cos_cache.shape[0] == sin_cache.shape[0]
        assert cos_cache.shape[0] == qkv.shape[0] or cos_cache.shape[0] == 1
        extra_params = {
            "q_rms_eps": q_rms_eps, "k_rms_eps": k_rms_eps, "rope_theta": rope_theta, "skip_q_norm": skip_q_norm,
            "skip_k_norm": skip_k_norm
        }
        self._convert_op("qk_norm_rope_update_kvcache", layer_id,
                         [[qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache],
                          [key_cache, value_cache, q_norm_rope]], extra_params)

    def make_qkv_pack_qk_norm_rope_split_v(self, qkv: torch.Tensor, kv_lens: torch.Tensor, q_rms_weight: torch.Tensor,
                                           k_rms_weight: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                                           q_norm_rope: torch.Tensor, k_norm_rope: torch.Tensor, v: torch.Tensor,
                                           q_rms_eps: float, k_rms_eps: float, layer_id=0):
        """
            Inputs:
                qkv: [bs, seq_len, num_q_heads + 2 * num_kv_heads, head_dim]
                kv_lens: [bs]
                q/k_rms_weight: [head_dim]
                cos_cache/sin_cache: [MAX_SEQ_LEN, head_dim] or [1, MAX_SEQ_LEN, head_dim]

            Outputs:
                q_norm_rope: [bs, seq_len, num_q_heads, head_dim]
                k_norm_rope: [bs, seq_len, num_kv_heads, head_dim]
                v: [bs, seq_len, num_kv_heads, head_dim]
            
            Formula:
                q, k, v = qkv.split([num_q_heads, num_kv_heads, num_kv_heads], dim=-2)
                q_norm_rope = rope(rms_norm(q, q_rms_weight, q_rms_eps), cos_cache, sin_cache)
                k_norm_rope = rope(rms_norm(k, k_rms_weight, k_rms_eps), cos_cache, sin_cache)
        """
        check_tensor_dim(qkv, 4)
        check_tensor_dim(q_norm_rope, 4)
        check_tensor_dim(k_norm_rope, 4)
        bs, seq_len, num_total_heads, head_dim = qkv.shape
        num_q_heads = q_norm_rope.shape[-2]
        num_kv_heads = k_norm_rope.shape[-2]
        assert num_total_heads == num_q_heads + 2 * num_kv_heads
        check_tensor_shape(q_norm_rope, (bs, seq_len, num_q_heads, head_dim))
        check_tensor_shape(k_norm_rope, (bs, seq_len, num_kv_heads, head_dim))
        check_tensor_shape(v, (bs, seq_len, num_kv_heads, head_dim))
        check_tensor_shape(kv_lens, (bs, ))
        assert len(sin_cache.shape) == len(cos_cache.shape)
        assert sin_cache.shape == cos_cache.shape
        assert sin_cache.shape[-1] == head_dim

        check_tensor_dtype(q_norm_rope, torch.bfloat16)
        check_tensor_dtype(k_norm_rope, torch.bfloat16)
        check_tensor_dtype(cos_cache, torch.float32)
        check_tensor_dtype(sin_cache, torch.float32)
        check_tensor_dtype(q_rms_weight, torch.bfloat16)
        check_tensor_dtype(k_rms_weight, torch.bfloat16)

        extra_params = {"q_rms_eps": q_rms_eps, "k_rms_eps": k_rms_eps}
        self._convert_op(
            "qkv_pack_qk_norm_rope_split_v", layer_id,
            [[qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache], [q_norm_rope, k_norm_rope, v]],
            extra_params)

    def make_rms_norm(self, input: torch.Tensor, rms_weight: torch.Tensor, output: torch.Tensor, rms_eps: float = 1e-6,
                      layer_id=0):
        check_tensor_dtype(input, torch.bfloat16)
        check_tensor_dtype(rms_weight, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)
        check_tensor_dim(rms_weight, 1)
        # reshape to 2d tensor
        input = input.reshape(-1, input.shape[-1])
        output = output.reshape(-1, input.shape[-1])

        assert input.shape == output.shape
        assert input.shape[-1] == rms_weight.shape[0]
        extra_params = {"rms_eps": rms_eps}
        self._convert_op("rms_norm", layer_id, [[input, rms_weight], [output]], extra_params)

    def make_add(self, lhs: torch.Tensor, rhs: torch.Tensor, output: torch.Tensor, layer_id=0):
        check_tensor_dtype(lhs, torch.bfloat16)
        check_tensor_dtype(rhs, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)

        assert lhs.shape == rhs.shape and lhs.shape == output.shape
        lhs = lhs.reshape(-1)
        rhs = rhs.reshape(-1)
        output = output.reshape(-1)

        self._convert_op("add", layer_id, [[lhs, rhs], [output]])

    def make_barrier_all_intra_node(self, wait_inputs=None, layer_id=0):
        """
            `wait_inputs` is used to build data dependency.
        """
        assert self.world_size > 1
        wait_inputs = [] if wait_inputs is None else wait_inputs
        extra_params = {"local_rank": self.local_rank, "local_world_size": self.local_world_size}
        self._convert_op("barrier_all_intra_node", layer_id,
                         [[self.barrier_all_intra_node_buf] + wait_inputs, wait_inputs], extra_params)

    def make_allreduce(self, input: torch.Tensor, output: torch.Tensor, double_input_buffer=False, layer_id=0,
                       implementation="multimem"):
        """
            if double_input_buffer is True, user needs to ensure that the input of two consecutive allreduce are completely different buffers,
            otherwise, the output may be wrong.
        """
        assert self.world_size > 1
        input = input.reshape(-1)
        output = output.reshape(-1)
        nbytes = input.numel() * input.element_size()
        assert input.shape == output.shape and input.dtype == output.dtype
        if implementation == "multimem":
            if not is_multicast_ptr(input):
                raise ValueError(
                    "The input tensor needs to be a symmetric buffer and AR is only supported in Hopper and later GPU architectures"
                )
            assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0"  # for multicast
            assert nbytes % 128 == 0
        elif implementation == "nvshmem":
            if self._enable_runtime_scheduler:
                raise ValueError("NVSHMEM allreduce overlap path does not support enable_runtime_scheduler=True")
            producer_dependency = self.last_dependency
            builder_cls = self.get_task_builder("allreduce_nvshmem")
            kernel_config = builder_cls.create_config()
            num_tiles = (input.numel() + kernel_config.BLOCK_SIZE - 1) // kernel_config.BLOCK_SIZE
            scratch_buf = self.create_symm_tensor((num_tiles * self.local_world_size * kernel_config.BLOCK_SIZE, ),
                                                  input.dtype)
            signal_buf = self.create_symm_tensor((num_tiles * self.local_world_size, ), NVSHMEM_SIGNAL_DTYPE)
            signal_buf.zero_()
            phase_buf = torch.zeros((1, ), dtype=NVSHMEM_SIGNAL_DTYPE, device=torch.cuda.current_device())
            self.allreduce_phase_tensors.append(phase_buf)
        else:
            raise ValueError(f"Unsupported allreduce implementation: {implementation}")
        if implementation == "multimem":
            self.make_barrier_all_intra_node(wait_inputs=[input], layer_id=layer_id)
            self._convert_op("allreduce", layer_id, [[input], [output]])
        else:
            # NVSHMEM uses per-tile phase/signal buffers, so it does not need a full-rank arrival barrier here.
            local_io_tensors = [[input, scratch_buf, signal_buf, phase_buf], [output]]
            push_io_tensors = [[input, scratch_buf, signal_buf, phase_buf], []]
            local_builder_cls = self.get_task_builder("allreduce_nvshmem")
            push_builder_cls = self.get_task_builder("allreduce_nvshmem_push")
            local_tasks = local_builder_cls.build_tasks(device_prop=self.device_prop,
                                                        layer_id=layer_id,
                                                        dependency=self.last_dependency,
                                                        io_tensors=local_io_tensors,
                                                        extra_params={})
            push_tasks = push_builder_cls.build_tasks(device_prop=self.device_prop,
                                                      layer_id=layer_id,
                                                      dependency=TaskDependency(),
                                                      io_tensors=push_io_tensors,
                                                      extra_params={})
            self._update_tasks(local_tasks)
            self._graph.new_node(tasks=local_tasks,
                                 op_type="allreduce_nvshmem",
                                 io_tensors=local_io_tensors,
                                 extra_params={})
            self._update_metrics("allreduce_nvshmem", local_io_tensors)
            self._update_tasks(push_tasks, do_not_update_dependency=True)
            self._graph.new_node(tasks=push_tasks,
                                 op_type="allreduce_nvshmem_push",
                                 io_tensors=push_io_tensors,
                                 extra_params={})
            self._update_metrics("allreduce_nvshmem_push", push_io_tensors)
            self._interleave_recent_producer_with_allreduce_tasks(producer_dependency, local_tasks, push_tasks)
        if not double_input_buffer:
            self.make_barrier_all_intra_node(wait_inputs=[output], layer_id=layer_id)

    def make_prefetch(self, weight: torch.tensor, layer_id=0):
        io_tensors = [[weight], []]
        check_contiguous(io_tensors[0])
        check_contiguous(io_tensors[1])
        check_alignment(io_tensors[0])
        check_alignment(io_tensors[1])
        assert weight.dtype == torch.bfloat16
        check_tensor_dim(weight, 2)
        check_tensor_dtype(weight, torch.bfloat16)
        assert weight.shape[1] % 32 == 0

        builder_cls = self.get_task_builder("prefetch")
        tasks = builder_cls.build_tasks(device_prop=self.device_prop, layer_id=layer_id,
                                        dependency=TaskDependency(),  # no dependency
                                        io_tensors=io_tensors, extra_params={})
        self._update_tasks(tasks, do_not_update_dependency=True)

    def reset(self):
        TaskIDManager.reset_all_ids()

    def compile(self):
        self.logger.log(f"num_total_tasks = {len(self.megakernel_tasks)}", level="debug")
        if self._enable_dep_opt:
            # Build optimized dependencies in-place without overriding the explicit enqueue order.
            self._graph.to_tasks()
        megakernel_tasks = self.megakernel_tasks
        if self._enable_runtime_scheduler:
            num_sms = 1
        else:
            num_sms = self.device_prop.NUM_SMS
        (self.wq_tensor, self.num_tasks_tensor, self.scoreboard, self.task_deps_tensor,
         self.trace_dependency_metadata) = enque_tasks(num_sms,
                                                       megakernel_tasks,
                                                       "round_robin",
                                                       enable_dependency_opt=not self._enable_runtime_scheduler,
                                                       return_trace_metadata=True)
        self.scoreboard = torch.zeros((self.max_layer_id + 1, self.max_task_id + 1, self.MAX_NUM_TILES_PER_OP),
                                      dtype=torch.int32, device=torch.cuda.current_device())

        src, task_types_to_str = self._code_generator.generate_code(self.megakernel_tasks, self._codegen_options)
        self.logger.log(src, level="debug")
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp.write(src.encode('utf-8'))
            tmp_path = tmp.name

        module_name = os.path.basename(tmp_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._gen_kernel = module.MEGA_TRITON_KERNEL
        self.task_types_to_str = task_types_to_str
        max_num_profile_slots = (self.wq_tensor.shape[0] + 4) * self.device_prop.NUM_SMS * 4
        self.logger.log(f"max_num_profile_slots = {max_num_profile_slots}", level="debug")
        if self._enable_profiling:
            from triton_dist.tools.profiler import alloc_profiler_buffer

            self.profile_buf = alloc_profiler_buffer(max_num_profile_slots)
        else:
            self.profile_buf = None

    def dump_trace(self, trace_file_prefix="MEGA_KERNEL_TRACE"):
        if self._enable_profiling:
            from triton_dist.tools.profiler import export_to_perfetto_trace

            profiler_dir = os.environ.get("MEGA_KERNEL_PRODILER_DIR", "./prof")
            os.makedirs(profiler_dir, exist_ok=True)
            trace_file = os.path.join(profiler_dir, f"{trace_file_prefix}_RANK_{self.rank}")
            self.dump_trace_dependency_map(trace_file)
            export_to_perfetto_trace(self.profile_buf,
                                     self.task_types_to_str,
                                     trace_file,
                                     dependency_metadata=self.trace_dependency_metadata)
        else:
            self.logger.log("profiler not enabled, please set enable_profiling=True", level="warning")

    def dump_trace_dependency_map(self, trace_file):
        if self.trace_dependency_metadata is None:
            return
        metadata = copy.deepcopy(self.trace_dependency_metadata)
        metadata.update({
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "perfetto_trace_file": trace_file if trace_file.endswith(".perfetto-trace") else trace_file + ".perfetto-trace",
            "task_type_id_to_name": {str(k): v for k, v in self.task_types_to_str.items()},
            "num_tasks_per_block": self.num_tasks_tensor.detach().cpu().tolist(),
        })
        dependency_file = trace_file + ".deps.json"
        with open(dependency_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def run(self):
        grid = lambda META: (self.device_prop.NUM_SMS, )
        work_queue_start = torch.empty((1, ), dtype=torch.int32, device=torch.cuda.current_device())
        if self._enable_runtime_scheduler:
            work_queue_start.fill_(0)
        for phase_tensor in self.allreduce_phase_tensors:
            phase_tensor.add_(1)
        if self._enable_profiling:
            from triton_dist.tools.profiler import reset_profiler_buffer

            assert self.profile_buf is not None
            reset_profiler_buffer(self.profile_buf)
            self._gen_kernel[grid](
                self.profile_buf,
                work_queue_start,
                self.wq_tensor,
                self.num_tasks_tensor,
                self.scoreboard,
                self.task_deps_tensor,
                INT_PER_DEPS=self.task_deps_tensor.shape[1],
                INT_PER_TASK=self.wq_tensor.shape[2],
                MAX_TASK_ID=self.scoreboard.shape[1],
                MAX_NUM_TILES_PER_OP=self.scoreboard.shape[2],
                MAX_NUM_TENSOR_DIMS=self._max_tensor_dim,
                NUM_SMS=self.device_prop.NUM_SMS,
                num_warps=self.num_warps,
            )
        else:
            self._gen_kernel[grid](
                work_queue_start,
                self.wq_tensor,
                self.num_tasks_tensor,
                self.scoreboard,
                self.task_deps_tensor,
                INT_PER_DEPS=self.task_deps_tensor.shape[1],
                INT_PER_TASK=self.wq_tensor.shape[2],
                MAX_TASK_ID=self.scoreboard.shape[1],
                MAX_NUM_TILES_PER_OP=self.scoreboard.shape[2],
                MAX_NUM_TENSOR_DIMS=self._max_tensor_dim,
                NUM_SMS=self.device_prop.NUM_SMS,
                num_warps=self.num_warps,
            )
        self.scoreboard.zero_()

    def finalize(self):
        for ten in self.all_symm_tensors:
            nvshmem_free_tensor_sync(ten)
