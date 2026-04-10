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
from typing import List
from .utils import cdiv
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class AllReduceConfig(ConfigBase):
    BLOCK_SIZE: int = 1024


@dataclass
class AllReduceTask(TaskBase):
    config: AllReduceConfig


@dataclass
class AllReduceNVSHMEMTask(TaskBase):
    config: AllReduceConfig


@dataclass
class AllReduceNVSHMEMPushTask(TaskBase):
    config: AllReduceConfig


def allreduce_config_factory(**kwargs) -> AllReduceConfig:
    return dataclasses.replace(AllReduceConfig(), **kwargs)


def codegen_allreduce(task: AllReduceConfig) -> str:
    config: AllReduceConfig = task.config

    code = f"""
allreduce_task_compute(task_base_info, scoreboard, BLOCK_SIZE={config.BLOCK_SIZE})
"""
    return code


def codegen_allreduce_nvshmem(task: AllReduceNVSHMEMTask) -> str:
    config: AllReduceConfig = task.config
    scratch_tensor = task.io_tensors[0][0]
    num_local_pes = scratch_tensor.numel() // (task.num_tiles * config.BLOCK_SIZE)
    code = f"""
allreduce_nvshmem_task_compute(task_base_info, scoreboard, BLOCK_SIZE={config.BLOCK_SIZE}, NUM_LOCAL_PES={num_local_pes})
"""
    return code


def codegen_allreduce_nvshmem_push(task: AllReduceNVSHMEMPushTask) -> str:
    config: AllReduceConfig = task.config
    scratch_tensor = task.io_tensors[0][1]
    num_local_pes = scratch_tensor.numel() // (task.num_tiles * config.BLOCK_SIZE)
    code = f"""
allreduce_nvshmem_push_task_compute(task_base_info, scoreboard, BLOCK_SIZE={config.BLOCK_SIZE}, NUM_LOCAL_PES={num_local_pes})
"""
    return code


@registry.register_task(op_type="allreduce", task_cls=AllReduceTask, config_factory=allreduce_config_factory,
                        codegen_func=codegen_allreduce)
class AllReduceTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        input, output = io_tensors[0][0], io_tensors[1][0]
        num_elements = output.numel()
        assert input.shape == output.shape and len(input.shape) == 1
        assert num_elements * output.element_size() % 128 == 0
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        num_tiles = cdiv(num_elements, kernel_config.BLOCK_SIZE)

        cls.log(
            f"AllReduce Task: num_tiles = {num_tiles}, num_elements = {num_elements}, BLOCK_SIZE = {kernel_config.BLOCK_SIZE}, dependency = {dependency}"
        )
        tasks = []
        for i in range(num_tiles):
            tile_start = i * kernel_config.BLOCK_SIZE
            tile_size = min(kernel_config.BLOCK_SIZE, num_elements - tile_start)
            input_desc = InputDependencyDesc(input, require_full=False, start_indices=(tile_start, ),
                                             data_sizes=(tile_size, ))
            output_desc = OutputTilingDesc(start_indices=(tile_start, ), tile_sizes=(tile_size, ))
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params,
                                 inputs_dep={input: input_desc}, outs_tile_mapping={output: output_desc}))
        return tasks


@registry.register_task(op_type="allreduce_nvshmem_push", task_cls=AllReduceNVSHMEMPushTask,
                        config_factory=allreduce_config_factory, codegen_func=codegen_allreduce_nvshmem_push)
class AllReduceNVSHMEMPushTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        input, scratch, signal, phase = io_tensors[0]
        scratch_out, signal_out = io_tensors[1]
        num_elements = input.numel()
        assert len(input.shape) == 1
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        num_tiles = cdiv(num_elements, kernel_config.BLOCK_SIZE)
        assert scratch.numel() % (num_tiles * kernel_config.BLOCK_SIZE) == 0
        num_local_pes = scratch.numel() // (num_tiles * kernel_config.BLOCK_SIZE)
        assert scratch_out.data_ptr() == scratch.data_ptr()
        assert signal_out.data_ptr() == signal.data_ptr()
        assert signal.numel() == num_tiles * num_local_pes

        cls.log(
            f"AllReduceNVSHMEMPush Task: num_tiles = {num_tiles}, num_elements = {num_elements}, BLOCK_SIZE = {kernel_config.BLOCK_SIZE}, dependency = {dependency}"
        )
        tasks = []
        for i in range(num_tiles):
            tile_start = i * kernel_config.BLOCK_SIZE
            tile_size = min(kernel_config.BLOCK_SIZE, num_elements - tile_start)
            scratch_start = i * num_local_pes * kernel_config.BLOCK_SIZE
            signal_start = i * num_local_pes
            input_desc = InputDependencyDesc(input, require_full=False, start_indices=(tile_start, ),
                                             data_sizes=(tile_size, ))
            phase_desc = InputDependencyDesc(phase, require_full=True)
            scratch_desc = OutputTilingDesc(start_indices=(scratch_start, ),
                                            tile_sizes=(num_local_pes * kernel_config.BLOCK_SIZE, ))
            signal_desc = OutputTilingDesc(start_indices=(signal_start, ), tile_sizes=(num_local_pes, ))
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params,
                                 inputs_dep={
                                     input: input_desc,
                                     phase: phase_desc,
                                 },
                                 outs_tile_mapping={
                                     scratch_out: scratch_desc,
                                     signal_out: signal_desc,
                                 }))
        return tasks


@registry.register_task(op_type="allreduce_nvshmem", task_cls=AllReduceNVSHMEMTask,
                        config_factory=allreduce_config_factory, codegen_func=codegen_allreduce_nvshmem)
class AllReduceNVSHMEMTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        scratch, signal, phase = io_tensors[0]
        output = io_tensors[1][0]
        num_elements = output.numel()
        assert len(output.shape) == 1
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        num_tiles = cdiv(num_elements, kernel_config.BLOCK_SIZE)
        expected_scratch_elems = num_tiles * kernel_config.BLOCK_SIZE
        assert scratch.numel() % expected_scratch_elems == 0
        num_local_pes = scratch.numel() // expected_scratch_elems
        assert signal.numel() == num_tiles * num_local_pes

        cls.log(
            f"AllReduceNVSHMEM Task: num_tiles = {num_tiles}, num_elements = {num_elements}, BLOCK_SIZE = {kernel_config.BLOCK_SIZE}, dependency = {dependency}"
        )
        tasks = []
        for i in range(num_tiles):
            tile_start = i * kernel_config.BLOCK_SIZE
            tile_size = min(kernel_config.BLOCK_SIZE, num_elements - tile_start)
            scratch_start = i * num_local_pes * kernel_config.BLOCK_SIZE
            signal_start = i * num_local_pes
            scratch_desc = InputDependencyDesc(scratch, require_full=False, start_indices=(scratch_start, ),
                                               data_sizes=(num_local_pes * kernel_config.BLOCK_SIZE, ))
            signal_desc = InputDependencyDesc(signal, require_full=False, start_indices=(signal_start, ),
                                              data_sizes=(num_local_pes, ))
            phase_desc = InputDependencyDesc(phase, require_full=True)
            output_desc = OutputTilingDesc(start_indices=(tile_start, ), tile_sizes=(tile_size, ))
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params,
                                 inputs_dep={
                                     scratch: scratch_desc,
                                     signal: signal_desc,
                                     phase: phase_desc,
                                 },
                                 outs_tile_mapping={output: output_desc}))
        return tasks
