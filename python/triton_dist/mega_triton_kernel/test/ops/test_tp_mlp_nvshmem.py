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
import subprocess
from types import SimpleNamespace

import torch
import torch.distributed as dist

from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.mega_triton_kernel.models.layers import TPMLPBuilder
from triton_dist.utils import finalize_distributed, initialize_distributed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fuse_fc1_silu", default=False, action="store_true")
    parser.add_argument("--perf", default=False, action="store_true", help="benchmark torch+cudagraph and mega+cudagraph")
    parser.add_argument("--bench_warmup", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=100)
    parser.add_argument("--check_atol", type=float, default=5e-2)
    parser.add_argument("--check_rtol", type=float, default=5e-2)
    return parser.parse_args()


def check_args(args, world_size, local_world_size):
    if world_size <= 1:
        raise ValueError("TP MLP NVSHMEM test requires at least 2 ranks")
    if world_size != local_world_size:
        raise ValueError(
            "TP MLP NVSHMEM test only supports intra-node runs, "
            f"got WORLD_SIZE={world_size}, LOCAL_WORLD_SIZE={local_world_size}"
        )
    if args.iters <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")
    if args.hidden_size % 32 != 0:
        raise ValueError(f"--hidden_size must be divisible by 32, got {args.hidden_size}")
    if args.intermediate_size % world_size != 0:
        raise ValueError(
            f"--intermediate_size must be divisible by WORLD_SIZE={world_size}, got {args.intermediate_size}"
        )
    if args.perf and world_size not in (4, 8):
        raise ValueError(f"--perf currently supports 4-card and 8-card runs, got WORLD_SIZE={world_size}")
    if args.perf and args.bench_warmup <= 0:
        raise ValueError(f"--bench_warmup must be positive, got {args.bench_warmup}")
    if args.perf and args.bench_iters <= 0:
        raise ValueError(f"--bench_iters must be positive, got {args.bench_iters}")
    if args.check_atol < 0:
        raise ValueError(f"--check_atol must be non-negative, got {args.check_atol}")
    if args.check_rtol < 0:
        raise ValueError(f"--check_rtol must be non-negative, got {args.check_rtol}")


def broadcast_randn(shape, dtype, group, scale):
    tensor = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
    if dist.get_rank(group) == 0:
        tensor.normal_(mean=0.0, std=scale)
    dist.broadcast(tensor, src=0, group=group)
    return tensor


def make_mlp(gate_weight, up_weight, down_weight):
    gate_proj = torch.nn.Linear(gate_weight.shape[1], gate_weight.shape[0], bias=False, dtype=gate_weight.dtype,
                                device=gate_weight.device)
    up_proj = torch.nn.Linear(up_weight.shape[1], up_weight.shape[0], bias=False, dtype=up_weight.dtype,
                              device=up_weight.device)
    down_proj = torch.nn.Linear(down_weight.shape[1], down_weight.shape[0], bias=False, dtype=down_weight.dtype,
                                device=down_weight.device)
    gate_proj.weight.data.copy_(gate_weight)
    up_proj.weight.data.copy_(up_weight)
    down_proj.weight.data.copy_(down_weight)
    return SimpleNamespace(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj, act_fn=torch.nn.SiLU())


def torch_tp_mlp_ref(x, gate_up_proj, down_proj, group):
    x_2d = x.reshape(-1, x.shape[-1])
    fused = torch.nn.functional.linear(x_2d, gate_up_proj)
    gate, up = torch.chunk(fused, 2, dim=-1)
    out = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, down_proj)
    dist.all_reduce(out, group=group)
    return out.reshape_as(x)


def assert_uses_nvshmem_split_allreduce(builder):
    task_names = {type(task).__name__ for task in builder.megakernel_tasks}
    expected = {"AllReduceNVSHMEMTask", "AllReduceNVSHMEMPushTask"}
    missing = expected - task_names
    if missing:
        raise AssertionError(f"missing NVSHMEM split allreduce tasks: {sorted(missing)}")


def run_cmd(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def get_gpu_index_by_uuid():
    output = run_cmd(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    index_by_uuid = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid = [item.strip() for item in line.split(",", 1)]
        index_by_uuid[uuid] = int(index)
    return index_by_uuid


def get_benchmark_gpu_indices(local_world_size):
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        return set(range(local_world_size))

    visible_devices = [dev.strip() for dev in cuda_visible_devices.split(",") if dev.strip()]
    if len(visible_devices) < local_world_size:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES exposes {len(visible_devices)} devices, "
            f"but LOCAL_WORLD_SIZE={local_world_size}"
        )

    index_by_uuid = get_gpu_index_by_uuid()
    gpu_indices = set()
    for dev in visible_devices[:local_world_size]:
        if dev.isdigit():
            gpu_indices.add(int(dev))
        elif dev in index_by_uuid:
            gpu_indices.add(index_by_uuid[dev])
        else:
            raise RuntimeError(f"Unsupported CUDA_VISIBLE_DEVICES entry for GPU idle check: {dev}")
    return gpu_indices


def get_compute_apps_by_gpu():
    try:
        output = run_cmd([
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
    except subprocess.CalledProcessError as e:
        if "No running processes found" in e.stderr:
            return {}
        raise

    index_by_uuid = get_gpu_index_by_uuid()
    apps_by_gpu = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        gpu_uuid, pid, process_name, used_gpu_memory = [item.strip() for item in line.split(",", 3)]
        if gpu_uuid not in index_by_uuid:
            continue
        gpu_index = index_by_uuid[gpu_uuid]
        apps_by_gpu.setdefault(gpu_index, []).append((int(pid), process_name, used_gpu_memory))
    return apps_by_gpu


def assert_no_other_gpu_processes(group, local_world_size):
    local_info = (dist.get_rank(group), int(os.environ.get("LOCAL_RANK", 0)), os.getpid())
    gathered_infos = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered_infos, local_info, group=group)
    allowed_pids = {pid for _, _, pid in gathered_infos}

    error_message = None
    if dist.get_rank(group) == 0:
        benchmark_gpu_indices = get_benchmark_gpu_indices(local_world_size)
        apps_by_gpu = get_compute_apps_by_gpu()
        offenders = []
        for gpu_index in sorted(benchmark_gpu_indices):
            for pid, process_name, used_gpu_memory in apps_by_gpu.get(gpu_index, []):
                if pid not in allowed_pids:
                    offenders.append((gpu_index, pid, process_name, used_gpu_memory))
        if offenders:
            offender_lines = "\n".join(
                f"GPU {gpu}: pid={pid}, process={process_name}, used_gpu_memory={memory} MiB"
                for gpu, pid, process_name, memory in offenders
            )
            error_message = (
                "Refusing to run performance benchmark because benchmark GPUs have other compute processes:\n"
                f"{offender_lines}"
            )

    messages = [error_message]
    dist.broadcast_object_list(messages, src=0, group=group)
    if messages[0] is not None:
        raise RuntimeError(messages[0])
    dist.barrier(group=group)


def make_cuda_graph(func, warmup_iters):
    for _ in range(warmup_iters):
        func()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        func()
    torch.cuda.synchronize()
    return graph


def benchmark_cuda_graph(graph, iters):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(iters):
        graph.replay()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / iters


def gather_rank_times(local_ms, group):
    rank_times = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(rank_times, float(local_ms), group=group)
    return rank_times


def run_perf_benchmark(args, x, out, builder, tp_mlp, group, rank, local_world_size):
    assert_no_other_gpu_processes(group, local_world_size)

    bench_input = broadcast_randn(x.shape, x.dtype, group, scale=0.2)
    x.copy_(bench_input)
    torch_graph_out = torch.empty_like(out)

    dist.barrier(group=group)
    mega_graph = make_cuda_graph(builder.run, args.bench_warmup)
    torch_graph = make_cuda_graph(
        lambda: torch_graph_out.copy_(torch_tp_mlp_ref(x, tp_mlp.gate_up_proj, tp_mlp.down_proj, group)),
        args.bench_warmup,
    )
    dist.barrier(group=group)

    mega_ms = benchmark_cuda_graph(mega_graph, args.bench_iters)
    dist.barrier(group=group)
    torch_ms = benchmark_cuda_graph(torch_graph, args.bench_iters)
    dist.barrier(group=group)

    mega_rank_times = gather_rank_times(mega_ms, group)
    torch_rank_times = gather_rank_times(torch_ms, group)
    if rank == 0:
        mega_e2e_ms = max(mega_rank_times)
        torch_e2e_ms = max(torch_rank_times)
        print(
            "TP MLP perf cudagraph: "
            f"world_size={dist.get_world_size(group)}, shape={tuple(x.shape)}, "
            f"warmup={args.bench_warmup}, iters={args.bench_iters}, "
            f"mega_nvshmem_ms={mega_e2e_ms:.4f}, torch_ms={torch_e2e_ms:.4f}, "
            f"speedup={torch_e2e_ms / mega_e2e_ms:.3f}x"
        )
        print(f"  mega_rank_ms={','.join(f'{ms:.4f}' for ms in mega_rank_times)}")
        print(f"  torch_rank_ms={','.join(f'{ms:.4f}' for ms in torch_rank_times)}")


if __name__ == "__main__":
    args = parse_args()

    TP_GROUP = initialize_distributed(seed=args.seed)
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    check_args(args, WORLD_SIZE, LOCAL_WORLD_SIZE)

    dtype = torch.bfloat16
    gate_weight = broadcast_randn((args.intermediate_size, args.hidden_size), dtype, TP_GROUP, scale=0.02)
    up_weight = broadcast_randn((args.intermediate_size, args.hidden_size), dtype, TP_GROUP, scale=0.02)
    down_weight = broadcast_randn((args.hidden_size, args.intermediate_size), dtype, TP_GROUP, scale=0.02)
    mlp = make_mlp(gate_weight, up_weight, down_weight)

    builder = ModelBuilder(
        rank=RANK,
        world_size=WORLD_SIZE,
        local_world_size=LOCAL_WORLD_SIZE,
        enable_mlp_fc1_silu_fusion=args.fuse_fc1_silu,
    )
    tp_mlp = TPMLPBuilder(builder=builder, rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP)
    tp_mlp._init_parameters(mlp)

    x = torch.empty((args.batch, args.seq_len, args.hidden_size), dtype=dtype, device=torch.cuda.current_device())
    out = tp_mlp.build_fwd(x, allreduce_impl="nvshmem")
    assert_uses_nvshmem_split_allreduce(builder)
    builder.compile()

    for iter_idx in range(args.iters):
        local_input = broadcast_randn(x.shape, dtype, TP_GROUP, scale=0.2)
        x.copy_(local_input)
        builder.run()
        mega_out = out.clone()
        torch_out = torch_tp_mlp_ref(local_input, tp_mlp.gate_up_proj, tp_mlp.down_proj, TP_GROUP)
        try:
            torch.testing.assert_close(torch_out, mega_out, atol=args.check_atol, rtol=args.check_rtol)
        except Exception as e:
            print(f"RANK = {RANK}, iteration {iter_idx} failed with {e}")
            raise

    if RANK == 0:
        print(
            "TP MLP NVSHMEM split allreduce passed: "
            f"world_size={WORLD_SIZE}, shape={tuple(x.shape)}, iters={args.iters}, "
            f"fuse_fc1_silu={args.fuse_fc1_silu}, "
            f"check_atol={args.check_atol}, check_rtol={args.check_rtol}"
        )

    if args.perf:
        run_perf_benchmark(args, x, out, builder, tp_mlp, TP_GROUP, RANK, LOCAL_WORLD_SIZE)

    builder.finalize()
    finalize_distributed()
