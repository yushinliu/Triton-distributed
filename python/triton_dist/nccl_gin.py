################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

_DEV_COMM_TENSOR: Optional[torch.Tensor] = None


def _backend():
    from triton._C.libtriton_distributed import distributed
    return distributed.nccl_gin


_DLPACK_DTYPES = {
    torch.int8: (0, 8, 1),
    torch.int16: (0, 16, 1),
    torch.int32: (0, 32, 1),
    torch.int64: (0, 64, 1),
    torch.uint8: (1, 8, 1),
    torch.float16: (2, 16, 1),
    torch.float32: (2, 32, 1),
    torch.float64: (2, 64, 1),
    torch.bfloat16: (4, 16, 1),
}


def empty(shape, dtype: torch.dtype = torch.float32, device: Optional[torch.device | str | int] = None) -> torch.Tensor:
    if dtype not in _DLPACK_DTYPES:
        raise TypeError(f"Unsupported NCCL GIN dtype: {dtype}")
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(dim) for dim in shape)
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if device.type != "cuda":
        raise ValueError("NCCL GIN allocations require a CUDA device")
    code, bits, lanes = _DLPACK_DTYPES[dtype]
    with torch.cuda.device(device):
        capsule = _backend().empty_dlpack(list(shape), code, bits, lanes)
        return torch.utils.dlpack.from_dlpack(capsule)


def mem_alloc(nbytes: int) -> int:
    return int(_backend().mem_alloc(int(nbytes)))


def mem_free(data_ptr: int) -> None:
    _backend().mem_free(int(data_ptr))


def _broadcast_unique_id(pg: dist.ProcessGroup) -> bytes:
    backend = _backend()
    rank = pg.rank()
    unique_id = backend.get_unique_id() if rank == 0 else None
    objects = [unique_id]
    dist.broadcast_object_list(objects, src=dist.get_global_rank(pg, 0), group=pg)
    if objects[0] is None:
        raise RuntimeError("Failed to broadcast NCCL GIN unique id")
    return objects[0]


def probe_by_torch_process_group(pg: dist.ProcessGroup) -> dict:
    unique_id = _broadcast_unique_id(pg)
    torch.cuda.synchronize()
    props = _backend().probe(
        unique_id,
        pg.rank(),
        pg.size(),
        torch.cuda.current_device(),
    )
    dist.barrier(group=pg)
    torch.cuda.synchronize()
    return dict(props)


def init_by_torch_process_group(
    pg: dist.ProcessGroup,
    barrier_count: int = 0,
    gin_signal_count: int = 0,
    gin_context_count: int = 4,
) -> None:
    backend = _backend()
    if backend.is_initialized():
        raise RuntimeError("NCCL GIN has already been initialized")

    rank = pg.rank()
    world_size = pg.size()
    unique_id = _broadcast_unique_id(pg)

    torch.cuda.synchronize()
    backend.init(
        unique_id,
        rank,
        world_size,
        torch.cuda.current_device(),
        barrier_count,
        gin_signal_count,
        gin_context_count,
    )
    dist.barrier(group=pg)
    torch.cuda.synchronize()


def is_initialized() -> bool:
    try:
        return bool(_backend().is_initialized())
    except Exception:
        return False


def get_dev_comm_tensor(device: Optional[torch.device | str | int] = None) -> torch.Tensor:
    global _DEV_COMM_TENSOR
    if not is_initialized():
        raise RuntimeError("NCCL GIN has not been initialized")
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    if _DEV_COMM_TENSOR is None or _DEV_COMM_TENSOR.device != device:
        data = bytes(_backend().dev_comm_bytes())
        _DEV_COMM_TENSOR = torch.tensor(list(data), dtype=torch.uint8, device=device)
    return _DEV_COMM_TENSOR


@dataclass
class NCCLGinWindow:
    tensor: torch.Tensor
    collective_symmetric: bool = True
    strict_ordering: bool = False
    handle: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.tensor.is_cuda:
            raise ValueError("NCCL GIN windows require CUDA tensors")
        if not self.tensor.is_contiguous():
            raise ValueError("NCCL GIN windows require contiguous tensors")
        nbytes = self.tensor.numel() * self.tensor.element_size()
        self.handle = int(
            _backend().register_window(
                int(self.tensor.data_ptr()),
                int(nbytes),
                self.collective_symmetric,
                self.strict_ordering,
            )
        )

    def close(self) -> None:
        if self.handle is not None and is_initialized():
            _backend().deregister_window(int(self.handle))
        self.handle = None

    def __int__(self) -> int:
        if self.handle is None:
            raise RuntimeError("NCCL GIN window has been closed")
        return int(self.handle)

    def __index__(self) -> int:
        return int(self)

    def __enter__(self) -> "NCCLGinWindow":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def register_window(
    tensor: torch.Tensor,
    collective_symmetric: bool = True,
    strict_ordering: bool = False,
) -> NCCLGinWindow:
    return NCCLGinWindow(
        tensor=tensor,
        collective_symmetric=collective_symmetric,
        strict_ordering=strict_ordering,
    )


def properties():
    return dict(_backend().properties())


def finalize() -> None:
    global _DEV_COMM_TENSOR
    _DEV_COMM_TENSOR = None
    if is_initialized():
        _backend().finalize()
