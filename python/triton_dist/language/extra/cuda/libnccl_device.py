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

from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call

void_ptr = core.pointer_type(core.void)


def _u64(x, _semantic):
    return tl.cast(x, tl.uint64, _semantic=_semantic)


def _i32(x, _semantic):
    return tl.cast(x, tl.int32, _semantic=_semantic)


@core.extern
def gin_read_signal(dev_comm, context, signal, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [tl.cast(dev_comm, void_ptr, _semantic=_semantic), _i32(context, _semantic), _i32(signal, _semantic)],
        {
            (void_ptr, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_read_signal_wrapper",
                tl.uint64,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_read_va_signal(dev_comm, signal_win, signal_offset, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.int32): (
                "triton_dist_nccl_gin_read_va_signal_wrapper",
                tl.uint64,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_put_cta(dev_comm, dst_win, dst_offset, src_win, src_offset, nbytes, peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(dst_win, _semantic),
            _u64(dst_offset, _semantic),
            _u64(src_win, _semantic),
            _u64(src_offset, _semantic),
            _u64(nbytes, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_put_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_put_warp(dev_comm, dst_win, dst_offset, src_win, src_offset, nbytes, peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(dst_win, _semantic),
            _u64(dst_offset, _semantic),
            _u64(src_win, _semantic),
            _u64(src_offset, _semantic),
            _u64(nbytes, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_put_warp_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_put_thread(dev_comm, dst_win, dst_offset, src_win, src_offset, nbytes, peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(dst_win, _semantic),
            _u64(dst_offset, _semantic),
            _u64(src_win, _semantic),
            _u64(src_offset, _semantic),
            _u64(nbytes, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_put_thread_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_get_cta(dev_comm, remote_win, remote_offset, local_win, local_offset, nbytes, peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(remote_win, _semantic),
            _u64(remote_offset, _semantic),
            _u64(local_win, _semantic),
            _u64(local_offset, _semantic),
            _u64(nbytes, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_get_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_put_signal_inc_cta(dev_comm, dst_win, dst_offset, src_win, src_offset, nbytes, peer, context, signal,
                           _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(dst_win, _semantic),
            _u64(dst_offset, _semantic),
            _u64(src_win, _semantic),
            _u64(src_offset, _semantic),
            _u64(nbytes, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
            _i32(signal, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.int32, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_put_signal_inc_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_put_va_signal_inc_cta(dev_comm, dst_win, dst_offset, src_win, src_offset, nbytes, signal_win, signal_offset,
                              peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(dst_win, _semantic),
            _u64(dst_offset, _semantic),
            _u64(src_win, _semantic),
            _u64(src_offset, _semantic),
            _u64(nbytes, _semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.int32,
             tl.int32): (
                 "triton_dist_nccl_gin_put_va_signal_inc_cta_wrapper",
                 (),
             ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_put_va_signal_inc_warp(dev_comm, dst_win, dst_offset, src_win, src_offset, nbytes, signal_win, signal_offset,
                               peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(dst_win, _semantic),
            _u64(dst_offset, _semantic),
            _u64(src_win, _semantic),
            _u64(src_offset, _semantic),
            _u64(nbytes, _semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.uint64, tl.int32,
             tl.int32): (
                 "triton_dist_nccl_gin_put_va_signal_inc_warp_wrapper",
                 (),
             ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_signal_inc_cta(dev_comm, peer, context, signal, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
            _i32(signal, _semantic),
        ],
        {
            (void_ptr, tl.int32, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_signal_inc_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_wait_signal_cta(dev_comm, context, signal, least, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _i32(context, _semantic),
            _i32(signal, _semantic),
            _u64(least, _semantic),
        ],
        {
            (void_ptr, tl.int32, tl.int32, tl.uint64): (
                "triton_dist_nccl_gin_wait_signal_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_wait_va_signal_cta(dev_comm, signal_win, signal_offset, least, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _u64(least, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.uint64, tl.int32): (
                "triton_dist_nccl_gin_wait_va_signal_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_signal_va_inc_cta(dev_comm, signal_win, signal_offset, peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_signal_va_inc_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_signal_va_inc_warp(dev_comm, signal_win, signal_offset, peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_signal_va_inc_warp_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_signal_va_inc_thread(dev_comm, signal_win, signal_offset, peer, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _i32(peer, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_signal_va_inc_thread_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_reset_signal_cta(dev_comm, context, signal, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [tl.cast(dev_comm, void_ptr, _semantic=_semantic), _i32(context, _semantic), _i32(signal, _semantic)],
        {
            (void_ptr, tl.int32, tl.int32): (
                "triton_dist_nccl_gin_reset_signal_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_reset_va_signal_cta(dev_comm, signal_win, signal_offset, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [
            tl.cast(dev_comm, void_ptr, _semantic=_semantic),
            _u64(signal_win, _semantic),
            _u64(signal_offset, _semantic),
            _i32(context, _semantic),
        ],
        {
            (void_ptr, tl.uint64, tl.uint64, tl.int32): (
                "triton_dist_nccl_gin_reset_va_signal_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_flush_cta(dev_comm, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [tl.cast(dev_comm, void_ptr, _semantic=_semantic), _i32(context, _semantic)],
        {
            (void_ptr, tl.int32): (
                "triton_dist_nccl_gin_flush_cta_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_flush_warp(dev_comm, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [tl.cast(dev_comm, void_ptr, _semantic=_semantic), _i32(context, _semantic)],
        {
            (void_ptr, tl.int32): (
                "triton_dist_nccl_gin_flush_warp_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def gin_flush_thread(dev_comm, context, _semantic=None):
    return extern_call(
        "libnccl_device",
        "",
        [tl.cast(dev_comm, void_ptr, _semantic=_semantic), _i32(context, _semantic)],
        {
            (void_ptr, tl.int32): (
                "triton_dist_nccl_gin_flush_thread_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )
