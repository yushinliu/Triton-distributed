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
"""Triton device facade for NVSHMEM IBGDA remote operations.

The functions in this module intentionally bind to the nvshmem wrapper symbols
from tools/compile/nvshmem_wrapper.cu. Referencing wrapper symbols makes the
Triton post-compile hook link the source-built NVSHMEM device cubin, which is
required for NVSHMEM IBGDA support.
"""

from triton.language import core
import triton.language as tl

from triton_dist.language.core import extern_call

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))
pi_i64_t = tl.core.pointer_type(tl.core.dtype("int64"))
void_ptr = core.pointer_type(core.void)

NVSHMEM_CMP_EQ = 0
NVSHMEM_CMP_NE = 1
NVSHMEM_CMP_GT = 2
NVSHMEM_CMP_LE = 3
NVSHMEM_CMP_LT = 4
NVSHMEM_CMP_GE = 5
NVSHMEM_SIGNAL_SET = 9
NVSHMEM_SIGNAL_ADD = 10


def _scope_prefix(scope_suffix: core.constexpr):
    return "nvshmemx" if scope_suffix.value else "nvshmem"


@core.extern
def _putmem_impl(dest, source, nbytes, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""),
                 _semantic=None):
    prefix = _scope_prefix(SCOPE_SUFFIX)
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, void_ptr, _semantic=_semantic),
            tl.cast(source, void_ptr, _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (void_ptr, void_ptr, tl.uint64, tl.int32): (
                f"{prefix}_putmem{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def putmem(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_nbi(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def putmem_warp(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_nbi_warp(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr("_nbi"),
                        _semantic=_semantic)


@core.extern
def putmem_block(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_block"), core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_nbi_block(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_block"), core.constexpr("_nbi"),
                        _semantic=_semantic)


@core.extern
def _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, SCOPE_SUFFIX: core.constexpr,
                        NBI: core.constexpr = core.constexpr(""), _semantic=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t or sig_addr.dtype == pi_i64_t,
                     "sig_addr should be a pointer of uint64_t/int64_t", _semantic=_semantic)
    prefix = _scope_prefix(SCOPE_SUFFIX)
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, void_ptr, _semantic=_semantic),
            tl.cast(source, void_ptr, _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(sig_addr, pi_u64_t, _semantic=_semantic),
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (void_ptr, void_ptr, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                f"{prefix}_putmem_signal{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def putmem_signal(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr(""),
                               core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_signal_nbi(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr(""),
                               core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def putmem_signal_warp(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_warp"),
                               core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_signal_nbi_warp(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_warp"),
                               core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def putmem_signal_block(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_block"),
                               core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_signal_nbi_block(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, core.constexpr("_block"),
                               core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t or sig_addr.dtype == pi_i64_t,
                     "sig_addr should be a pointer of uint64_t/int64_t", _semantic=_semantic)
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(sig_addr, pi_u64_t, _semantic=_semantic),
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "nvshmemx_signal_op_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t or sig_addr.dtype == pi_i64_t,
                     "sig_addr should be a pointer of uint64_t/int64_t", _semantic=_semantic)
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(sig_addr, pi_u64_t, _semantic=_semantic),
            tl.cast(cmp_, tl.int32, _semantic=_semantic),
            tl.cast(cmp_val, tl.uint64, _semantic=_semantic),
        ],
        {
            (pi_u64_t, tl.int32, tl.uint64): (
                "nvshmem_signal_wait_until_wrapper",
                tl.uint64,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def quiet(_semantic=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_quiet_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fence(_semantic=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_fence_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
