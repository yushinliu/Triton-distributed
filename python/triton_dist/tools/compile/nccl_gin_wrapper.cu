/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <nccl_device.h>

extern "C" {

__device__ uint64_t triton_dist_nccl_gin_read_signal_wrapper(
    void *dev_comm_ptr, int context, int signal) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclGin gin{*dev_comm, context};
  return gin.readSignal(static_cast<ncclGinSignal_t>(signal));
}

__device__ uint64_t triton_dist_nccl_gin_read_va_signal_wrapper(
    void *dev_comm_ptr, unsigned long long signal_win_raw,
    unsigned long long signal_offset, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  return gin.readSignal(signal_win, static_cast<size_t>(signal_offset));
}

__device__ void triton_dist_nccl_gin_put_cta_wrapper(
    void *dev_comm_ptr, unsigned long long dst_win_raw,
    unsigned long long dst_offset, unsigned long long src_win_raw,
    unsigned long long src_offset, unsigned long long bytes, int peer,
    int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t dst_win = reinterpret_cast<ncclWindow_t>(dst_win_raw);
  ncclWindow_t src_win = reinterpret_cast<ncclWindow_t>(src_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.put(ncclTeamWorld(*dev_comm), peer, dst_win, static_cast<size_t>(dst_offset),
          src_win, static_cast<size_t>(src_offset), static_cast<size_t>(bytes),
          ncclGin_None{}, ncclGin_None{}, ncclCoopCta{});
}

__device__ void triton_dist_nccl_gin_put_warp_wrapper(
    void *dev_comm_ptr, unsigned long long dst_win_raw,
    unsigned long long dst_offset, unsigned long long src_win_raw,
    unsigned long long src_offset, unsigned long long bytes, int peer,
    int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t dst_win = reinterpret_cast<ncclWindow_t>(dst_win_raw);
  ncclWindow_t src_win = reinterpret_cast<ncclWindow_t>(src_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.put(ncclTeamWorld(*dev_comm), peer, dst_win, static_cast<size_t>(dst_offset),
          src_win, static_cast<size_t>(src_offset), static_cast<size_t>(bytes),
          ncclGin_None{}, ncclGin_None{}, ncclCoopWarp{});
}

__device__ void triton_dist_nccl_gin_put_thread_wrapper(
    void *dev_comm_ptr, unsigned long long dst_win_raw,
    unsigned long long dst_offset, unsigned long long src_win_raw,
    unsigned long long src_offset, unsigned long long bytes, int peer,
    int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t dst_win = reinterpret_cast<ncclWindow_t>(dst_win_raw);
  ncclWindow_t src_win = reinterpret_cast<ncclWindow_t>(src_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.put(ncclTeamWorld(*dev_comm), peer, dst_win, static_cast<size_t>(dst_offset),
          src_win, static_cast<size_t>(src_offset), static_cast<size_t>(bytes),
          ncclGin_None{}, ncclGin_None{}, ncclCoopThread{});
}

__device__ void triton_dist_nccl_gin_get_cta_wrapper(
    void *dev_comm_ptr, unsigned long long remote_win_raw,
    unsigned long long remote_offset, unsigned long long local_win_raw,
    unsigned long long local_offset, unsigned long long bytes, int peer,
    int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t remote_win = reinterpret_cast<ncclWindow_t>(remote_win_raw);
  ncclWindow_t local_win = reinterpret_cast<ncclWindow_t>(local_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.get(ncclTeamWorld(*dev_comm), peer, remote_win,
          static_cast<size_t>(remote_offset), local_win,
          static_cast<size_t>(local_offset), static_cast<size_t>(bytes),
          ncclCoopCta{});
}

__device__ void triton_dist_nccl_gin_put_signal_inc_cta_wrapper(
    void *dev_comm_ptr, unsigned long long dst_win_raw,
    unsigned long long dst_offset, unsigned long long src_win_raw,
    unsigned long long src_offset, unsigned long long bytes, int peer,
    int context, int signal) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t dst_win = reinterpret_cast<ncclWindow_t>(dst_win_raw);
  ncclWindow_t src_win = reinterpret_cast<ncclWindow_t>(src_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.put(ncclTeamWorld(*dev_comm), peer, dst_win, static_cast<size_t>(dst_offset),
          src_win, static_cast<size_t>(src_offset), static_cast<size_t>(bytes),
          ncclGin_SignalInc{static_cast<ncclGinSignal_t>(signal)}, ncclGin_None{},
          ncclCoopCta{});
}

__device__ void triton_dist_nccl_gin_put_va_signal_inc_cta_wrapper(
    void *dev_comm_ptr, unsigned long long dst_win_raw,
    unsigned long long dst_offset, unsigned long long src_win_raw,
    unsigned long long src_offset, unsigned long long bytes,
    unsigned long long signal_win_raw, unsigned long long signal_offset,
    int peer, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t dst_win = reinterpret_cast<ncclWindow_t>(dst_win_raw);
  ncclWindow_t src_win = reinterpret_cast<ncclWindow_t>(src_win_raw);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.put(ncclTeamWorld(*dev_comm), peer, dst_win, static_cast<size_t>(dst_offset),
          src_win, static_cast<size_t>(src_offset), static_cast<size_t>(bytes),
          ncclGin_VASignalInc{signal_win, static_cast<size_t>(signal_offset)},
          ncclGin_None{}, ncclCoopCta{});
}

__device__ void triton_dist_nccl_gin_put_va_signal_inc_warp_wrapper(
    void *dev_comm_ptr, unsigned long long dst_win_raw,
    unsigned long long dst_offset, unsigned long long src_win_raw,
    unsigned long long src_offset, unsigned long long bytes,
    unsigned long long signal_win_raw, unsigned long long signal_offset,
    int peer, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t dst_win = reinterpret_cast<ncclWindow_t>(dst_win_raw);
  ncclWindow_t src_win = reinterpret_cast<ncclWindow_t>(src_win_raw);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.put(ncclTeamWorld(*dev_comm), peer, dst_win, static_cast<size_t>(dst_offset),
          src_win, static_cast<size_t>(src_offset), static_cast<size_t>(bytes),
          ncclGin_VASignalInc{signal_win, static_cast<size_t>(signal_offset)},
          ncclGin_None{}, ncclCoopWarp{});
}

__device__ void triton_dist_nccl_gin_signal_inc_cta_wrapper(
    void *dev_comm_ptr, int peer, int context, int signal) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclGin gin{*dev_comm, context};
  gin.signal(ncclTeamWorld(*dev_comm), peer,
             ncclGin_SignalInc{static_cast<ncclGinSignal_t>(signal)},
             ncclCoopCta{});
}

__device__ void triton_dist_nccl_gin_wait_signal_cta_wrapper(
    void *dev_comm_ptr, int context, int signal, unsigned long long least) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclGin gin{*dev_comm, context};
  gin.waitSignal(ncclCoopCta{}, static_cast<ncclGinSignal_t>(signal),
                 static_cast<uint64_t>(least));
}

__device__ void triton_dist_nccl_gin_wait_va_signal_cta_wrapper(
    void *dev_comm_ptr, unsigned long long signal_win_raw,
    unsigned long long signal_offset, unsigned long long least, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.waitSignal(ncclCoopCta{}, signal_win, static_cast<size_t>(signal_offset),
                 static_cast<uint64_t>(least));
}

__device__ void triton_dist_nccl_gin_signal_va_inc_cta_wrapper(
    void *dev_comm_ptr, unsigned long long signal_win_raw,
    unsigned long long signal_offset, int peer, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.signal(ncclTeamWorld(*dev_comm), peer,
             ncclGin_VASignalInc{signal_win, static_cast<size_t>(signal_offset)},
             ncclCoopCta{});
}

__device__ void triton_dist_nccl_gin_signal_va_inc_warp_wrapper(
    void *dev_comm_ptr, unsigned long long signal_win_raw,
    unsigned long long signal_offset, int peer, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.signal(ncclTeamWorld(*dev_comm), peer,
             ncclGin_VASignalInc{signal_win, static_cast<size_t>(signal_offset)},
             ncclCoopWarp{});
}

__device__ void triton_dist_nccl_gin_signal_va_inc_thread_wrapper(
    void *dev_comm_ptr, unsigned long long signal_win_raw,
    unsigned long long signal_offset, int peer, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.signal(ncclTeamWorld(*dev_comm), peer,
             ncclGin_VASignalInc{signal_win, static_cast<size_t>(signal_offset)},
             ncclCoopThread{});
}

__device__ void triton_dist_nccl_gin_reset_signal_cta_wrapper(
    void *dev_comm_ptr, int context, int signal) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclGin gin{*dev_comm, context};
  gin.resetSignal(static_cast<ncclGinSignal_t>(signal));
}

__device__ void triton_dist_nccl_gin_reset_va_signal_cta_wrapper(
    void *dev_comm_ptr, unsigned long long signal_win_raw,
    unsigned long long signal_offset, int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclWindow_t signal_win = reinterpret_cast<ncclWindow_t>(signal_win_raw);
  ncclGin gin{*dev_comm, context};
  gin.resetSignal(signal_win, static_cast<size_t>(signal_offset));
}

__device__ void triton_dist_nccl_gin_flush_cta_wrapper(void *dev_comm_ptr,
                                                       int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclGin gin{*dev_comm, context};
  gin.flush(ncclCoopCta{});
}

__device__ void triton_dist_nccl_gin_flush_warp_wrapper(void *dev_comm_ptr,
                                                        int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclGin gin{*dev_comm, context};
  gin.flush(ncclCoopWarp{});
}

__device__ void triton_dist_nccl_gin_flush_thread_wrapper(void *dev_comm_ptr,
                                                          int context) {
  ncclDevComm const *dev_comm =
      reinterpret_cast<ncclDevComm const *>(dev_comm_ptr);
  ncclGin gin{*dev_comm, context};
  gin.flush(ncclCoopThread{});
}

}
