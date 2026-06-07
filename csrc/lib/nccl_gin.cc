#ifdef TRITON_DIST_BUILD_NCCL_GIN_PLUGIN
/*
 * Minimal intra-node NCCL GIN proxy plugin for Triton-distributed.
 * It implements the NCCL GIN v13 ABI and services device GIN proxy queues
 * with CUDA IPC copies between local ranks.
 */

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device/net_device.h>
#include <nccl_device/gin/proxy/gin_proxy_device_host_common.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

constexpr uint32_t kHandleMagic = 0x54444749;
constexpr int kQueueSizeDefault = 1024;
constexpr size_t kMaxNetSize = 1024ULL * 1024ULL * 1024ULL;
constexpr int NCCL_PTR_HOST_COMPAT = 0x1;
constexpr int NCCL_PTR_CUDA_COMPAT = 0x2;
constexpr int NCCL_PTR_DMABUF_COMPAT = 0x4;
constexpr int NCCL_NET_MR_FLAG_FORCE_SO_COMPAT = 1 << 0;
constexpr int NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET_COMPAT = 1 << 1;

using ncclDebugLogger_t = void (*)(int level, unsigned long flags, const char *file, int line, const char *fmt, ...);

struct ncclNetVDeviceProps_v12_t {
  int ndevs;
  int devs[8];
};

struct ncclNetProperties_v12_t {
  char *name;
  char *pciPath;
  uint64_t guid;
  int ptrSupport;
  int regIsGlobal;
  int forceFlush;
  int speed;
  int port;
  float latency;
  int maxComms;
  int maxRecvs;
  ncclNetDeviceType netDeviceType;
  int netDeviceVersion;
  ncclNetVDeviceProps_v12_t vProps;
  size_t maxP2pBytes;
  size_t maxCollBytes;
  int maxMultiRequestSize;
  int16_t railId;
  int16_t planeId;
};

struct ncclGinConfig_v13_t {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
};

struct ncclGin_v13_t {
  const char *name;
  ncclResult_t (*init)(void **ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int *ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t *props);
  ncclResult_t (*listen)(void *ctx, int dev, void *handle, void **listenComm);
  ncclResult_t (*connect)(void *ctx, void *handles[], int nranks, int rank, void *listenComm, void **collComm);
  ncclResult_t (*createContext)(void *collComm, ncclGinConfig_v13_t *config, void **ginCtx,
                                ncclNetDeviceHandle_v11_t **devHandle);
  ncclResult_t (*regMrSym)(void *collComm, void *data, size_t size, int type, uint64_t mrFlags, void **mhandle,
                           void **ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void *collComm, void *data, size_t size, int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void **mhandle, void **ginHandle);
  ncclResult_t (*deregMrSym)(void *collComm, void *mhandle);
  ncclResult_t (*destroyContext)(void *ginCtx);
  ncclResult_t (*closeColl)(void *collComm);
  ncclResult_t (*closeListen)(void *listenComm);
  ncclResult_t (*iput)(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size, uint64_t dstOff,
                       void *dstMhandle, uint32_t rank, void **request);
  ncclResult_t (*iputSignal)(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size,
                             uint64_t dstOff, void *dstMhandle, uint32_t rank, uint64_t signalOff,
                             void *signalMhandle, uint64_t signalValue, uint32_t signalOp, void **request);
  ncclResult_t (*iget)(void *ginCtx, int context, uint64_t remoteOff, void *remoteMhandle, size_t size,
                       uint64_t localOff, void *localMhandle, uint32_t rank, void **request);
  ncclResult_t (*iflush)(void *ginCtx, int context, void *mhandle, uint32_t rank, void **request);
  ncclResult_t (*test)(void *collComm, void *request, int *done);
  ncclResult_t (*ginProgress)(void *ginCtx);
  ncclResult_t (*queryLastError)(void *ginCtx, bool *hasError);
  ncclResult_t (*finalize)(void *ctx);
};

struct PluginCtx { uint64_t commId = 0; };
struct WireHandle { uint32_t magic; char path[108]; };
struct ListenComm { int fd = -1; char path[108] = {}; };
struct CollComm { int nranks = 0; int rank = 0; int cudaDev = 0; std::vector<int> sockets; std::mutex mutex; };
struct WireMemInfo { int type; uint64_t base; uint64_t size; int ipcValid; int dmaBufValid; int dmaBufFd; int dmaBufPid; uint64_t dmaBufOffset; uint64_t dmaBufMapSize; cudaIpcMemHandle_t ipc; };
struct MemHandle {
  int type = 0;
  size_t size = 0;
  void *local = nullptr;
  std::vector<void *> ptrs;
  std::vector<uint64_t> remoteBases;
  std::vector<void *> openedIpc;
  std::vector<void *> importedMappings;
  std::vector<CUmemGenericAllocationHandle> importedHandles;
  std::vector<size_t> importedMapSizes;
  std::vector<int> ownedDmaBufFds;
};
struct HostGpuCtx {
  int contextId = 0;
  uint32_t queueSize = 0;
  ncclGinProxyGfd_t *queuesHost = nullptr;
  ncclGinProxyGfd_t *queuesDev = nullptr;
  uint32_t *pisDev = nullptr;
  uint32_t *cisHost = nullptr;
  uint32_t *cisDev = nullptr;
  uint32_t *cisShadow = nullptr;
  uint32_t *sis = nullptr;
};
struct GinCtx {
  CollComm *coll = nullptr;
  int nContexts = 0;
  int nCounters = 0;
  int nSignals = 0;
  uint64_t *countersHost = nullptr;
  uint64_t *countersDev = nullptr;
  uint64_t *signalsDev = nullptr;
  MemHandle *signalsHandle = nullptr;
  HostGpuCtx *contexts = nullptr;
  ncclGinProxyGpuCtx_t *devCtxArray = nullptr;
  ncclNetDeviceHandle_v11_t *devHandle = nullptr;
  std::mutex mutex;
  bool hasError = false;
};
struct RmaCtx {
  CollComm *coll = nullptr;
  int nContexts = 0;
  bool hasError = false;
  cudaStream_t stream = nullptr;
  ncclNetDeviceHandle_v11_t *dummyDevHandle = nullptr;
};

static int envInt(const char *name, int fallback) {
  const char *v = std::getenv(name);
  if (!v || !*v) return fallback;
  char *end = nullptr;
  long out = std::strtol(v, &end, 10);
  return end && *end == 0 && out > 0 ? static_cast<int>(out) : fallback;
}
static bool logProgress() { return envInt("TRITON_DIST_GIN_PROXY_LOG_PROGRESS", 0) != 0; }
static bool isPowerOfTwo(uint32_t x) { return x && ((x & (x - 1)) == 0); }
static ncclResult_t cudaToNccl(cudaError_t err) { return err == cudaSuccess ? ncclSuccess : ncclSystemError; }
#define NCCLCHECK(cmd) do { ncclResult_t _r = (cmd); if (_r != ncclSuccess) return _r; } while (0)

static ncclResult_t sendAll(int fd, const void *buf, size_t len) {
  const char *p = static_cast<const char *>(buf);
  while (len) { ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL); if (n < 0 && errno == EINTR) continue; if (n <= 0) return ncclSystemError; p += n; len -= size_t(n); }
  return ncclSuccess;
}
static ncclResult_t recvAll(int fd, void *buf, size_t len) {
  char *p = static_cast<char *>(buf);
  while (len) { ssize_t n = ::recv(fd, p, len, MSG_WAITALL); if (n < 0 && errno == EINTR) continue; if (n <= 0) return ncclSystemError; p += n; len -= size_t(n); }
  return ncclSuccess;
}
static void closeFd(int &fd) { if (fd >= 0) { ::close(fd); fd = -1; } }

static ncclResult_t sendFd(int sock, int fd) {
  char byte = 'F';
  iovec iov{};
  iov.iov_base = &byte;
  iov.iov_len = sizeof(byte);
  char control[CMSG_SPACE(sizeof(int))];
  std::memset(control, 0, sizeof(control));
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
  msg.msg_controllen = cmsg->cmsg_len;
  ssize_t n = -1;
  do {
    n = ::sendmsg(sock, &msg, MSG_NOSIGNAL);
  } while (n < 0 && errno == EINTR);
  if (n != 1) {
    std::fprintf(stderr, "TDGIN sendFd sock=%d fd=%d -> %zd errno=%d\n", sock, fd, n, errno);
    std::fflush(stderr);
    return ncclSystemError;
  }
  return ncclSuccess;
}

static ncclResult_t recvFd(int sock, int *fdOut) {
  *fdOut = -1;
  char byte = 0;
  iovec iov{};
  iov.iov_base = &byte;
  iov.iov_len = sizeof(byte);
  char control[CMSG_SPACE(sizeof(int))];
  std::memset(control, 0, sizeof(control));
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  int flags = 0;
#ifdef MSG_CMSG_CLOEXEC
  flags |= MSG_CMSG_CLOEXEC;
#endif
  ssize_t n = -1;
  do {
    n = ::recvmsg(sock, &msg, flags);
  } while (n < 0 && errno == EINTR);
  if (n != 1) {
    std::fprintf(stderr, "TDGIN recvFd sock=%d -> %zd errno=%d\n", sock, n, errno);
    std::fflush(stderr);
    return ncclSystemError;
  }
  cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS || cmsg->cmsg_len < CMSG_LEN(sizeof(int))) {
    std::fprintf(stderr, "TDGIN recvFd missing SCM_RIGHTS sock=%d\n", sock);
    std::fflush(stderr);
    return ncclSystemError;
  }
  std::memcpy(fdOut, CMSG_DATA(cmsg), sizeof(*fdOut));
#ifndef MSG_CMSG_CLOEXEC
  if (*fdOut >= 0) ::fcntl(*fdOut, F_SETFD, FD_CLOEXEC);
#endif
  if (logProgress()) {
    std::fprintf(stderr, "TDGIN recvFd sock=%d -> fd=%d\n", sock, *fdOut);
    std::fflush(stderr);
  }
  return *fdOut >= 0 ? ncclSuccess : ncclSystemError;
}

static void *resolvePtr(MemHandle *h, int rank, uint64_t off) {
  if (!h || rank < 0 || rank >= int(h->ptrs.size()) || !h->ptrs[rank]) return nullptr;
  uintptr_t mapped = reinterpret_cast<uintptr_t>(h->ptrs[rank]);
  uintptr_t remoteBase = uintptr_t(h->remoteBases[rank]);
  if (off >= remoteBase && off < remoteBase + h->size) return reinterpret_cast<void *>(mapped + (off - remoteBase));
  if (off < h->size) return reinterpret_cast<void *>(mapped + off);
  return reinterpret_cast<void *>(uintptr_t(off));
}
static uint64_t extractSignalVal(ncclGinProxyGfd_t *gfd) {
  uint64_t v = gfd->qword[ncclGinProxyGfdCompletion].completion.signalValLow;
  v |= uint64_t(gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2) << 16;
  v |= uint64_t(gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh) << 32;
  return v;
}
static ncclGinProxyOp_t extractOp(ncclGinProxyGfd_t *gfd) { return ncclGinProxyOp_t(gfd->qword[ncclGinProxyGfdHeaderExt].headerExt.op); }
static ncclResult_t setCollDevice(CollComm *cc) {
  if (!cc) return ncclInvalidArgument;
  int cur = -1;
  cudaError_t err = cudaGetDevice(&cur);
  if (err != cudaSuccess) return cudaToNccl(err);
  if (cur == cc->cudaDev) return ncclSuccess;
  err = cudaSetDevice(cc->cudaDev);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "TDGIN cudaSetDevice failed current=%d target=%d err=%s\n", cur, cc->cudaDev, cudaGetErrorString(err));
    std::fflush(stderr);
  }
  return cudaToNccl(err);
}
static ncclResult_t copyBytes(void *dst, const void *src, size_t size, cudaStream_t stream = nullptr) {
  if (size == 0) return ncclSuccess;
  if (!dst || !src) {
    std::fprintf(stderr, "TDGIN copyBytes invalid dst=%p src=%p size=%zu\n", dst, src, size);
    std::fflush(stderr);
    return ncclInvalidArgument;
  }
  cudaError_t err = stream ? cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream)
                           : cudaMemcpy(dst, src, size, cudaMemcpyDefault);
  if (err != cudaSuccess) {
    int dev = -1;
    cudaGetDevice(&dev);
    std::fprintf(stderr, "TDGIN copyBytes cudaMemcpy failed dev=%d stream=%p dst=%p src=%p size=%zu err=%s\n", dev, (void *)stream, dst, src, size, cudaGetErrorString(err));
    std::fflush(stderr);
    return cudaToNccl(err);
  }
  err = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    int dev = -1;
    cudaGetDevice(&dev);
    std::fprintf(stderr, "TDGIN copyBytes synchronize failed dev=%d stream=%p dst=%p src=%p size=%zu err=%s\n", dev, (void *)stream, dst, src, size, cudaGetErrorString(err));
    std::fflush(stderr);
  }
  return cudaToNccl(err);
}
static int getSignalLockFd() {
  static int fd = ::open("/tmp/tdgin_signal.lock", O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  return fd;
}
static ncclResult_t lockSignalRmw() {
  int fd = getSignalLockFd();
  if (fd < 0) return ncclSystemError;
  while (::flock(fd, LOCK_EX) != 0) {
    if (errno != EINTR) return ncclSystemError;
  }
  return ncclSuccess;
}
static void unlockSignalRmw() {
  int fd = getSignalLockFd();
  if (fd >= 0) ::flock(fd, LOCK_UN);
}
static ncclResult_t signalValue(void *ptr, uint64_t value, uint32_t op, cudaStream_t stream = nullptr) {
  if (!ptr) {
    std::fprintf(stderr, "TDGIN signalValue invalid ptr=null value=%lu op=%u\n", (unsigned long)value, op);
    std::fflush(stderr);
    return ncclInvalidArgument;
  }
  NCCLCHECK(lockSignalRmw());
  uint64_t old = 0;
  cudaError_t err = stream ? cudaMemcpyAsync(&old, ptr, sizeof(old), cudaMemcpyDefault, stream)
                           : cudaMemcpy(&old, ptr, sizeof(old), cudaMemcpyDefault);
  if (err == cudaSuccess && stream) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    int dev = -1;
    cudaGetDevice(&dev);
    std::fprintf(stderr, "TDGIN signalValue read failed dev=%d stream=%p ptr=%p value=%lu op=%u err=%s\n", dev, (void *)stream, ptr, (unsigned long)value, op, cudaGetErrorString(err));
    std::fflush(stderr);
    unlockSignalRmw();
    return cudaToNccl(err);
  }
  uint64_t next = (op == 1 || op == 0x2) ? old + value : old + 1;
  err = stream ? cudaMemcpyAsync(ptr, &next, sizeof(next), cudaMemcpyDefault, stream)
               : cudaMemcpy(ptr, &next, sizeof(next), cudaMemcpyDefault);
  if (err != cudaSuccess) {
    int dev = -1;
    cudaGetDevice(&dev);
    std::fprintf(stderr, "TDGIN signalValue write failed dev=%d stream=%p ptr=%p old=%lu next=%lu op=%u err=%s\n", dev, (void *)stream, ptr, (unsigned long)old, (unsigned long)next, op, cudaGetErrorString(err));
    std::fflush(stderr);
    unlockSignalRmw();
    return cudaToNccl(err);
  }
  err = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    int dev = -1;
    cudaGetDevice(&dev);
    std::fprintf(stderr, "TDGIN signalValue synchronize failed dev=%d stream=%p ptr=%p next=%lu op=%u err=%s\n", dev, (void *)stream, ptr, (unsigned long)next, op, cudaGetErrorString(err));
    std::fflush(stderr);
  } else if (envInt("TRITON_DIST_GIN_PROXY_LOG_PROGRESS", 0)) {
    static int signalLogCount = 0;
    if (signalLogCount < 64) {
      int dev = -1;
      cudaGetDevice(&dev);
      std::fprintf(stderr, "TDGIN signalValue dev=%d stream=%p ptr=%p old=%lu next=%lu value=%lu op=%u\n", dev, (void *)stream, ptr, (unsigned long)old, (unsigned long)next, (unsigned long)value, op);
      std::fflush(stderr);
      signalLogCount++;
    }
  }
  unlockSignalRmw();
  return cudaToNccl(err);
}


static size_t hostPageSize() {
  long page = ::sysconf(_SC_PAGESIZE);
  return page > 0 ? size_t(page) : 4096;
}
static size_t alignUp(size_t size, size_t align) { return (size + align - 1) & ~(align - 1); }
static uintptr_t alignDown(uintptr_t value, size_t align) { return value & ~(uintptr_t(align) - 1); }
static size_t pageAlign(size_t size) { return alignUp(size, hostPageSize()); }
static ncclResult_t cuToNccl(CUresult res, const char *what) {
  if (res == CUDA_SUCCESS) return ncclSuccess;
  const char *name = nullptr;
  const char *msg = nullptr;
  cuGetErrorName(res, &name);
  cuGetErrorString(res, &msg);
  std::fprintf(stderr, "TDGIN %s failed: %s %s (%d)\n", what, name ? name : "<unknown>", msg ? msg : "", int(res));
  std::fflush(stderr);
  return ncclSystemError;
}
static ncclResult_t exportPosixFdForMapping(void *data, size_t size, int *fdOut, uint64_t *offsetOut,
                                             uint64_t *mapSizeOut) {
  CUresult cres = cuInit(0);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuInit");
  CUdeviceptr dataPtr = CUdeviceptr(reinterpret_cast<uintptr_t>(data));
  CUdeviceptr mappingBase = 0;
  size_t mappingSize = 0;
  cres = cuPointerGetAttribute(&mappingBase, CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR, dataPtr);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuPointerGetAttribute(MAPPING_BASE_ADDR)");
  cres = cuPointerGetAttribute(&mappingSize, CU_POINTER_ATTRIBUTE_MAPPING_SIZE, dataPtr);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuPointerGetAttribute(MAPPING_SIZE)");
  unsigned int allowed = 0;
  CUresult attrRes = cuPointerGetAttribute(&allowed, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, dataPtr);
  if (attrRes == CUDA_SUCCESS && logProgress()) {
    std::fprintf(stderr, "TDGIN pointer allowed handle types=0x%x mappingBase=%p mappingSize=%zu\n", allowed, reinterpret_cast<void *>(uintptr_t(mappingBase)), mappingSize);
    std::fflush(stderr);
  }
  if (mappingBase == 0 || mappingSize == 0) return ncclSystemError;
  uintptr_t base = reinterpret_cast<uintptr_t>(data);
  uintptr_t mapBase = uintptr_t(mappingBase);
  if (base < mapBase || base + size > mapBase + mappingSize) return ncclInvalidArgument;
  CUmemGenericAllocationHandle handle{};
  cres = cuMemRetainAllocationHandle(&handle, data);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuMemRetainAllocationHandle");
  int fd = -1;
  cres = cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  cuMemRelease(handle);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuMemExportToShareableHandle(POSIX_FD)");
  *fdOut = fd;
  *offsetOut = uint64_t(base - mapBase);
  *mapSizeOut = uint64_t(mappingSize);
  return ncclSuccess;
}
static ncclResult_t exportDmaBufForAddressRange(void *data, size_t size, int *fdOut, uint64_t *offsetOut,
                                                  uint64_t *mapSizeOut) {
  CUresult cres = cuInit(0);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuInit");
  int cudaDev = 0;
  cudaError_t cerr = cudaGetDevice(&cudaDev);
  if (cerr != cudaSuccess) return cudaToNccl(cerr);
  CUdevice cuDev{};
  cres = cuDeviceGet(&cuDev, cudaDev);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuDeviceGet");
  int supported = 0;
  cres = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cuDev);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuDeviceGetAttribute(DMA_BUF_SUPPORTED)");
  if (!supported) return ncclSystemError;
  size_t page = hostPageSize();
  uintptr_t base = reinterpret_cast<uintptr_t>(data);
  uintptr_t alignedBase = alignDown(base, page);
  size_t offset = size_t(base - alignedBase);
  size_t mapSize = alignUp(offset + size, page);
  int fd = -1;
  cres = cuMemGetHandleForAddressRange(&fd, CUdeviceptr(alignedBase), mapSize, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuMemGetHandleForAddressRange");
  *fdOut = fd;
  *offsetOut = offset;
  *mapSizeOut = mapSize;
  return ncclSuccess;
}
static int duplicateRemoteFd(int pid, int fd) {
  if (pid == int(::getpid())) {
    int localFd = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
    std::fprintf(stderr, "TDGIN dup local fd pid=%d fd=%d -> %d errno=%d\n", pid, fd, localFd, errno);
    std::fflush(stderr);
    return localFd;
  }
#ifdef SYS_pidfd_open
  int pidfd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
  int pidfdErrno = errno;
  std::fprintf(stderr, "TDGIN pidfd_open pid=%d -> %d errno=%d\n", pid, pidfd, pidfdErrno);
  std::fflush(stderr);
  if (pidfd >= 0) {
#ifdef SYS_pidfd_getfd
    int localFd = static_cast<int>(::syscall(SYS_pidfd_getfd, pidfd, fd, 0));
    int savedErrno = errno;
    std::fprintf(stderr, "TDGIN pidfd_getfd pidfd=%d fd=%d -> %d errno=%d\n", pidfd, fd, localFd, savedErrno);
    std::fflush(stderr);
    ::close(pidfd);
    if (localFd >= 0) return localFd;
    errno = savedErrno;
#else
    ::close(pidfd);
#endif
  }
#endif
  char path[128];
  std::snprintf(path, sizeof(path), "/proc/%d/fd/%d", pid, fd);
  int localFd = ::open(path, O_RDONLY | O_CLOEXEC);
  std::fprintf(stderr, "TDGIN open proc fd path=%s -> %d errno=%d\n", path, localFd, errno);
  std::fflush(stderr);
  return localFd;
}
static ncclResult_t importDmaBufMappingFd(int localFd, size_t size, void **ptr, CUmemGenericAllocationHandle *handle,
                                          size_t *mapSizeOut) {
  if (localFd < 0) return ncclSystemError;
  if (logProgress()) {
    std::fprintf(stderr, "TDGIN import shareable fd=%d size=%zu\n", localFd, size);
    std::fflush(stderr);
  }
  CUresult cres = cuInit(0);
  if (cres != CUDA_SUCCESS) { ::close(localFd); return cuToNccl(cres, "cuInit"); }
  CUmemGenericAllocationHandle h{};
  cres = cuMemImportFromShareableHandle(&h, reinterpret_cast<void *>(uintptr_t(localFd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  ::close(localFd);
  if (cres != CUDA_SUCCESS) return cuToNccl(cres, "cuMemImportFromShareableHandle");
  size_t mapSize = pageAlign(size);
  CUdeviceptr addr = 0;
  cres = cuMemAddressReserve(&addr, mapSize, 0, 0, 0);
  if (cres != CUDA_SUCCESS) { cuMemRelease(h); return cuToNccl(cres, "cuMemAddressReserve"); }
  cres = cuMemMap(addr, mapSize, 0, h, 0);
  if (cres != CUDA_SUCCESS) { cuMemAddressFree(addr, mapSize); cuMemRelease(h); return cuToNccl(cres, "cuMemMap"); }
  int dev = 0;
  cudaError_t cerr = cudaGetDevice(&dev);
  if (cerr != cudaSuccess) {
    cuMemUnmap(addr, mapSize); cuMemAddressFree(addr, mapSize); cuMemRelease(h);
    return cudaToNccl(cerr);
  }
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = dev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cres = cuMemSetAccess(addr, mapSize, &access, 1);
  if (cres != CUDA_SUCCESS) { cuMemUnmap(addr, mapSize); cuMemAddressFree(addr, mapSize); cuMemRelease(h); return cuToNccl(cres, "cuMemSetAccess"); }
  *ptr = reinterpret_cast<void *>(addr);
  *handle = h;
  *mapSizeOut = mapSize;
  return ncclSuccess;
}
static ncclResult_t ginInit(void **ctx, uint64_t commId, ncclDebugLogger_t) { auto *c = new PluginCtx(); c->commId = commId; *ctx = c; return ncclSuccess; }
static ncclResult_t ginDevices(int *ndev) { *ndev = 1; return ncclSuccess; }
static ncclResult_t ginGetProperties(int dev, ncclNetProperties_v12_t *props) {
  std::memset(props, 0, sizeof(*props));
  props->name = const_cast<char *>("TritonDistCudaIpcProxy");
  props->ptrSupport = NCCL_PTR_CUDA_COMPAT | NCCL_PTR_HOST_COMPAT | NCCL_PTR_DMABUF_COMPAT;
  props->speed = 100000; props->port = dev; props->maxComms = 1024 * 1024; props->maxRecvs = envInt("TRITON_DIST_GIN_PROXY_MAX_RECVS", 64);
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY; props->netDeviceVersion = NCCL_GIN_PROXY_VERSION;
  props->vProps.ndevs = 1; props->vProps.devs[0] = dev; props->maxP2pBytes = kMaxNetSize; props->maxCollBytes = kMaxNetSize; props->maxMultiRequestSize = 1;
  return ncclSuccess;
}
static ncclResult_t ginListen(void *, int, void *handle, void **listenComm) {
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return ncclSystemError;
  auto *lc = new ListenComm();
  static unsigned listenSeq = 0;
  unsigned seq = __sync_fetch_and_add(&listenSeq, 1);
  std::snprintf(lc->path, sizeof(lc->path), "/tmp/tdgin.%d.%u.sock", int(::getpid()), seq);
  ::unlink(lc->path);
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::strncpy(addr.sun_path, lc->path, sizeof(addr.sun_path) - 1);
  if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) || ::listen(fd, 256)) {
    int saved = errno;
    closeFd(fd);
    ::unlink(lc->path);
    delete lc;
    errno = saved;
    return ncclSystemError;
  }
  lc->fd = fd;
  WireHandle wh{};
  wh.magic = kHandleMagic;
  std::strncpy(wh.path, lc->path, sizeof(wh.path) - 1);
  std::memset(handle, 0, 128);
  std::memcpy(handle, &wh, sizeof(wh));
  *listenComm = lc;
  return ncclSuccess;
}
static ncclResult_t ginConnect(void *, void *handles[], int nranks, int rank, void *listenComm, void **collComm) {
  auto *lc = static_cast<ListenComm *>(listenComm);
  auto *cc = new CollComm();
  cc->nranks = nranks;
  cc->rank = rank;
  cudaError_t devErr = cudaGetDevice(&cc->cudaDev);
  if (devErr != cudaSuccess) {
    delete cc;
    return cudaToNccl(devErr);
  }
  if (logProgress()) {
    std::fprintf(stderr, "TDGIN connect rank=%d nranks=%d cudaDev=%d\n", rank, nranks, cc->cudaDev);
    std::fflush(stderr);
  }
  cc->sockets.assign(nranks, -1);
  for (int peer = rank + 1; peer < nranks; ++peer) {
    WireHandle wh{};
    std::memcpy(&wh, handles[peer], sizeof(wh));
    if (wh.magic != kHandleMagic || wh.path[0] == 0) return ncclInvalidArgument;
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return ncclSystemError;
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, wh.path, sizeof(addr.sun_path) - 1);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr))) {
      closeFd(fd);
      return ncclSystemError;
    }
    NCCLCHECK(sendAll(fd, &rank, sizeof(rank)));
    cc->sockets[peer] = fd;
  }
  for (int i = 0; i < rank; ++i) {
    sockaddr_un addr{};
    socklen_t len = sizeof(addr);
    int fd = ::accept(lc->fd, reinterpret_cast<sockaddr *>(&addr), &len);
    if (fd < 0) return ncclSystemError;
    int peer = -1;
    NCCLCHECK(recvAll(fd, &peer, sizeof(peer)));
    if (peer < 0 || peer >= nranks || cc->sockets[peer] != -1) {
      closeFd(fd);
      return ncclInvalidArgument;
    }
    cc->sockets[peer] = fd;
  }
  *collComm = cc;
  return ncclSuccess;
}
static ncclResult_t ginRegMrSymCommon(void *collComm, void *data, size_t size, int type, int dmaBufFd,
                                        uint64_t dmaBufOffset, uint64_t, void **mhandle, void **ginHandle) {
  if (logProgress()) {
    std::fprintf(stderr, "TDGIN regMr data=%p size=%zu type=%d dmabuf=%d\n", data, size, type, dmaBufFd);
    std::fflush(stderr);
  }
  auto *cc = static_cast<CollComm *>(collComm);
  auto *mh = new MemHandle();
  mh->type = type;
  mh->size = size;
  mh->local = data;
  mh->ptrs.assign(cc->nranks, nullptr);
  mh->remoteBases.assign(cc->nranks, 0);
  mh->openedIpc.assign(cc->nranks, nullptr);
  mh->importedMappings.assign(cc->nranks, nullptr);
  mh->importedHandles.assign(cc->nranks, CUmemGenericAllocationHandle{});
  mh->importedMapSizes.assign(cc->nranks, 0);
  mh->ptrs[cc->rank] = data;
  mh->remoteBases[cc->rank] = uint64_t(uintptr_t(data));

  WireMemInfo local{};
  local.type = type;
  local.base = uint64_t(uintptr_t(data));
  local.size = size;
  int ownedDmaBufFd = -1;
  if ((type & NCCL_PTR_CUDA_COMPAT) && dmaBufFd >= 0) {
    local.dmaBufValid = 1;
    local.dmaBufFd = dmaBufFd;
    local.dmaBufPid = int(::getpid());
    local.dmaBufOffset = dmaBufOffset;
    local.dmaBufMapSize = pageAlign(size_t(dmaBufOffset) + size);
  } else if (type & NCCL_PTR_CUDA_COMPAT) {
    cudaError_t err = cudaIpcGetMemHandle(&local.ipc, data);
    if (err == cudaSuccess) {
      local.ipcValid = 1;
    } else {
      // ncclMemAlloc/VMM pointers are not CUDA-IPC exportable. This expected
      // fallback still leaves cudaErrorInvalidValue as the runtime last-error,
      // so clear it before returning to PyTorch or later launches can fail.
      (void)cudaGetLastError();
      if (logProgress()) {
        std::fprintf(stderr, "TDGIN cudaIpcGetMemHandle failed data=%p size=%zu type=%d err=%s; trying shareable-handle export\n", data, size, type, cudaGetErrorString(err));
        std::fflush(stderr);
      }
      uint64_t exportedOffset = 0;
      uint64_t exportedSize = 0;
      ncclResult_t exportRet = exportPosixFdForMapping(data, size, &ownedDmaBufFd, &exportedOffset, &exportedSize);
      const char *exportKind = "posix";
      if (exportRet != ncclSuccess) {
        if (logProgress()) {
          std::fprintf(stderr, "TDGIN posix export failed; trying dmabuf export\n");
          std::fflush(stderr);
        }
        exportRet = exportDmaBufForAddressRange(data, size, &ownedDmaBufFd, &exportedOffset, &exportedSize);
        exportKind = "dmabuf";
      }
      if (exportRet != ncclSuccess) {
        delete mh;
        return exportRet;
      }
      local.dmaBufValid = 1;
      local.dmaBufFd = ownedDmaBufFd;
      local.dmaBufPid = int(::getpid());
      local.dmaBufOffset = exportedOffset;
      local.dmaBufMapSize = exportedSize;
      mh->ownedDmaBufFds.push_back(ownedDmaBufFd);
      if (logProgress()) {
        std::fprintf(stderr, "TDGIN %s export data=%p fd=%d offset=%lu mapSize=%lu\n", exportKind, data, ownedDmaBufFd, (unsigned long)exportedOffset, (unsigned long)exportedSize);
        std::fflush(stderr);
      }
    }
  }

  std::lock_guard<std::mutex> guard(cc->mutex);
  for (int peer = 0; peer < cc->nranks; ++peer) {
    if (peer == cc->rank) continue;
    NCCLCHECK(sendAll(cc->sockets[peer], &local, sizeof(local)));
    if (local.dmaBufValid) NCCLCHECK(sendFd(cc->sockets[peer], local.dmaBufFd));
  }
  for (int peer = 0; peer < cc->nranks; ++peer) {
    if (peer == cc->rank) continue;
    WireMemInfo remote{};
    NCCLCHECK(recvAll(cc->sockets[peer], &remote, sizeof(remote)));
    int remoteDmaBufFd = -1;
    if (remote.dmaBufValid) NCCLCHECK(recvFd(cc->sockets[peer], &remoteDmaBufFd));
    mh->remoteBases[peer] = remote.base;
    if (remote.type & NCCL_PTR_CUDA_COMPAT) {
      void *ptr = nullptr;
      if (remote.dmaBufValid) {
        CUmemGenericAllocationHandle h{};
        size_t mapSize = 0;
        NCCLCHECK(importDmaBufMappingFd(remoteDmaBufFd, remote.dmaBufMapSize ? remote.dmaBufMapSize : remote.size, &ptr, &h, &mapSize));
        mh->ptrs[peer] = static_cast<char *>(ptr) + remote.dmaBufOffset;
        mh->importedMappings[peer] = ptr;
        mh->importedHandles[peer] = h;
        mh->importedMapSizes[peer] = mapSize;
      } else {
        if (!remote.ipcValid) return ncclInvalidUsage;
        cudaError_t err = cudaIpcOpenMemHandle(&ptr, remote.ipc, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
          std::fprintf(stderr, "TDGIN cudaIpcOpenMemHandle failed peer=%d base=0x%lx size=%lu type=%d err=%s\n", peer, (unsigned long)remote.base, (unsigned long)remote.size, remote.type, cudaGetErrorString(err));
          std::fflush(stderr);
          return cudaToNccl(err);
        }
        mh->ptrs[peer] = ptr;
        mh->openedIpc[peer] = ptr;
      }
    } else {
      mh->ptrs[peer] = reinterpret_cast<void *>(uintptr_t(remote.base));
    }
  }
  *mhandle = mh;
  *ginHandle = mh;
  return ncclSuccess;
}
static ncclResult_t ginRegMrSym(void *collComm, void *data, size_t size, int type, uint64_t mrFlags,
                                void **mhandle, void **ginHandle) {
  return ginRegMrSymCommon(collComm, data, size, type, -1, 0, mrFlags, mhandle, ginHandle);
}
static ncclResult_t ginRegMrSymDmaBuf(void *collComm, void *data, size_t size, int type, uint64_t offset, int fd,
                                      uint64_t mrFlags, void **mhandle, void **ginHandle) {
  return ginRegMrSymCommon(collComm, data, size, type, fd, offset, mrFlags, mhandle, ginHandle);
}
static ncclResult_t ginDeregMrSym(void *, void *mhandle) {
  auto *mh = static_cast<MemHandle *>(mhandle);
  if (!mh) return ncclSuccess;
  for (void *ptr : mh->openedIpc) {
    if (ptr) cudaIpcCloseMemHandle(ptr);
  }
  for (int fd : mh->ownedDmaBufFds) {
    int tmp = fd;
    closeFd(tmp);
  }
  for (size_t i = 0; i < mh->importedMappings.size(); ++i) {
    if (mh->importedMappings[i]) {
      CUdeviceptr addr = reinterpret_cast<CUdeviceptr>(mh->importedMappings[i]);
      size_t bytes = mh->importedMapSizes[i];
      cuMemUnmap(addr, bytes);
      cuMemAddressFree(addr, bytes);
      cuMemRelease(mh->importedHandles[i]);
    }
  }
  delete mh;
  return ncclSuccess;
}
static ncclResult_t ginCreateContext(void *collComm, ncclGinConfig_v13_t *config, void **ginCtxOut, ncclNetDeviceHandle_v11_t **devHandleOut) {
  std::fprintf(stderr, "TDGIN createContext enter nSignals=%d nCounters=%d nContexts=%d queueDepth=%d\n", config->nSignals, config->nCounters, config->nContexts, config->queueDepth); std::fflush(stderr);
  auto *cc = static_cast<CollComm *>(collComm); auto *gc = new GinCtx(); gc->coll = cc; gc->nContexts = config->nContexts > 0 ? config->nContexts : 1; gc->nCounters = config->nCounters; gc->nSignals = config->nSignals;
  uint32_t queueSize = config->queueDepth > 0 ? uint32_t(config->queueDepth) : kQueueSizeDefault; queueSize = uint32_t(envInt("TRITON_DIST_GIN_PROXY_QUEUE_SIZE", queueSize)); if (!isPowerOfTwo(queueSize)) queueSize = kQueueSizeDefault;
  if (gc->nCounters > 0) { size_t bytes = size_t(gc->nCounters) * gc->nContexts * sizeof(uint64_t); cudaError_t err = cudaHostAlloc(&gc->countersHost, bytes, cudaHostAllocMapped); if (err != cudaSuccess) return cudaToNccl(err); std::memset(gc->countersHost, 0, bytes); err = cudaHostGetDevicePointer(&gc->countersDev, gc->countersHost, 0); if (err != cudaSuccess) return cudaToNccl(err); }
  if (gc->nSignals > 0) { std::fprintf(stderr, "TDGIN createContext signals\n"); std::fflush(stderr); size_t bytes = size_t(gc->nSignals) * gc->nContexts * sizeof(uint64_t); cudaError_t err = cudaMalloc(&gc->signalsDev, bytes); if (err != cudaSuccess) return cudaToNccl(err); cudaMemset(gc->signalsDev, 0, bytes); void *hostHandle = nullptr; void *ginHandle = nullptr; NCCLCHECK(ginRegMrSym(collComm, gc->signalsDev, bytes, NCCL_PTR_CUDA_COMPAT, NCCL_NET_MR_FLAG_FORCE_SO_COMPAT | NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET_COMPAT, &hostHandle, &ginHandle)); gc->signalsHandle = static_cast<MemHandle *>(hostHandle); }
  std::fprintf(stderr, "TDGIN createContext queues\n"); std::fflush(stderr); gc->contexts = new HostGpuCtx[gc->nContexts](); std::vector<ncclGinProxyGpuCtx_t> devCtx(gc->nContexts);
  for (int c = 0; c < gc->nContexts; ++c) { HostGpuCtx &h = gc->contexts[c]; h.contextId = c; h.queueSize = queueSize; size_t qCount = size_t(queueSize) * cc->nranks; cudaError_t err = cudaHostAlloc(&h.queuesHost, qCount * sizeof(ncclGinProxyGfd_t), cudaHostAllocMapped); if (err != cudaSuccess) return cudaToNccl(err); std::memset(h.queuesHost, 0, qCount * sizeof(ncclGinProxyGfd_t)); err = cudaHostGetDevicePointer(&h.queuesDev, h.queuesHost, 0); if (err != cudaSuccess) return cudaToNccl(err); err = cudaMalloc(&h.pisDev, cc->nranks * sizeof(uint32_t)); if (err != cudaSuccess) return cudaToNccl(err); cudaMemset(h.pisDev, 0, cc->nranks * sizeof(uint32_t)); err = cudaHostAlloc(&h.cisHost, cc->nranks * sizeof(uint32_t), cudaHostAllocMapped); if (err != cudaSuccess) return cudaToNccl(err); std::memset(h.cisHost, 0, cc->nranks * sizeof(uint32_t)); err = cudaHostGetDevicePointer(&h.cisDev, h.cisHost, 0); if (err != cudaSuccess) return cudaToNccl(err); h.cisShadow = static_cast<uint32_t *>(std::calloc(cc->nranks, sizeof(uint32_t))); h.sis = static_cast<uint32_t *>(std::calloc(cc->nranks, sizeof(uint32_t))); devCtx[c].nranks = cc->nranks; devCtx[c].queueSize = queueSize; devCtx[c].queues = h.queuesDev; devCtx[c].pis = h.pisDev; devCtx[c].cis = h.cisDev; devCtx[c].counters = gc->countersDev ? gc->countersDev + size_t(c) * gc->nCounters : nullptr; devCtx[c].signals = gc->signalsDev ? gc->signalsDev + size_t(c) * gc->nSignals : nullptr; }
  cudaError_t err = cudaMalloc(&gc->devCtxArray, devCtx.size() * sizeof(ncclGinProxyGpuCtx_t)); if (err != cudaSuccess) return cudaToNccl(err); err = cudaMemcpy(gc->devCtxArray, devCtx.data(), devCtx.size() * sizeof(ncclGinProxyGpuCtx_t), cudaMemcpyHostToDevice); if (err != cudaSuccess) return cudaToNccl(err);
  gc->devHandle = new ncclNetDeviceHandle_v11_t(); std::memset(gc->devHandle, 0, sizeof(*gc->devHandle)); gc->devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY; gc->devHandle->netDeviceVersion = NCCL_GIN_PROXY_VERSION; gc->devHandle->handle = gc->devCtxArray; gc->devHandle->needsProxyProgress = envInt("TRITON_DIST_GIN_PROXY_DISABLE_PROGRESS", 0) ? 0 : 1;
  std::fprintf(stderr, "TDGIN createContext done handle=%p devctx=%p\n", (void*)gc->devHandle, (void*)gc->devCtxArray); std::fflush(stderr); *ginCtxOut = gc; *devHandleOut = gc->devHandle; return ncclSuccess;
}

static ncclResult_t processGfd(GinCtx *gc, HostGpuCtx *h, int targetRank, ncclGinProxyGfd_t *gfd) {
  ncclGinProxyOp_t op = extractOp(gfd);
  if (op & ncclGinProxyOpVASignal) { auto *signalHandle = reinterpret_cast<MemHandle *>(uintptr_t(gfd->qword[ncclGinProxyGfdVASignalHandle].vaSignalHandle.vaSignalHandle)); void *ptr = resolvePtr(signalHandle, targetRank, gfd->qword[ncclGinProxyGfdVASignalOff].vaSignalOff.vaSignalOff); return signalValue(ptr, extractSignalVal(gfd), (op & ncclGinProxyOpWithSignalAdd) ? 0x2 : 0x1); }
  if (op & ncclGinProxyOpFlush) return cudaToNccl(cudaDeviceSynchronize());
  size_t bytes = gfd->qword[ncclGinProxyGfdHeader].header.size;
  if (op & ncclGinProxyOpGet) { auto *srcHandle = reinterpret_cast<MemHandle *>(uintptr_t(gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle)); auto *dstHandle = reinterpret_cast<MemHandle *>(uintptr_t(gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle)); void *src = resolvePtr(srcHandle, targetRank, gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff); void *dst = resolvePtr(dstHandle, gc->coll->rank, gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff); return copyBytes(dst, src, bytes); }
  void *src = nullptr; uint64_t inlineStorage = 0;
  if (op & ncclGinProxyOpWithInline) { inlineStorage = gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow; if (bytes > 4) inlineStorage |= uint64_t(gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2) << 32; if (bytes > 6) inlineStorage |= uint64_t(gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh) << 48; src = &inlineStorage; } else { auto *srcHandle = reinterpret_cast<MemHandle *>(uintptr_t(gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle)); src = resolvePtr(srcHandle, gc->coll->rank, gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff); }
  auto *dstHandle = reinterpret_cast<MemHandle *>(uintptr_t(gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle)); void *dst = resolvePtr(dstHandle, targetRank, gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff); NCCLCHECK(copyBytes(dst, src, bytes));
  if (op & (ncclGinProxyOpWithSignalInc | ncclGinProxyOpWithSignalAdd)) { uint32_t signalId = gfd->qword[ncclGinProxyGfdCompletion].completion.signalId; uint64_t signalOff = (uint64_t(h->contextId) * gc->nSignals + signalId) * sizeof(uint64_t); void *sig = resolvePtr(gc->signalsHandle, targetRank, signalOff); NCCLCHECK(signalValue(sig, extractSignalVal(gfd), (op & ncclGinProxyOpWithSignalAdd) ? 0x2 : 0x1)); }
  return ncclSuccess;
}
static ncclResult_t ginProgress(void *ginCtx) {
  static int progressLogCount = 0;
  if (envInt("TRITON_DIST_GIN_PROXY_LOG_PROGRESS", 0) && progressLogCount < 16) {
    std::fprintf(stderr, "TDGIN progress ginCtx=%p call=%d\n", ginCtx, progressLogCount); std::fflush(stderr);
    progressLogCount++;
  }
  auto *gc = static_cast<GinCtx *>(ginCtx); std::lock_guard<std::mutex> guard(gc->mutex);
  for (int c = 0; c < gc->nContexts; ++c) { HostGpuCtx &h = gc->contexts[c]; for (int target = 0; target < gc->coll->nranks; ++target) { while (true) { uint32_t idx = h.sis[target] & (h.queueSize - 1); ncclGinProxyGfd_t *slot = h.queuesHost + size_t(target) * h.queueSize + idx; if (slot->qword[ncclGinProxyGfdHeader].flag.v == 0) break; ncclGinProxyGfd_t gfd = *slot; std::memset(slot, 0, sizeof(*slot)); h.sis[target]++; ncclResult_t ret = processGfd(gc, &h, target, &gfd); if (ret != ncclSuccess) { gc->hasError = true; return ret; } h.cisHost[target] = ++h.cisShadow[target]; } } }
  return ncclSuccess;
}
static ncclResult_t ginDestroyContext(void *ginCtx) { auto *gc = static_cast<GinCtx *>(ginCtx); if (!gc) return ncclSuccess; if (gc->signalsHandle) ginDeregMrSym(gc->coll, gc->signalsHandle); if (gc->signalsDev) cudaFree(gc->signalsDev); if (gc->countersHost) cudaFreeHost(gc->countersHost); if (gc->contexts) { for (int c = 0; c < gc->nContexts; ++c) { HostGpuCtx &h = gc->contexts[c]; if (h.queuesHost) cudaFreeHost(h.queuesHost); if (h.pisDev) cudaFree(h.pisDev); if (h.cisHost) cudaFreeHost(h.cisHost); std::free(h.cisShadow); std::free(h.sis); } delete[] gc->contexts; } if (gc->devCtxArray) cudaFree(gc->devCtxArray); delete gc->devHandle; delete gc; return ncclSuccess; }
static ncclResult_t ginCloseColl(void *collComm) { auto *cc = static_cast<CollComm *>(collComm); if (!cc) return ncclSuccess; for (int &fd : cc->sockets) closeFd(fd); delete cc; return ncclSuccess; }
static ncclResult_t ginCloseListen(void *listenComm) { auto *lc = static_cast<ListenComm *>(listenComm); if (!lc) return ncclSuccess; closeFd(lc->fd); if (lc->path[0]) ::unlink(lc->path); delete lc; return ncclSuccess; }
static ncclResult_t ginRequestDone(void *, void *, int *done) { *done = 1; return ncclSuccess; }
static ncclResult_t ginQueryLastError(void *ginCtx, bool *hasError) { auto *gc = static_cast<GinCtx *>(ginCtx); *hasError = gc && gc->hasError; return ncclSuccess; }
static ncclResult_t ginFinalize(void *ctx) { delete static_cast<PluginCtx *>(ctx); return ncclSuccess; }


static ncclResult_t rmaCreateContextV13(void *collComm, ncclGinConfig_v13_t *config, void **rmaCtxOut,
                                        ncclNetDeviceHandle_v11_t **devHandleOut) {
  auto *cc = static_cast<CollComm *>(collComm);
  auto *rc = new RmaCtx();
  rc->coll = cc;
  int nContexts = config ? config->nContexts : 1;
  rc->nContexts = nContexts > 0 ? nContexts : 1;
  NCCLCHECK(setCollDevice(cc));
  cudaError_t streamErr = cudaStreamCreateWithFlags(&rc->stream, cudaStreamNonBlocking);
  if (streamErr != cudaSuccess) {
    delete rc;
    return cudaToNccl(streamErr);
  }
  rc->dummyDevHandle = new ncclNetDeviceHandle_v11_t();
  std::memset(rc->dummyDevHandle, 0, sizeof(*rc->dummyDevHandle));
  rc->dummyDevHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  rc->dummyDevHandle->netDeviceVersion = NCCL_GIN_PROXY_VERSION;
  rc->dummyDevHandle->handle = reinterpret_cast<void *>(uintptr_t(1));
  rc->dummyDevHandle->needsProxyProgress = 0;
  if (devHandleOut) *devHandleOut = rc->dummyDevHandle;
  *rmaCtxOut = rc;
  if (envInt("TRITON_DIST_GIN_PROXY_LOG_PROGRESS", 0)) {
    std::fprintf(stderr, "TDGIN RMA createContext nContexts=%d ctx=%p\n", rc->nContexts, (void *)rc); std::fflush(stderr);
  }
  return ncclSuccess;
}
static ncclResult_t rmaDestroyContext(void *rmaCtx) {
  auto *rc = static_cast<RmaCtx *>(rmaCtx);
  if (!rc) return ncclSuccess;
  if (rc->stream) cudaStreamDestroy(rc->stream);
  delete rc->dummyDevHandle;
  delete rc;
  return ncclSuccess;
}
static ncclResult_t rmaRegMrSym(void *collComm, void *data, size_t size, int type, uint64_t mrFlags, void **mhandle,
                                void **ginHandle) {
  void *mh = nullptr;
  void *gh = nullptr;
  NCCLCHECK(ginRegMrSym(collComm, data, size, type, mrFlags, &mh, &gh));
  if (mhandle) *mhandle = mh;
  if (ginHandle) *ginHandle = gh;
  return ncclSuccess;
}
static ncclResult_t rmaRegMrSymDmaBuf(void *collComm, void *data, size_t size, int type, uint64_t offset, int fd,
                                      uint64_t mrFlags, void **mhandle, void **ginHandle) {
  void *mh = nullptr;
  void *gh = nullptr;
  NCCLCHECK(ginRegMrSymDmaBuf(collComm, data, size, type, offset, fd, mrFlags, &mh, &gh));
  if (mhandle) *mhandle = mh;
  if (ginHandle) *ginHandle = gh;
  return ncclSuccess;
}
static ncclResult_t rmaIput(void *rmaCtx, int, uint64_t srcOff, void *srcMhandle, size_t size, uint64_t dstOff,
                            void *dstMhandle, uint32_t rank, void **request) {
  auto *rc = static_cast<RmaCtx *>(rmaCtx);
  NCCLCHECK(setCollDevice(rc ? rc->coll : nullptr));
  auto *srcHandle = static_cast<MemHandle *>(srcMhandle);
  auto *dstHandle = static_cast<MemHandle *>(dstMhandle);
  void *src = resolvePtr(srcHandle, rc->coll->rank, srcOff);
  void *dst = resolvePtr(dstHandle, int(rank), dstOff);
  ncclResult_t ret = copyBytes(dst, src, size, rc->stream);
  if (ret != ncclSuccess) {
    std::fprintf(stderr, "TDGIN rmaIput failed ctx=%p localRank=%d rank=%u srcOff=0x%lx dstOff=0x%lx size=%zu srcHandle=%p dstHandle=%p src=%p dst=%p ret=%d\n", rmaCtx, rc ? rc->coll->rank : -1, rank, (unsigned long)srcOff, (unsigned long)dstOff, size, srcMhandle, dstMhandle, src, dst, int(ret));
    std::fflush(stderr);
    rc->hasError = true;
  } else if (envInt("TRITON_DIST_GIN_PROXY_LOG_PROGRESS", 0)) {
    static int putLogCount = 0;
    if (putLogCount < 64) {
      std::fprintf(stderr, "TDGIN rmaIput localRank=%d rank=%u srcOff=0x%lx dstOff=0x%lx size=%zu src=%p dst=%p\n", rc ? rc->coll->rank : -1, rank, (unsigned long)srcOff, (unsigned long)dstOff, size, src, dst);
      std::fflush(stderr);
      putLogCount++;
    }
  }
  if (request) *request = nullptr;
  return ret;
}
static ncclResult_t rmaIputSignal(void *rmaCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size,
                                  uint64_t dstOff, void *dstMhandle, uint32_t rank, uint64_t signalOff,
                                  void *signalMhandle, uint64_t signalValueArg, uint32_t signalOp, void **request) {
  auto *rc = static_cast<RmaCtx *>(rmaCtx);
  NCCLCHECK(setCollDevice(rc ? rc->coll : nullptr));
  if (size > 0) {
    NCCLCHECK(rmaIput(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request));
  }
  auto *sigHandle = static_cast<MemHandle *>(signalMhandle);
  void *sig = resolvePtr(sigHandle, int(rank), signalOff);
  ncclResult_t ret = signalValue(sig, signalValueArg, signalOp, rc->stream);
  if (ret != ncclSuccess) {
    std::fprintf(stderr, "TDGIN rmaIputSignal failed ctx=%p localRank=%d rank=%u context=%d signalOff=0x%lx signalHandle=%p signalPtr=%p value=%lu op=%u ret=%d\n", rmaCtx, rc ? rc->coll->rank : -1, rank, context, (unsigned long)signalOff, signalMhandle, sig, (unsigned long)signalValueArg, signalOp, int(ret));
    std::fflush(stderr);
    rc->hasError = true;
  }
  if (request) *request = nullptr;
  return ret;
}
static ncclResult_t rmaIget(void *rmaCtx, int, uint64_t remoteOff, void *remoteMhandle, size_t size,
                            uint64_t localOff, void *localMhandle, uint32_t rank, void **request) {
  auto *rc = static_cast<RmaCtx *>(rmaCtx);
  NCCLCHECK(setCollDevice(rc ? rc->coll : nullptr));
  auto *srcHandle = static_cast<MemHandle *>(remoteMhandle);
  auto *dstHandle = static_cast<MemHandle *>(localMhandle);
  void *src = resolvePtr(srcHandle, int(rank), remoteOff);
  void *dst = resolvePtr(dstHandle, rc->coll->rank, localOff);
  ncclResult_t ret = copyBytes(dst, src, size, rc->stream);
  if (ret != ncclSuccess) rc->hasError = true;
  if (request) *request = nullptr;
  return ret;
}
static ncclResult_t rmaIflush(void *rmaCtx, int, void *, uint32_t, void **request) {
  auto *rc = static_cast<RmaCtx *>(rmaCtx);
  NCCLCHECK(setCollDevice(rc ? rc->coll : nullptr));
  ncclResult_t ret = cudaToNccl(rc->stream ? cudaStreamSynchronize(rc->stream) : cudaDeviceSynchronize());
  if (ret != ncclSuccess) rc->hasError = true;
  if (request) *request = nullptr;
  return ret;
}
static ncclResult_t rmaTest(void *, void *, int *done) { *done = 1; return ncclSuccess; }
static ncclResult_t rmaProgress(void *) { return ncclSuccess; }
static ncclResult_t rmaQueryLastError(void *rmaCtx, bool *hasError) {
  auto *rc = static_cast<RmaCtx *>(rmaCtx);
  *hasError = rc && rc->hasError;
  return ncclSuccess;
}

} // namespace

extern "C" __attribute__((visibility("default"))) ncclGin_v13_t ncclGinPlugin_v13;
extern "C" __attribute__((visibility("default"))) ncclGin_v13_t ncclRmaPlugin_v13;
__attribute__((visibility("default"))) ncclGin_v13_t ncclGinPlugin_v13 = {
    "TritonDistCudaIpcProxy", ginInit, ginDevices, ginGetProperties, ginListen, ginConnect, rmaCreateContextV13,
    rmaRegMrSym, rmaRegMrSymDmaBuf, ginDeregMrSym, rmaDestroyContext, ginCloseColl, ginCloseListen,
    rmaIput, rmaIputSignal, rmaIget, rmaIflush, rmaTest, rmaProgress, rmaQueryLastError, ginFinalize,
};
__attribute__((visibility("default"))) ncclGin_v13_t ncclRmaPlugin_v13 = {
    "TritonDistCudaIpcRma", ginInit, ginDevices, ginGetProperties, ginListen, ginConnect, rmaCreateContextV13,
    rmaRegMrSym, rmaRegMrSymDmaBuf, ginDeregMrSym, rmaDestroyContext, ginCloseColl, ginCloseListen,
    rmaIput, rmaIputSignal, rmaIget, rmaIflush, rmaTest, rmaProgress, rmaQueryLastError, ginFinalize,
};

#else
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

#include "registry.h"

#include <dlfcn.h>
#include <link.h>

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_device.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace distributed {
namespace ops {
namespace {


struct NCCLHostApi {
  using GetUniqueIdFn = ncclResult_t (*)(ncclUniqueId *);
  using CommInitRankFn = ncclResult_t (*)(ncclComm_t *, int, ncclUniqueId, int);
  using CommDestroyFn = ncclResult_t (*)(ncclComm_t);
  using GetErrorStringFn = const char *(*)(ncclResult_t);
  using CommQueryPropertiesFn = ncclResult_t (*)(ncclComm_t, ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(ncclComm_t, ncclDevCommRequirements_t const *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t, ncclDevComm_t *);
  using CommWindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t, ncclWindow_t *, int);
  using CommWindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);

  void *handle = nullptr;
  GetUniqueIdFn get_unique_id = nullptr;
  CommInitRankFn comm_init_rank = nullptr;
  CommDestroyFn comm_destroy = nullptr;
  GetErrorStringFn get_error_string = nullptr;
  CommQueryPropertiesFn comm_query_properties = nullptr;
  DevCommCreateFn dev_comm_create = nullptr;
  DevCommDestroyFn dev_comm_destroy = nullptr;
  CommWindowRegisterFn comm_window_register = nullptr;
  CommWindowDeregisterFn comm_window_deregister = nullptr;
  MemAllocFn mem_alloc = nullptr;
  MemFreeFn mem_free = nullptr;
};

void *open_nccl_library() {
  int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_DEEPBIND
  flags |= RTLD_DEEPBIND;
#endif
  std::vector<std::string> candidates;
  if (const char *env = std::getenv("TRITON_DIST_NCCL_LIB")) {
    candidates.emplace_back(env);
  }
  candidates.emplace_back("/usr/lib/x86_64-linux-gnu/libnccl.so.2");
  candidates.emplace_back("/usr/lib/x86_64-linux-gnu/libnccl.so");
  candidates.emplace_back("/lib/x86_64-linux-gnu/libnccl.so.2");
  candidates.emplace_back("libnccl.so.2");
  candidates.emplace_back("libnccl.so");

  std::ostringstream errors;
  for (auto const &candidate : candidates) {
    dlerror();
    void *handle = nullptr;
    if (std::getenv("TRITON_DIST_NCCL_USE_DLMOPEN") && !candidate.empty() && candidate[0] == '/') {
#ifdef __GLIBC__
      handle = dlmopen(LM_ID_NEWLM, candidate.c_str(), flags);
#endif
    }
    if (handle == nullptr) {
      handle = dlopen(candidate.c_str(), flags);
    }
    if (handle != nullptr) {
      return handle;
    }
    if (const char *err = dlerror()) {
      errors << "  " << candidate << ": " << err << "\n";
    }
  }
  throw std::runtime_error("Unable to load NCCL host library for GIN:\n" + errors.str());
}

template <typename Fn>
Fn load_nccl_symbol(void *handle, const char *name, const char *prefixed_name) {
  dlerror();
  auto *symbol = dlsym(handle, name);
  if (symbol == nullptr && prefixed_name != nullptr) {
    symbol = dlsym(handle, prefixed_name);
  }
  if (symbol == nullptr) {
    std::ostringstream oss;
    oss << "NCCL GIN requires symbol " << name << " in the loaded NCCL library";
    if (const char *err = dlerror()) {
      oss << ": " << err;
    }
    throw std::runtime_error(oss.str());
  }
  return reinterpret_cast<Fn>(symbol);
}

NCCLHostApi &api() {
  static NCCLHostApi api = []() {
    NCCLHostApi out;
    out.handle = open_nccl_library();
    out.get_unique_id = load_nccl_symbol<NCCLHostApi::GetUniqueIdFn>(out.handle, "ncclGetUniqueId", "pncclGetUniqueId");
    out.comm_init_rank = load_nccl_symbol<NCCLHostApi::CommInitRankFn>(out.handle, "ncclCommInitRank", "pncclCommInitRank");
    out.comm_destroy = load_nccl_symbol<NCCLHostApi::CommDestroyFn>(out.handle, "ncclCommDestroy", "pncclCommDestroy");
    out.get_error_string = load_nccl_symbol<NCCLHostApi::GetErrorStringFn>(out.handle, "ncclGetErrorString", "pncclGetErrorString");
    out.comm_query_properties = load_nccl_symbol<NCCLHostApi::CommQueryPropertiesFn>(out.handle, "ncclCommQueryProperties", "pncclCommQueryProperties");
    out.dev_comm_create = load_nccl_symbol<NCCLHostApi::DevCommCreateFn>(out.handle, "ncclDevCommCreate", "pncclDevCommCreate");
    out.dev_comm_destroy = load_nccl_symbol<NCCLHostApi::DevCommDestroyFn>(out.handle, "ncclDevCommDestroy", "pncclDevCommDestroy");
    out.comm_window_register = load_nccl_symbol<NCCLHostApi::CommWindowRegisterFn>(out.handle, "ncclCommWindowRegister", "pncclCommWindowRegister");
    out.comm_window_deregister = load_nccl_symbol<NCCLHostApi::CommWindowDeregisterFn>(out.handle, "ncclCommWindowDeregister", "pncclCommWindowDeregister");
    out.mem_alloc = load_nccl_symbol<NCCLHostApi::MemAllocFn>(out.handle, "ncclMemAlloc", "pncclMemAlloc");
    out.mem_free = load_nccl_symbol<NCCLHostApi::MemFreeFn>(out.handle, "ncclMemFree", "pncclMemFree");
    return out;
  }();
  return api;
}

std::string nccl_error(ncclResult_t result, const char *expr, const char *file,
                       int line) {
  std::ostringstream oss;
  oss << "NCCL error at " << file << ":" << line << " for " << expr << ": "
      << api().get_error_string(result);
  return oss.str();
}

std::string cuda_error(cudaError_t result, const char *expr, const char *file,
                       int line) {
  std::ostringstream oss;
  oss << "CUDA error at " << file << ":" << line << " for " << expr << ": "
      << cudaGetErrorString(result);
  return oss.str();
}

#define NCCL_GIN_CHECK(expr)                                                   \
  do {                                                                         \
    ncclResult_t _result = (expr);                                             \
    if (_result != ncclSuccess) {                                              \
      throw std::runtime_error(nccl_error(_result, #expr, __FILE__, __LINE__)); \
    }                                                                          \
  } while (0)

#define CUDA_GIN_CHECK(expr)                                                   \
  do {                                                                         \
    cudaError_t _result = (expr);                                              \
    if (_result != cudaSuccess) {                                              \
      throw std::runtime_error(cuda_error(_result, #expr, __FILE__, __LINE__)); \
    }                                                                          \
  } while (0)

struct NCCLGinContext {
  std::mutex mutex;
  bool initialized = false;
  bool dev_comm_initialized = false;
  int rank = -1;
  int world_size = -1;
  int device = -1;
  ncclComm_t comm = nullptr;
  ncclDevComm_t dev_comm{};
  std::unordered_map<std::uintptr_t, ncclWindow_t> windows;
};

NCCLGinContext &context() {
  static NCCLGinContext ctx;
  return ctx;
}

constexpr int kDLCUDA = 2;

struct DLDeviceCompat {
  int32_t device_type;
  int32_t device_id;
};

struct DLDataTypeCompat {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct DLTensorCompat {
  void *data;
  DLDeviceCompat device;
  int32_t ndim;
  DLDataTypeCompat dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
};

struct DLManagedTensorCompat {
  DLTensorCompat dl_tensor;
  void *manager_ctx;
  void (*deleter)(DLManagedTensorCompat *);
};

void dlpack_managed_tensor_deleter(DLManagedTensorCompat *managed) {
  if (managed == nullptr) {
    return;
  }
  void *data = managed->dl_tensor.data;
  if (data != nullptr) {
    try {
      ncclResult_t result = api().mem_free(data);
      if (result != ncclSuccess) {
        std::fprintf(stderr, "NCCL GIN ncclMemFree failed in DLPack deleter: %s\n", api().get_error_string(result));
        std::fflush(stderr);
      }
    } catch (...) {
      std::fprintf(stderr, "NCCL GIN ncclMemFree threw in DLPack deleter\n");
      std::fflush(stderr);
    }
  }
  delete[] managed->dl_tensor.shape;
  delete managed;
}

void dlpack_capsule_destructor(PyObject *capsule) {
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    auto *managed = reinterpret_cast<DLManagedTensorCompat *>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed != nullptr && managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  }
}

void ensure_initialized(NCCLGinContext const &ctx) {
  if (!ctx.initialized) {
    throw std::runtime_error("NCCL GIN has not been initialized");
  }
}

void reset_context(NCCLGinContext &ctx) {
  ctx.initialized = false;
  ctx.dev_comm_initialized = false;
  ctx.rank = -1;
  ctx.world_size = -1;
  ctx.device = -1;
  ctx.comm = nullptr;
  std::memset(&ctx.dev_comm, 0, sizeof(ctx.dev_comm));
  ctx.windows.clear();
}

py::dict properties_to_dict(ncclCommProperties_t const &props) {
  py::dict ret;
  ret["rank"] = props.rank;
  ret["nRanks"] = props.nRanks;
  ret["cudaDev"] = props.cudaDev;
  ret["deviceApiSupport"] = props.deviceApiSupport;
  ret["ginType"] = static_cast<int>(props.ginType);
  ret["railedGinType"] = static_cast<int>(props.railedGinType);
  ret["multimemSupport"] = props.multimemSupport;
  ret["hostRmaSupport"] = props.hostRmaSupport;
  return ret;
}

std::string properties_to_string(ncclCommProperties_t const &props) {
  std::ostringstream oss;
  oss << "rank=" << props.rank << ", nRanks=" << props.nRanks
      << ", cudaDev=" << props.cudaDev
      << ", deviceApiSupport=" << props.deviceApiSupport
      << ", ginType=" << static_cast<int>(props.ginType)
      << ", railedGinType=" << static_cast<int>(props.railedGinType)
      << ", multimemSupport=" << props.multimemSupport
      << ", hostRmaSupport=" << props.hostRmaSupport;
  return oss.str();
}

ncclUniqueId unique_id_from_bytes(py::bytes unique_id_bytes) {
  std::string unique_id = py::cast<std::string>(unique_id_bytes);
  if (unique_id.size() != NCCL_UNIQUE_ID_BYTES) {
    std::ostringstream oss;
    oss << "NCCL unique id must be " << NCCL_UNIQUE_ID_BYTES << " bytes, got "
        << unique_id.size();
    throw std::runtime_error(oss.str());
  }

  ncclUniqueId id;
  std::memcpy(&id, unique_id.data(), NCCL_UNIQUE_ID_BYTES);
  return id;
}

py::bytes get_unique_id() {
  ncclUniqueId unique_id;
  auto &nccl = api();
  NCCL_GIN_CHECK(nccl.get_unique_id(&unique_id));
  return py::bytes(reinterpret_cast<const char *>(&unique_id),
                   NCCL_UNIQUE_ID_BYTES);
}

py::dict probe(py::bytes unique_id_bytes, int rank, int world_size, int device) {
  if (rank < 0 || rank >= world_size) {
    throw std::runtime_error("Invalid NCCL GIN rank/world_size");
  }

  ncclUniqueId id = unique_id_from_bytes(unique_id_bytes);
  CUDA_GIN_CHECK(cudaSetDevice(device));
  auto &nccl = api();

  ncclComm_t comm = nullptr;
  NCCL_GIN_CHECK(nccl.comm_init_rank(&comm, world_size, id, rank));
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t query_result = nccl.comm_query_properties(comm, &props);
  ncclResult_t destroy_result = nccl.comm_destroy(comm);
  if (query_result != ncclSuccess) {
    throw std::runtime_error(nccl_error(query_result, "ncclCommQueryProperties", __FILE__, __LINE__));
  }
  if (destroy_result != ncclSuccess) {
    throw std::runtime_error(nccl_error(destroy_result, "ncclCommDestroy", __FILE__, __LINE__));
  }
  return properties_to_dict(props);
}

void init(py::bytes unique_id_bytes, int rank, int world_size, int device,
          int barrier_count, int gin_signal_count, int gin_context_count) {
  auto &ctx = context();
  std::lock_guard<std::mutex> guard(ctx.mutex);
  if (ctx.initialized) {
    throw std::runtime_error("NCCL GIN has already been initialized");
  }
  if (rank < 0 || rank >= world_size) {
    throw std::runtime_error("Invalid NCCL GIN rank/world_size");
  }
  if (barrier_count < 0 || gin_signal_count < 0 || gin_context_count <= 0) {
    throw std::runtime_error("Invalid NCCL GIN device requirements");
  }

  ncclUniqueId id = unique_id_from_bytes(unique_id_bytes);

  CUDA_GIN_CHECK(cudaSetDevice(device));
  auto &nccl = api();
  NCCL_GIN_CHECK(nccl.comm_init_rank(&ctx.comm, world_size, id, rank));

  ncclCommProperties_t properties = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_GIN_CHECK(nccl.comm_query_properties(ctx.comm, &properties));
  if (!properties.deviceApiSupport) {
    NCCL_GIN_CHECK(nccl.comm_destroy(ctx.comm));
    reset_context(ctx);
    throw std::runtime_error(
        "NCCL communicator does not support the device API. Queried "
        "properties: " + properties_to_string(properties));
  }
  if (properties.ginType == NCCL_GIN_TYPE_NONE) {
    NCCL_GIN_CHECK(nccl.comm_destroy(ctx.comm));
    reset_context(ctx);
    throw std::runtime_error(
        "NCCL communicator does not report GIN support. NCCL GIN requires "
        "a compatible NCCL runtime plus a loadable GIN plugin/runtime "
        "(for example libnccl-gin.so). If PyTorch loads its bundled NCCL "
        "first, preload the compatible system NCCL so both PyTorch and "
        "Triton-distributed use the same NCCL library. Queried properties: " +
        properties_to_string(properties));
  }

  ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = barrier_count;
  reqs.ginSignalCount = gin_signal_count;
  reqs.ginContextCount = gin_context_count;
  reqs.ginExclusiveContexts = true;
  const char *queue_depth_env = std::getenv("NCCL_GIN_PROXY_QUEUE_SIZE");
  reqs.ginQueueDepth = (queue_depth_env && *queue_depth_env) ? std::atoi(queue_depth_env) : 1024;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 7)
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
  reqs.ginForceEnable = true;
#endif

  NCCL_GIN_CHECK(nccl.dev_comm_create(ctx.comm, &reqs, &ctx.dev_comm));
  ctx.rank = rank;
  ctx.world_size = world_size;
  ctx.device = device;
  ctx.initialized = true;
  ctx.dev_comm_initialized = true;
}

bool is_initialized() { return context().initialized; }

py::bytes dev_comm_bytes() {
  auto &ctx = context();
  std::lock_guard<std::mutex> guard(ctx.mutex);
  ensure_initialized(ctx);
  return py::bytes(reinterpret_cast<const char *>(&ctx.dev_comm),
                   sizeof(ctx.dev_comm));
}

std::size_t dev_comm_size() { return sizeof(ncclDevComm_t); }

std::uintptr_t mem_alloc(std::size_t nbytes) {
  if (nbytes == 0) {
    return 0;
  }
  void *ptr = nullptr;
  auto &nccl = api();
  NCCL_GIN_CHECK(nccl.mem_alloc(&ptr, nbytes));
  return reinterpret_cast<std::uintptr_t>(ptr);
}

void mem_free(std::uintptr_t data_ptr) {
  if (data_ptr == 0) {
    return;
  }
  auto &nccl = api();
  NCCL_GIN_CHECK(nccl.mem_free(reinterpret_cast<void *>(data_ptr)));
}

py::capsule empty_dlpack(std::vector<std::int64_t> shape, int dtype_code,
                         int dtype_bits, int dtype_lanes) {
  if (dtype_bits <= 0 || dtype_lanes <= 0 || (dtype_bits * dtype_lanes) % 8 != 0) {
    throw std::runtime_error("Invalid DLPack dtype for NCCL GIN allocation");
  }
  std::size_t numel = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      throw std::runtime_error("NCCL GIN tensor shapes cannot have negative dimensions");
    }
    numel *= static_cast<std::size_t>(dim);
  }
  std::size_t nbytes = numel * static_cast<std::size_t>(dtype_bits * dtype_lanes / 8);
  void *ptr = nullptr;
  if (nbytes != 0) {
    auto &nccl = api();
    NCCL_GIN_CHECK(nccl.mem_alloc(&ptr, nbytes));
  }

  int device = 0;
  CUDA_GIN_CHECK(cudaGetDevice(&device));
  auto *managed = new DLManagedTensorCompat();
  std::memset(managed, 0, sizeof(*managed));
  auto ndim = static_cast<int32_t>(shape.size());
  auto *shape_copy = new int64_t[shape.empty() ? 1 : shape.size()];
  for (std::size_t i = 0; i < shape.size(); ++i) {
    shape_copy[i] = shape[i];
  }
  managed->dl_tensor.data = ptr;
  managed->dl_tensor.device = {kDLCUDA, device};
  managed->dl_tensor.ndim = ndim;
  managed->dl_tensor.dtype = {static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits), static_cast<uint16_t>(dtype_lanes)};
  managed->dl_tensor.shape = shape_copy;
  managed->dl_tensor.strides = nullptr;
  managed->dl_tensor.byte_offset = 0;
  managed->manager_ctx = nullptr;
  managed->deleter = dlpack_managed_tensor_deleter;
  return py::capsule(managed, "dltensor", dlpack_capsule_destructor);
}

std::uintptr_t register_window(std::uintptr_t data_ptr, std::size_t nbytes,
                               bool collective_symmetric,
                               bool strict_ordering) {
  auto &ctx = context();
  std::lock_guard<std::mutex> guard(ctx.mutex);
  ensure_initialized(ctx);
  if (data_ptr == 0 || nbytes == 0) {
    throw std::runtime_error("Cannot register an empty NCCL GIN window");
  }
  ncclWindow_t window = nullptr;
  int flags = collective_symmetric ? NCCL_WIN_COLL_SYMMETRIC : NCCL_WIN_DEFAULT;
  if (strict_ordering) {
    flags |= NCCL_WIN_STRICT_ORDERING;
  }
  auto &nccl = api();
  NCCL_GIN_CHECK(nccl.comm_window_register(
      ctx.comm, reinterpret_cast<void *>(data_ptr), nbytes, &window, flags));
  auto handle = reinterpret_cast<std::uintptr_t>(window);
  ctx.windows.emplace(handle, window);
  return handle;
}

void deregister_window(std::uintptr_t handle) {
  auto &ctx = context();
  std::lock_guard<std::mutex> guard(ctx.mutex);
  ensure_initialized(ctx);
  auto it = ctx.windows.find(handle);
  if (it == ctx.windows.end()) {
    return;
  }
  auto &nccl = api();
  NCCL_GIN_CHECK(nccl.comm_window_deregister(ctx.comm, it->second));
  ctx.windows.erase(it);
}

py::dict properties() {
  auto &ctx = context();
  std::lock_guard<std::mutex> guard(ctx.mutex);
  ensure_initialized(ctx);
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
  auto &nccl = api();
  NCCL_GIN_CHECK(nccl.comm_query_properties(ctx.comm, &props));
  return properties_to_dict(props);
}

void finalize() {
  auto &ctx = context();
  std::lock_guard<std::mutex> guard(ctx.mutex);
  if (!ctx.initialized) {
    return;
  }
  auto &nccl = api();
  for (auto const &item : ctx.windows) {
    NCCL_GIN_CHECK(nccl.comm_window_deregister(ctx.comm, item.second));
  }
  ctx.windows.clear();
  if (ctx.dev_comm_initialized) {
    NCCL_GIN_CHECK(nccl.dev_comm_destroy(ctx.comm, &ctx.dev_comm));
  }
  if (ctx.comm != nullptr) {
    NCCL_GIN_CHECK(nccl.comm_destroy(ctx.comm));
  }
  reset_context(ctx);
}

void init_bindings(py::module &m) {
  auto gin = m.def_submodule("nccl_gin");
  gin.def("get_unique_id", &get_unique_id);
  gin.def("probe", &probe, py::arg("unique_id"), py::arg("rank"),
          py::arg("world_size"), py::arg("device"));
  gin.def("init", &init, py::arg("unique_id"), py::arg("rank"),
          py::arg("world_size"), py::arg("device"),
          py::arg("barrier_count") = 1, py::arg("gin_signal_count") = 1,
          py::arg("gin_context_count") = 4);
  gin.def("is_initialized", &is_initialized);
  gin.def("dev_comm_bytes", &dev_comm_bytes);
  gin.def("dev_comm_size", &dev_comm_size);
  gin.def("mem_alloc", &mem_alloc);
  gin.def("mem_free", &mem_free);
  gin.def("empty_dlpack", &empty_dlpack, py::arg("shape"),
          py::arg("dtype_code"), py::arg("dtype_bits"),
          py::arg("dtype_lanes") = 1);
  gin.def("register_window", &register_window, py::arg("data_ptr"),
          py::arg("nbytes"), py::arg("collective_symmetric") = true,
          py::arg("strict_ordering") = false);
  gin.def("deregister_window", &deregister_window);
  gin.def("properties", &properties);
  gin.def("finalize", &finalize);
}

const bool registered = []() {
  OpInitRegistry::instance().register_one("nccl_gin", init_bindings);
  return true;
}();

} // namespace
} // namespace ops
} // namespace distributed

#endif  // TRITON_DIST_BUILD_NCCL_GIN_PLUGIN
