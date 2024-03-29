/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "nccl_net.h"
#include <stdio.h>
#include <chrono>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include <pthread.h>

#define OFCCL_LOG(PRE, FMT, args...) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args)
#define OFCCL_LOG1(PRE, FMT) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__)
#define OFCCL_LOG0(PRE) printf("[%s:%d] <%s> " #PRE "\n", __FILE__, __LINE__, __func__)

#define OFCCL_LOG_RANK_0(PRE, FMT, args...) do { if (thrdCudaDev==0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_RANK_0_SHMEM(PRE, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_RANK_0_THRD_0_SHMEM(PRE, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==0 && threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define OFCCL_LOG_RANK_X(PRE, X, FMT, args...) do { if (thrdCudaDev==X) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_RANK_X_SHMEM(PRE, X, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==X) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_RANK_X_THRD_0_SHMEM(PRE, X, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==X && threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_RANK_X_THRD_0(PRE, X, FMT, args...) do { if (thrdCudaDev==X && threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define OFCCL_LOG_THRD_0(PRE, FMT, args...) do { if (threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_WARP_HEAD(PRE, FMT, args...) do { if (threadIdx.x % WARP_SIZE == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(PRE, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==0 && threadIdx.x % WARP_SIZE == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define OFCCL_LOG_BLK_0(PRE, FMT, args...) do { if (blockIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_BLK_1(PRE, FMT, args...) do { if (blockIdx.x == 1) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_BLK_0_THRD_0(PRE, FMT, args...) do { if (blockIdx.x == 0 && threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define OFCCL_LOG_RANK_0_BLK_0(PRE, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==0 && blockIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define OFCCL_LOG_RANK_1_BLK_0(PRE, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==1 && blockIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define OFCCL_LOG_RANK_0_BLK_0_THRD_0(PRE, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==0 && blockIdx.x == 0 && threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define OFCCL_LOG_RANK_0_THRD_0(PRE, FMT, args...) do { if (sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.rank==0 && threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define NCCL_LOG_RANK_0(PRE, FMT, args...) do { if (ncclShmem.comm.rank==0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define NCCL_LOG_RANK_0_THRD_0(PRE, FMT, args...) do { if (ncclShmem.comm.rank==0 && threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define NCCL_LOG_RANK_0_THRD_100(PRE, FMT, args...) do { if (ncclShmem.comm.rank==0 && threadIdx.x == 100) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define NCCL_LOG_THRD_0(PRE, FMT, args...) do { if (threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)
#define NCCL_LOG_RANK_0_THRD_0_PURE(FMT, args...) do { if (ncclShmem.comm.rank==0 && threadIdx.x == 0) printf(FMT "\t", args); } while(0)

// #define OFCCL_LOG(PRE, FMT, args...) do {} while (0)
// #define OFCCL_LOG1(PRE, FMT) do {} while (0)
// #define OFCCL_LOG0(PRE) do {} while (0)

#define OFCCL_LOG_FINAL(PRE, FMT, args...) do { if (threadIdx.x == 0) printf("[%s:%d] <%s> " #PRE " " FMT "\n", __FILE__, __LINE__, __func__, args); } while(0)

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

inline bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file,
                          int line) {
  if (code != cudaSuccess) {
    const char *err_name = cudaGetErrorName(code);
    const char *err_message = cudaGetErrorString(code);
    printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n",
           file, line, op, err_name, err_message);
    cudaGetLastError();
    return false;
  }
  return true;
}

// #define ofcclNcclCheck(op) __check_nccl_runtime((op), #op, __FILE__, __LINE__)

// inline bool __check_nccl_runtime(cudaError_t code, const char *op, const char *file,
//                           int line) {
//   if (code != cudaSuccess) {
//     const char *err_message = ncclGetErrorString(code);
//     printf("runtime error %s:%d  %s failed. \n  message = %s\n",
//            file, line, op, err_message);
//     cudaGetLastError();
//     return false;
//   }
//   return true;
// }                                                            

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen, const char delim);

void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *filefunc, int line, const char *fmt, ...) __attribute__ ((format (printf, 5, 6)));

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;

#define WARN(...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point ncclEpoch;
#else
#define TRACE(...)
#endif

void ncclSetThreadName(pthread_t thread, const char *fmt, ...);

#endif
