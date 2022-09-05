/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef OFCCL_ENQUEUE_H_
#define OFCCL_ENQUEUE_H_

#include "comm.h"
#include "nccl.h"
#include "group.h"
#include "devcomm.h"
#include "debug.h"
#include "collectives_ofccl.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <unordered_map>

#define MAX_ASYNC_PANELS 32
#define MAX_ASYNC_OPS 128
// #define MAX_LENGTH 587 // 587不超0xc000 shmem的限制，588就超了(0xc00c)，这时在启用enqueue_ofccl_dev.cu里边全部3个shared数组的情况下，假如587不够用，可以考虑根据使用频率删除其中的几个；编译器会优化掉没使用的static shared声明，测量时候要注意。
#define MAX_LENGTH 128 // TODO: 先搞小一点，开发之后再优化
// 队列长度搞大些，反正目前也不缺这点显存。就搞得和max collCount一样大，那就不会full了。
#define QLen MAX_LENGTH
#define tempPrintRound 100000

struct ofcclCommArgs {
  ncclResult_t ret;
  ncclComm_t comm;
};

struct DevComm7WorkElem {
  struct ncclDevComm* comm;
  ncclWorkElem first;
};

ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId);

#define RingBuffer_full(B) ((((B)->tail + 1) % (B)->length) == ((B)->head % (B)->length))

#define RingBuffer_empty(B) (((B)->tail % (B)->length) == ((B)->head % (B)->length))

#define RingBuffer_get_head(B) ((B)->buffer + ((B)->head % (B)->length))
#define RingBuffer_get(B, frontier) ((B)->buffer + (frontier% (B)->length))

#define RingBuffer_get_tail(B) ((B)->buffer + ((B)->tail % (B)->length))

#define RingBuffer_logic_head(B) ((B)->head % (B)->length)
#define GetLogicFrontier(B, frontier) (frontier % (B)->length)

#define RingBuffer_logic_tail(B) ((B)->tail % (B)->length)

// 对于device和CPU上的commit分别进行不同的多线程保护。
#define RingBuffer_commit_read(B, A) ((B)->head = ((B)->head + (A)) % (B)->length)

#define RingBuffer_commit_write(B, A) ((B)->tail = ((B)->tail + (A)) % (B)->length)

#define testBlkCnt4Coll(i) i % 2 == 0 ? daemonKernelGridDim.x : daemonKernelGridDim.x - 1


// static thread_local int CPUSleep = 0;
// __device__ static thread_local int GPUSleep = 0;
// static thread_local int CpuSleepUs = 1e6;
// __device__ static thread_local clock_t GpuSpin = 1e9 * 2;
// #define GpuSpin4Bid(i) (1 + i) * 1e9
// #define GpuSpin4BidSmall(i) (6 + i) * 1e6

typedef struct {
  int collId;
  int counter;
  int logicHead;

  const void *send_buffer;
  void *recv_buffer;

  bool quit;
} SQE;

typedef struct {
  SQE *buffer;
  int length;
  unsigned long long int head;
  unsigned long long int tail;
  pthread_mutex_t mutex;
} SQ;

SQ *sqCreate(int length);
void sqDestroy(SQ *sq);
// SQ read by device, written by host;
__device__ int sqRead(SQ *sq, unsigned long long int sqReadFrontier, SQE *target, int *globalBlkCount4Coll, int thrdCudaDev); // read 1 element each time
int sqWrite(SQ *sq, SQE *sqe, int thrdCudaDev, CallbackFunc callback, void *callbackArgs);

typedef struct {
  int collId;
  int counter;
} CQE;

typedef struct {
  CQE *buffer;
  int length;
  unsigned long long int head;
  unsigned long long int tail;
  pthread_mutex_t mutex;
} CQ;

CQ *cqCreate(int length);
void cqDestroy(CQ *cq);
// CQ read by host, written by device;
int cqRead(CQ *cq, CQE *target, int thrdCudaDev); // read 1 element each time
__device__ int cqWrite(CQ *cq, CQE *cqe, int thrdCudaDev);

typedef struct {
  int *poll_start;
  int *poll_stop;
  // std::unordered_map<int, CallbackFunc> *collId2callback;
  CallbackFunc *callbacks;
  int cudaDev;
  CQ *cq;
  void **callbackArgList;
} PollerArgs;

// typedef struct {
//   int collId;
//   int gotCqe;
// } CallBackArgs;

struct ofcclShmemGroup {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
};

// sizeof(ofcclShmemData)=42104, sizeof(ofcclShmemGroup)=248, sizeof(ncclDevComm)=40, sizeof(ncclChannel)=512, sizeof(ncclWork)=512
typedef struct {
  union {
    uint64_t ll128warp[NCCL_LL128_MAX_NTHREADS/WARP_SIZE][NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE]; // 这个占得大，占了40960
    struct ofcclShmemGroup groups[NCCL_MAX_GROUPS]; // 这个只占了3968
  };
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  struct ncclDevComm comm;
  struct ncclChannel channel;
  uint64_t pad;
  struct ncclWork work;
  // 代表当前的表项对应的集合通信被调用，还没有执行完成。初始化置0；发现了相应的sqe之后置1；执行完成后置0。
  int executing;
} ofcclShmemData;
static_assert(offsetof(ofcclShmemData, work)%16 == 0, "shmem.work needs to be 16B aligned");

// 初步设计是一个线程一个这个结构
typedef struct {
  int contextCount;
  int *contexts;
} CollExecContext;

typedef struct {
  SQ *sq;
  CQ *cq;
  cudaStream_t stream;
  int cudaDev;
  dim3 gridDim;
  dim3 blockDim;
  int collCount;
  CQE *globalCqes;
  int *globalBlkCount4Coll;
  int *globalThrdCount4Coll;
  int *globalCollIds;
  DevComm7WorkElem *globalDevComm7WorkElems;
  ofcclShmemData *globalBlk2Coll2Shmem;
  CollExecContext *globalCollExecContext;
} KernelThrdArgs;

__global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, ofcclShmemData *globalBlk2Coll2Shmem);




#endif // End include guard
