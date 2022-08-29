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
#include "collectives_ofccl.h"
#include <cuda_runtime.h>
#include <unordered_map>

#define MAX_ASYNC_PANELS 32
#define MAX_ASYNC_OPS 128
// 10000应该是大于大多数任务中会使用的集合通信的数目了。
#define MAX_LENGTH 10000
// 队列长度搞大些，反正目前也不缺这点显存。
#define QLen 10000
#define tempPrintRound 100000

struct ofcclCommArgs {
  ncclResult_t ret;
  ncclComm_t comm;
};

ncclResult_t ofcclEnqueueCheck(struct ncclInfo* info);
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
__device__ int sqRead(SQ *sq, unsigned long long int sqReadFrontier, SQE *target, int *BlkCount4Coll, int thrdCudaDev); // read 1 element each time
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
  SQ *sq;
  CQ *cq;
  CQE *cqes;
  int *BlkCount4Coll;
  cudaStream_t stream;
  int cudaDev;
  dim3 gridDim;
  dim3 blockDim;
} ThrdArgs;

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

__global__ void daemonKernel(SQ *sq, CQ *cq, CQE *cqes, int *BlkCount4Coll, int thrdCudaDev);










#endif // End include guard
