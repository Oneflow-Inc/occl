/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef OFCCL_ENQUEUE_H_
#define OFCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives_ofccl.h"
#include <cuda_runtime.h>

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

#define testBlkCnt4Coll(i) i % 2 == 0 ? deamonKernelGridDim.x : deamonKernelGridDim.x - 1

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
__device__ int sqRead(SQ *sq, unsigned long long int readFrontier, SQE *target, int *BlkCount4Coll); // read 1 element each time
int sqWrite(SQ *sq, SQE *sqe);

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
// CQ read by host, written by device; TODO: return ncclResult_t
int cqRead(CQ *cq, CQE *target, int collId); // read 1 element each time
__device__ int cqWrite(CQ *cq, CQE *cqe);

typedef struct {
  SQ *sq;
  CQ *cq;
  CQE *cqes;
  int *BlkCount4Coll;
  cudaStream_t stream;
  int cudaDev;
} ThrdArgs;

typedef struct {
  SQ *sq;
  CQ *cq;
  int collId;
  const void *send_buffer;
  void *recv_buffer;
  int requestCount;
  cudaStream_t stream;
} ClientThrdArgs;


// TODO: 需要thread local？

__global__ void deamonKernel(SQ *sq, CQ *cq, CQE *cqes, int *BlkCount4Coll);










#endif // End include guard
