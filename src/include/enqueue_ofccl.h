/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef OFCCL_ENQUEUE_H_
#define OFCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "debug.h"
#include "collectives_ofccl.h"
#include "nccl.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <semaphore.h>
#include <unordered_map>
#include <unordered_set>

// #define MAX_ASYNC_PANELS 32
// #define MAX_ASYNC_OPS 128

// #define RingBufferFull(B) ((((B)->tail + 1) % (B)->length) == ((B)->head % (B)->length))

// #define RingBufferEmpty(B) ((B)->tail == (B)->head)

#define RingBufferGetHead(B) ((B)->buffer + ((B)->head % (B)->length))

#define RingBufferGetTail(B) ((B)->buffer + ((B)->tail % (B)->length))

// #define RingBufferLogicHead(B) ((B)->head % (B)->length)

#define RingBufferLogicTail(B) ((B)->tail % (B)->length)

inline bool CpuSqFull(SQ *sq) { // sq->head由GPU更新。
  volatile unsigned long long int *headPtr = &(sq->head);
  return sq->tail + 1 - *headPtr == sq->length;
}

inline bool CpuCqEmpty(CQ *cq) { // cq->tail由GPU更新
  volatile unsigned long long int *tailPtr = &(cq->tail);
  return *tailPtr == cq->head;
}

inline unsigned long long int CpuLogicSqHead(SQ *sq) {
  volatile unsigned long long int *headPtr = &(sq->head);
  return *headPtr % sq->length;
}

inline unsigned long long int CpuLogicCqTail(CQ *cq) {
  volatile unsigned long long int *tailPtr = &(cq->tail);
  return *tailPtr % cq->length;
}

extern ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId, ofcclRankCtx_t rankCtx);

extern int sqWrite(SQ *sq, SQE *sqe, int thrdCudaDev, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx);

struct ofcclCommArgs {
  ncclResult_t ret;
  ncclComm_t comm;
};

typedef struct {
  ofcclRankCtx *rankCtx;
} PollerArgs;

typedef struct {
  ofcclRankCtx *rankCtx;
  int64_t TRAVERSE_TIMES;
  int64_t TOLERANT_UNPROGRESSED_CNT;
  int64_t BASE_CTX_SWITCH_THRESHOLD;
  int64_t BOUNS_SWITCH_4_PROCESSED_COLL;
} ObserverThrdArgs;
typedef struct {
  ofcclRankCtx *rankCtx;
} BarrierCntPrinterArgs;

// TODO: 不是线程安全的。
struct ofcclRankCtx {
  int rank;

  int *finallyQuit; // 只有一个int，最后收到quit sqe的时候，由0号block设置。因为startKernel7SqObserver线程里是在cudaStreamQuery返回cudaSuccess，表明kernel运行完退出，才会去查finallyQuit，这时候如果发现finallyQuit=1，那么可以有很大信心认为所有block都是最终退出了。

  sem_t getNewSqeSema;
  int noMoreSqes;
  pthread_mutex_t observer_mutex;
  pthread_t kernel7SqObserver;
  ObserverThrdArgs observerThrdArgs;

  BlkStatus *globalBlkStatus; // 由于quit，需要保存、恢复blkStatus

  ofcclCommArgs ofcclCommList[MAX_LENGTH];
  pthread_t ofcclPrepareThreads[MAX_LENGTH];
  int collCount;
  std::unordered_set<ncclComm_t> seenComms;

  dim3 daemonKernelGridDim;
  dim3 daemonKernelBlockDim;
  int queueLength;
  dim3 gridDim4Coll[MAX_LENGTH];
  dim3 blockDim4Coll[MAX_LENGTH]; // TODO: 这个可能意义不大，考虑删掉。

  void *argsptrs[18];
  cudaStream_t kernelStream;

  CQE hostCqes[MAX_LENGTH];
  CQE *globalCqes;
  int hostBlkCount4Coll[MAX_LENGTH];
  int *globalBlkCount4Coll;
  int hostThrdCount4Coll[MAX_LENGTH];
  int *globalThrdCount4Coll;
  int hostCollIds[MAX_LENGTH];
  int *globalCollIds;
  DevComm7WorkElem hostDevComm7WorkElems[MAX_LENGTH];
  DevComm7WorkElem *globalDevComm7WorkElems;
  CollCtx *globalBlk2CollId2CollCtx;
  
  SQ *sq;
  CQ *cq;

  // for poller thread
  pthread_t poller;
  PollerArgs pollerArgs;
  pthread_mutex_t poller_mutex; // 应该用mutex保护起来，因为这里的变量承担了“实时传递控制命令”的作用，poller线程在反复轮询这个值，而且相应的赋值还不是原子的。与其他单纯记录状态的变量相比，更需要被mutex保护。
  int poll_start;
  int poll_stop;
  
  void *callbackArgList[MAX_LENGTH];
  CallbackFunc callbacks[MAX_LENGTH];

  unsigned long long int *barrierCnt;
  unsigned long long int *collCounters; // 设计为每个block，对每个coll，有一串数
#ifdef ARRAY_DEBUG
  pthread_t barrierCntPrinter;
  BarrierCntPrinterArgs barrierCntPrinterArgs;
#endif
};


#endif // End include guard
