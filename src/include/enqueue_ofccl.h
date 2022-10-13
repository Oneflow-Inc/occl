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
#include <unordered_map>
#include <unordered_set>

// #define MAX_ASYNC_PANELS 32
// #define MAX_ASYNC_OPS 128

extern ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId, ofcclRankCtx_t rankCtx);
extern ncclResult_t ofcclFinalizeRankCtx7StartHostThrds(ofcclRankCtx_t rankCtx);
extern ncclResult_t ofcclRunDaemonKernel(ofcclRankCtx_t rankCtx);

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
} KernelThrdArgs;

// TODO: 不是线程安全的。
struct ofcclRankCtx {
  int rank;

  int inited; // TODO: 现阶段没什么用，是之后为了daemonKernel按需启停用的，防止再跑一遍prepareDone里初始化数据结构的代码。
  int daemonKernelStarted;
  int pollerThreadStarted;
  int *volunteerQuit; // 按需启停的状态记录，cudaMallocHost分配。

  ofcclCommArgs ofcclCommList[MAX_LENGTH];
  pthread_t ofcclPrepareThreads[MAX_LENGTH];
  int collCount;
  std::unordered_set<ncclComm_t> seenComms;

  dim3 daemonKernelGridDim;
  dim3 daemonKernelBlockDim;
  int queueLength;
  dim3 gridDim4Coll[MAX_LENGTH];
  dim3 blockDim4Coll[MAX_LENGTH];

  pthread_t kernelThrd;
  void *argsptrs[11];
  cudaStream_t kernelStream;
  KernelThrdArgs kernelThrdArgs;

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
};


#endif // End include guard
