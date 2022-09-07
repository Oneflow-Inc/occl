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

#include <cstddef>
#include <cuda_runtime.h>
#include <unordered_map>

// #define MAX_ASYNC_PANELS 32
// #define MAX_ASYNC_OPS 128

extern ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId);

extern int sqWrite(SQ *sq, SQE *sqe, int thrdCudaDev, CallbackFunc callback, void *callbackArgs);

struct ofcclCommArgs {
  ncclResult_t ret;
  ncclComm_t comm;
};

typedef struct {
  int *poll_start;
  int *poll_stop;
  // std::unordered_map<int, CallbackFunc> *collId2callback;
  CallbackFunc *callbacks;
  int cudaDev;
  CQ *cq;
  void **callbackArgList;
} PollerArgs;

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





#endif // End include guard
