/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "debug.h"
#include "enqueue_ofccl.h"
#include "info.h"
#include "nccl.h"
#include <pthread.h>

extern thread_local SQ *sq;
extern thread_local CQ *cq;

NCCL_API(ncclResult_t, ofcclPrepareAllReduce, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId);
ncclResult_t ofcclPrepareAllReduce(size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    nullptr, nullptr, count, datatype, op, 0, comm, nullptr, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ofcclPrepareCollComm(&info, collId);
  
}

NCCL_API(ncclResult_t, ofcclRunAllReduce, const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs);
ncclResult_t  ofcclRunAllReduce(const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs) {

  SQE sqe = { collId, 0, -1, sendbuff, recvbuff, false };
  int thrdCudaDev;
  checkRuntime(cudaGetDevice(&thrdCudaDev));
  
  OFCCL_LOG_RANK_0(OFCCL, "<%lu> rank=%d ofcclRunAllReduce, sendbuff @ %p, recvbuff @ %p", pthread_self(), thrdCudaDev, sendbuff, recvbuff);
  // OFCCL_LOG_RANK_0(OFCCL, "<%lu> rank=%d Enter ofcclRunAllReduce", pthread_self(), thrdCudaDev);

  while (sqWrite(sq, &sqe, thrdCudaDev, callback, callbackArgs) == -1) {

  }
  // OFCCL_LOG_RANK_0(OFCCL, "<%lu> rank=%d insert sqe for collId %d", pthread_self(), thrdCudaDev, collId);

  return ncclSuccess;
}
