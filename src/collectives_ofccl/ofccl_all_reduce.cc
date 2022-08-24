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

NCCL_API(ncclResult_t, ofcclPrepareAllReduce, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId);
ncclResult_t ofcclPrepareAllReduce(size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    nullptr, nullptr, count, datatype, op, 0, comm, nullptr, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ofcclPrepareCollComm(&info, collId);
  
}

NCCL_API(ncclResult_t, ofcclRunAllReduce, const void* sendbuff, void* recvbuff, int collId);
ncclResult_t  ofcclRunAllReduce(const void* sendbuff, void* recvbuff, int collId) {

  SQE sqe = { collId, 0, -1, sendbuff, recvbuff, false };
  int cudaDev;
  checkRuntime(cudaGetDevice(&cudaDev));
  
  OFCCL_LOG(OFCCL, "<%lu> rank=%d Enter ofcclRunAllReduce", pthread_self(), cudaDev);

  while (sqWrite(sq, &sqe) == -1) {

  }
  OFCCL_LOG(OFCCL, "<%lu> rank=%d insert sqe for collId %d", pthread_self(), cudaDev, collId);

  while (true) {
    CQE target;
    if (cqRead(cq, &target, collId) == -1) {

    } else {
      OFCCL_LOG(OFCCL, "<%lu> rank=%d get cqe for collId %d", pthread_self(), cudaDev, collId);
      break;
    }
  }

  return ncclSuccess;
}
