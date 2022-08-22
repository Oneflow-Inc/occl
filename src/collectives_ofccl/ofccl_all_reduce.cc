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

void *startClient(void *args) {
  SQ *sq = ((ClientThrdArgs *)args)->sq;
  CQ *cq = ((ClientThrdArgs *)args)->cq;
  // collId是ofcclRunAllReduce的调用者传进来的，更加用户的代码，知道自己用的comm的编号。
  int collId = ((ClientThrdArgs *)args)->collId;
  const void *send_buffer = ((ClientThrdArgs *)args)->send_buffer;
  void *recv_buffer = ((ClientThrdArgs *)args)->recv_buffer;
  int requestCount = ((ClientThrdArgs *)args)->requestCount;
  // cudaStream_t stream = ((ClientThrdArgs *)args)->stream;

  for (int i = 0; i < requestCount; i++) {
    SQE sqe = { collId, 0, -1, send_buffer, recv_buffer, false };
    while (sqWrite(sq, &sqe) == -1) {

    }
    // OFCCL_LOG(OFCCL, "<%lu> insert %dth sqe for collId %d", pthread_self(), i, collId);

    while (true) {
      CQE target;
      if (cqRead(cq, &target, collId) == -1) {

      } else {
        // OFCCL_LOG(OFCCL, "<%lu> get %dth cqe for collId %d", pthread_self(), i, collId);
        break;
      }
    }
  }

  return nullptr;
}

ncclResult_t  ofcclRunAllReduce(const void* sendbuff, void* recvbuff, int collId) {
  pthread_t clientThrd;
  // cudaStream_t clientStream;
  ClientThrdArgs args;

  // checkRuntime(cudaStreamCreate(&clientStream));
  // 这里调用时，ofcclRunAllReduce应该只对应一个request。
  // args = { sq, cq, collId, sendbuff, recvbuff, 1, clientStream };
  args = { sq, cq, collId, sendbuff, recvbuff, 1, nullptr };
  pthread_create(&clientThrd, NULL, startClient, &args);
  OFCCL_LOG(OFCCL, "<%lu> create client for collId %d", pthread_self(), collId);

  pthread_join(clientThrd, nullptr);

  return ncclSuccess;
}
