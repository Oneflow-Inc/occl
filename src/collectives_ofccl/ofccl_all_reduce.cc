/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue_ofccl.h"
#include "info.h"
#include "nccl.h"

NCCL_API(ncclResult_t, ofcclPrepareAllReduce, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId);
ncclResult_t ofcclPrepareAllReduce(size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  // struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
  //   nullptr, nullptr, count, datatype, op, 0, comm, nullptr, /* Args */
  //   ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  // return ofcclPrepareCollComm(&info, collId);
  
  // TODO: delete *info.
  struct ncclInfo *info = new struct ncclInfo();
  info->coll = ncclFuncAllReduce;
  info->opName = "AllReduce";
  info->sendbuff = nullptr;
  info->recvbuff = nullptr;
  info->count = count;
  info->datatype = datatype;
  info->op = op;
  info->root = 0;
  info->comm = comm;
  info->stream = nullptr;
  info->chunkSteps = ALLREDUCE_CHUNKSTEPS;
  info->sliceSteps = ALLREDUCE_SLICESTEPS;
  return ofcclPrepareCollComm(info, collId);
}


ncclResult_t  ofcclRunAllReduce(const void* sendbuff, void* recvbuff, int collId) {

  return ncclSuccess;
}
