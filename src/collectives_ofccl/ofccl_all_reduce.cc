/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue_ofccl.h"

NCCL_API(ncclResult_t, ofcclPrepareAllReduce, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId);
ncclResult_t ofcclPrepareAllReduce(size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    nullptr, nullptr, count, datatype, op, 0, comm, nullptr, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ofcclPrepareCollComm(&info, collId);
}
