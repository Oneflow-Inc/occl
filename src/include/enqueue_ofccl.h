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

ncclResult_t ofcclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId);

#endif // End include guard
