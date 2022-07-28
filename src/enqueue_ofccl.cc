/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue_ofccl.h"
#include "argcheck.h"
#include "bootstrap.h"
#include "channel.h"
#include "coll_net.h"
#include "debug.h"
#include "gdrwrap.h"
#include "group.h"
#include "nccl.h"
#include "transport.h"

#include <cstring> // std::memcpy

static void *const ofcclKerns[1] = {
    (void *)try_make_kern,
};

namespace {
void try_make() {
  dim3 gridDim, blockDim;
  gridDim.x = 8;
  blockDim.x = 4;
  int a = 1;
  int *b = &a;
  cudaLaunchKernel(ofcclKerns[0], gridDim, blockDim, (void **)&b, 0, NULL);
}

} // namespace

#define MAX_ASYNC_PANELS 32
#define MAX_ASYNC_OPS 128

struct ofcclCommArgs {
  ncclResult_t ret;
  ncclComm_t comm;
};

thread_local ofcclCommArgs ofcclCommList[MAX_ASYNC_PANELS][MAX_ASYNC_OPS];
thread_local pthread_t ofcclPrepareThreads[MAX_ASYNC_PANELS][MAX_ASYNC_OPS];
thread_local int ofcclCommListPanel = 0;
thread_local int ofcclCommListFront = 0;

NCCL_API(ncclResult_t, ofcclPrepareCollComm, struct ncclInfo *info, int collId);
ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId) {
  memset(ofcclCommList, 0,
         sizeof(ncclComm_t) * MAX_ASYNC_PANELS *
             MAX_ASYNC_OPS); // legacy in ncclGroupStart
  
  ncclResult_t ret = ncclSuccess;
  // Check arguments
  info->comm->checkPointers = false; // we do not assign buff pointers yet.
  NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
  NCCLCHECKGOTO(ArgsCheck(info), ret, end);

  // Copy reduction op state from op handle into info struct here since the
  // op handle may be destroyed before ncclGroupEnd().
  // ***** Ignore for now. *****
  // NCCLCHECKGOTO(hostToDevRedOp(&info->opFull, info->op, info->datatype,
  // info->comm), ret, end);

  // ***** ncclAsyncColl(info->comm) *****

  ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm = info->comm;
  // OFCCL_LOG(OFCCL, "i = %d, j = %d, tempcomm=%p(at %p), info->comm=%p(at %p), tempcomm->nRanks = %d, tempcomm->connect=%d", ofcclCommListPanel, ofcclCommListFront, ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm, &(ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm), info->comm, &(info->comm), ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm->nRanks, ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm->connect);

  ofcclCommListFront++;
  if (ofcclCommListFront >= MAX_ASYNC_OPS) {
    ofcclCommListFront = 0;
    ofcclCommListPanel++;
    if (ofcclCommListPanel >= MAX_ASYNC_PANELS) {
      WARN("Too many async operations in progress, max is %d",
           MAX_ASYNC_PANELS * MAX_ASYNC_OPS);
      ret = ncclInvalidUsage;
      goto end;
    }
  }

  // ***** Ignore *****
  // NCCLCHECKGOTO(checkSetStream(info), ret, end);

  // ***** ncclSaveAsyncColl(info) *****

  if (info->comm->asyncOpCount >= NCCL_MAX_OPS) {
    WARN("Too many async operations in progress, max is %d", NCCL_MAX_OPS);
    ret = ncclInvalidUsage;
    goto end;
  }

  memcpy(info->comm->asyncOps + info->comm->asyncOpCount, info,
         sizeof(struct ncclInfo));
  info->comm->asyncOpCount++;
  info->comm->asyncTotalSize += info->nBytes;

end:
  
  return ret;
}

void *ofcclAsyncThreadPreconnect(void* args_) {
  struct ofcclCommArgs* args = (struct ofcclCommArgs*)args_;
  CUDACHECKTHREAD(cudaSetDevice(args->comm->cudaDev));
  if (CPU_COUNT(&args->comm->cpuAffinity))
    sched_setaffinity(0, sizeof(cpu_set_t), &args->comm->cpuAffinity);
  NCCLCHECKTHREAD(ncclTransportP2pSetup(args->comm, NULL, 1));
  return args;
}

NCCL_API(ncclResult_t, ofcclPrepareDone);
ncclResult_t ofcclPrepareDone() {
  // ***** ncclGroupEnd() *****

  int savedDev;
  CUDACHECK(cudaGetDevice(&savedDev));
  // OFCCL_LOG(OFCCL, "savedDev is %d", savedDev);
  ncclResult_t ret = ncclSuccess;

  int front_of_panel = -1;

  // ***** ncclAsyncThreadPreconnect threads *****
  for (int i = 0; i <= ofcclCommListPanel; i++) {
    front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_PANELS : ofcclCommListFront;
    for (int j = 0; j < front_of_panel; j++) {
      ofcclCommArgs args = ofcclCommList[i][j];
      // OFCCL_LOG(OFCCL, "i=%d, j=%d, tempcomm=%p(at %p)", i, j, ofcclCommList[i][j].comm, &(ofcclCommList[i][j].comm));
      // OFCCL_LOG(OFCCL, "i=%d, j=%d, args.comm->connect=%d", i, j, args.comm->connect);
      if (args.comm->connect) {
        pthread_create(&ofcclPrepareThreads[i][j], NULL, ofcclAsyncThreadPreconnect, &args);
      }
    }
  }
  for (int i = 0; i <= ofcclCommListPanel; i++) {
    front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_PANELS : ofcclCommListFront;
    for (int j = 0; j < front_of_panel; j++) {
      ofcclCommArgs args = ofcclCommList[i][j];
      if (args.comm->connect) {
        int err = pthread_join(ofcclPrepareThreads[i][j], NULL);
        if (err != 0) {
          WARN("Error waiting for pthread_join : %s", strerror(errno));
          return ncclSystemError;
        }
        NCCLCHECKGOTO(args.ret, ret, end);
        args.comm->connect = 0;
        }
    }
  }

end:
  ofcclCommListPanel = 0;
  ofcclCommListFront = 0;
  CUDACHECK(cudaSetDevice(savedDev)); // do other clean-ups first before calling
  return ret;
}

ncclResult_t ofcclEnqueueCheck(struct ncclInfo *info) {
  try_make();
  return ncclSuccess;
}