#include "debug.h"
#include "enqueue_ofccl.h"
#include "info.h"
#include "nccl.h"
#include <pthread.h>

NCCL_API(ncclResult_t, ofcclPrepareBroadcast, size_t count, ncclDataType_t datatype, int root, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclPrepareBroadcast(size_t count, ncclDataType_t datatype, int root, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    nullptr, nullptr, count, datatype, ncclSum, root, comm, nullptr, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ofcclPrepareCollComm(&info, collId, rankCtx);
  
}

NCCL_API(ncclResult_t, ofcclRunBroadcast, const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx);
ncclResult_t  ofcclRunBroadcast(const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx) {

  SQE sqe = { collId, 0, sendbuff, recvbuff, false };
  int thrdCudaDev;
  checkRuntime(cudaGetDevice(&thrdCudaDev));


  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> ofcclRunBroadcast, sendbuff @ %p, recvbuff @ %p", pthread_self(), thrdCudaDev, sendbuff, recvbuff);
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> Enter ofcclRunBroadcast", pthread_self(), thrdCudaDev);

  while (sqWrite(rankCtx->sq, &sqe, thrdCudaDev, callback, callbackArgs, rankCtx) == -1) {}
  
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> insert sqe for coll_id = %d", pthread_self(), thrdCudaDev, collId);

  return ncclSuccess;
}

/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ofcclPrepareBcast, size_t count, ncclDataType_t datatype, int root, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclPrepareBcast(size_t count, ncclDataType_t datatype, int root, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) { // prepare的接口完全一致
  return ofcclPrepareBroadcast(count, datatype, root, comm, collId, rankCtx);
}
NCCL_API(ncclResult_t, ofcclRunBcast, void* buff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx);
ncclResult_t  ofcclRunBcast(void* buff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx) { // run把两个buff换成一个。
  return ofcclRunBroadcast(buff, buff, collId, callback, callbackArgs, rankCtx);
}
