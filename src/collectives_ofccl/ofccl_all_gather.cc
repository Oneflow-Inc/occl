#include "debug.h"
#include "enqueue_ofccl.h"
#include "info.h"
#include "nccl.h"
#include <pthread.h>

NCCL_API(ncclResult_t, ofcclPrepareAllGather, size_t sendcount, ncclDataType_t datatype, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclPrepareAllGather(size_t sendcount, ncclDataType_t datatype, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    nullptr, nullptr, sendcount, datatype, ncclSum, 0, comm, nullptr, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ofcclPrepareCollComm(&info, collId, rankCtx);
  
}

NCCL_API(ncclResult_t, ofcclRunAllGather, const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx);
ncclResult_t  ofcclRunAllGather(const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx) {

  SQE sqe = { collId, 0, sendbuff, recvbuff, false };
  int thrdCudaDev;
  checkRuntime(cudaGetDevice(&thrdCudaDev));

  ofcclInsert7UpdateProxy(collId, rankCtx);

  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> ofcclRunAllGather, sendbuff @ %p, recvbuff @ %p", pthread_self(), thrdCudaDev, sendbuff, recvbuff);
  // OFCCL_LOG_RANK_0(OFCCL, "<%lu> Rank<%d> Enter ofcclRunAllGather", pthread_self(), thrdCudaDev);

  while (sqWrite(rankCtx->sq, &sqe, thrdCudaDev, callback, callbackArgs, rankCtx) == -1) {}
  
  // OFCCL_LOG_RANK_0(OFCCL, "<%lu> Rank<%d> insert sqe for coll_id = %d", pthread_self(), thrdCudaDev, collId);

  return ncclSuccess;
}
