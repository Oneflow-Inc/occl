#include "debug.h"
#include "enqueue_ofccl.h"
#include "info.h"
#include "nccl.h"
#include <pthread.h>

NCCL_API(ncclResult_t, ofcclPrepareReduce, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclPrepareReduce(size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    nullptr, nullptr, count, datatype, op, root, comm, nullptr, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ofcclPrepareCollComm(&info, collId, rankCtx);
  
}

NCCL_API(ncclResult_t, ofcclRunReduce, const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx);
ncclResult_t  ofcclRunReduce(const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx) {

  SQE sqe = { collId, 0, sendbuff, recvbuff, false };
  int thrdCudaDev;
  checkRuntime(cudaGetDevice(&thrdCudaDev));


  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> ofcclRunReduce, sendbuff @ %p, recvbuff @ %p", pthread_self(), thrdCudaDev, sendbuff, recvbuff);
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> Enter ofcclRunReduce", pthread_self(), thrdCudaDev);

  while (sqWrite(rankCtx->sq, &sqe, thrdCudaDev, callback, callbackArgs, rankCtx) == -1) {}
  
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> insert sqe for coll_id = %d", pthread_self(), thrdCudaDev, collId);

  return ncclSuccess;
}