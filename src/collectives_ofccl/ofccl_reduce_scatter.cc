#include "debug.h"
#include "enqueue_ofccl.h"
#include "info.h"
#include "nccl.h"
#include <pthread.h>

NCCL_API(ncclResult_t, ofcclPrepareReduceScatter, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclPrepareReduceScatter(size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, int collId, ofcclRankCtx_t rankCtx) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    nullptr, nullptr, recvcount, datatype, op, 0, comm, nullptr, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  return ofcclPrepareCollComm(&info, collId, rankCtx);
  
}

NCCL_API(ncclResult_t, ofcclRunReduceScatter, const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx);
ncclResult_t  ofcclRunReduceScatter(const void* sendbuff, void* recvbuff, int collId, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx) {

  SQE sqe = { collId, 0, sendbuff, recvbuff, false };
  int thrdCudaDev;
  checkRuntime(cudaGetDevice(&thrdCudaDev));


  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> ofcclRunReduceScatter, sendbuff @ %p, recvbuff @ %p", pthread_self(), thrdCudaDev, sendbuff, recvbuff);
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> Enter ofcclRunReduceScatter", pthread_self(), thrdCudaDev);

  while (sqWrite(rankCtx->sq, &sqe, thrdCudaDev, callback, callbackArgs, rankCtx) == -1) {}
  
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> insert sqe for coll_id = %d", pthread_self(), thrdCudaDev, collId);

  return ncclSuccess;
}