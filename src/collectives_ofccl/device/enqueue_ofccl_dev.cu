#include "enqueue_ofccl_dev.h"

__shared__ CollCtx sharedCollCtx; // 不能static，primitives要用

__shared__ BlkStatus blkStatus; // 取消static，放到prim里边打印log。
// TODO: 下边这几个可以尝试用constant，先不急
__shared__ int sharedBlkCount4Coll[MAX_LENGTH];
__shared__ int sharedThrdCount4Coll[MAX_LENGTH];

__global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int *globalVolunteerQuitCounter, int *finallyQuit, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters, const int64_t TRAVERSE_TIMES, const int64_t TOLERANT_FAIL_CHECK_SQ_CNT, const int64_t CNT_BEFORE_QUIT, const int64_t TOLERANT_UNPROGRESSED_CNT, const int64_t BASE_CTX_SWITCH_THRESHOLD, const int64_t ARRAY_DEBUG, const int64_t SHOW_QUIT_CNT, const int64_t SHOW_SWITCH_CNT, const int64_t SHOW_RUNNING_CNT, const int64_t CQE_DEBUG_RANK_X, const int64_t CQE_DEBUG_ALL_RANK) {

  daemonKernelImpl<TRAVERSE_TIMES, TOLERANT_FAIL_CHECK_SQ_CNT, CNT_BEFORE_QUIT, TOLERANT_UNPROGRESSED_CNT, BASE_CTX_SWITCH_THRESHOLD, ARRAY_DEBUG, SHOW_QUIT_CNT, SHOW_SWITCH_CNT, SHOW_RUNNING_CNT, CQE_DEBUG_RANK_X, CQE_DEBUG_ALL_RANK>(sq, cq, thrdCudaDev, collCount, globalCqes, globalBlkCount4Coll, globalThrdCount4Coll, globalCollIds, globalDevComm7WorkElems, globalBlk2CollId2CollCtx, globalVolunteerQuitCounter, finallyQuit, globalBlkStatus, barrierCnt, collCounters);
}