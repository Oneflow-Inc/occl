#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

#define buffPrintNum 5
#define buffPrintStart 4095 
// 4096 : 32K的一半
// 134217728 : 1G的一半

// 跑几次 traverseGlobalCollCtx 后才去 checkSQ
#define TRAVERSE_TIMES 10
#define TOLERANT_FAIL_CHECK_SQ_CNT 5

#define OFCCL_SYNC_ALL_BAR_ID 4
#define OFCCL_SYNC_COLL_WORKER_BAR_ID 6
// Don't use barrier 0 as it's used by the final sync
inline __device__ void ofcclBarrier(int barId, int numThreads=blockDim.x) {
  asm volatile("bar.sync %0, %1;" :: "r"(barId), "r"(numThreads));
}
