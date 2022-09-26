#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

#define buffPrintNum 5
#define buffPrintStart 120000 + 5469

typedef struct {
  int quit; // TODO: 考虑守护者kernel按需启停的时候这里的调整
  int numActiveColls;
  int currActiveCollId;
  unsigned long long int sqReadFrontier; // 每个block的0号线程操作

  unsigned long long int totalCtxSwitchCnt; // 统计信息，测量绝对性能的时候考虑删掉。
} BlkStatus;

// 跑几次 traverseGlobalCollCtx 后才去 checkSQ
#define TRAVERSE_TIMES 1

#define OFCCL_SYNC_ALL_BAR_ID 4
#define OFCCL_SYNC_COLL_WORKER_BAR_ID 6
// Don't use barrier 0 as it's used by the final sync
inline __device__ void ofcclBarrier(int barId, int numThreads) {
  if (numThreads == WARP_SIZE)
    __syncwarp();
  else
    asm volatile("bar.sync %0, %1;" :: "r"(barId), "r"(numThreads));
}
