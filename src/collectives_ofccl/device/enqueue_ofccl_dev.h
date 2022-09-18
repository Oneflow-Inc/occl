#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

#define buffPrintNum 5
#define buffPrintStart 120000 + 5469

typedef struct {
  int quit;
  int numActiveColls;
  int currActiveCollId;
  unsigned long long int sqReadFrontier; // 每个block的0号线程操作

  unsigned long long int totalCtxSwitchCnt; // 统计信息，测量绝对性能的时候考虑删掉。
} BlkStatus;

// 跑几次 traverseGlobalCollCtx 后才去 checkSQ
#define TRAVERSE_TIMES 3