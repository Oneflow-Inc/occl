#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

typedef struct {
  int quit;
  int numActiveColls;
  int currActiveCollId;
  unsigned long long int sqReadFrontier; // 每个block的0号线程操作
} BlkStatus;

// 跑几次 traverseGlobalCollCtx 后才去 checkSQ
#define TRAVERSE_TIMES 3