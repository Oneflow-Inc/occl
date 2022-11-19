#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

#define buffPrintNum 5
#define buffPrintStart 4095 
// 4096 : 32K的一半
// 134217728 : 1G的一半

// 跑几次 traverseTaskQ 后才去 checkSQ7TidyTaskQ
#define TRAVERSE_TIMES 10
#define TOLERANT_FAIL_CHECK_SQ_CNT 500

#define CNT_BEFORE_QUIT 5

#define DevRingBufferGetFrontier(B, frontier) ((B)->buffer + (frontier % (B)->length))

#define DevRingBufferLogicFrontier(B, frontier) (frontier % (B)->length)

inline __device__ bool DevCqFull(CQ *cq) { // cq->head 由CPU维护。
  volatile unsigned long long int *headPtr = &(cq->head);
  volatile unsigned long long int *tailPtr = &(cq->tail);
  // return *headPtr % cq->length == (*tailPtr + gridDim.x) % cq->length;
  return *tailPtr + gridDim.x - *headPtr == cq->length; // 防止全部block同时更新cq->frontier，破坏cq->head处的没有被读取的cqe。// 不再使用取模计算，直接用相同语义的加减计算
}

inline __device__ bool DevSqEmpty(SQ *sq, unsigned long long int currSqFrontier) {
  volatile unsigned long long int *tailPtr = &(sq->tail);
  return *tailPtr == currSqFrontier;
}

inline __device__ unsigned long long int DevLogicSqTail(SQ *sq) {
  volatile unsigned long long int *tailPtr = &(sq->tail);
  return *tailPtr % sq->length;
}

inline __device__ unsigned long long int DevLogicSqHead(SQ *sq) {
  volatile unsigned long long int *headPtr = &(sq->head);
  return *headPtr % sq->length;
}

// Don't use barrier 0 as it's used by the final sync
inline __device__ void ofcclBarrier(int barId, int numThreads=blockDim.x) {
  asm volatile("bar.sync %0, %1;" :: "r"(barId), "r"(numThreads));
}
