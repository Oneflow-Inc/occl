#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

#define buffPrintNum 5
#define buffPrintStart 4095 
// 4096 : 32K的一半
// 134217728 : 1G的一半

// 跑几次 traverseTaskQ 后才去 checkSQ7TidyTaskQ
#define TRAVERSE_TIMES 10
#define TOLERANT_FAIL_CHECK_SQ_CNT 5

inline __device__ bool CqFull(CQ *cq) { // cq->head 由CPU维护。
  volatile unsigned long long int *headPtr = &(cq->head);
  volatile unsigned long long int *tailPtr = &(cq->tail);
  return *headPtr == *tailPtr;
}

#define DevRingBufferGetFrontier(B, frontier) ((B)->buffer + (frontier % (B)->length))

#define DevRingBufferLogicFrontier(B, frontier) (frontier % (B)->length)

inline __device__ unsigned long long int DevLogicSqTailInline(SQ *sq) {
  volatile unsigned long long int *tailPtr = &(sq->tail);
  return *tailPtr % sq->length;
}

inline __device__ unsigned long long int DevLogicSqHeadInline(SQ *sq) {
  volatile unsigned long long int *headPtr = &(sq->head);
  return *headPtr % sq->length;
}

// Don't use barrier 0 as it's used by the final sync
inline __device__ void ofcclBarrier(int barId, int numThreads=blockDim.x) {
  asm volatile("bar.sync %0, %1;" :: "r"(barId), "r"(numThreads));
}
