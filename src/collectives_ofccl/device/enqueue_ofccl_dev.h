#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

#define COPY_ELEM_SIZE 16
#define CHAR_ELEM_SIZE sizeof(char) // CollStatus
#define SHORT_ELEM_SIZE sizeof(short) // ActiveCollIds

#define DevRingBufferGetFrontier(B, frontier) ((B)->buffer + (frontier % (B)->length))

#define DevRingBufferLogicFrontier(B, frontier) (frontier % (B)->length)

#define DEBUG_PARA_LD 1
// #define DEBUG_PARA_SV 1

#define ONE_THRD_DO if (tid == 31) {
#define ONE_THRD_DO_END }

// #define ONE_THRD_DO {
// #define ONE_THRD_DO_END }

typedef struct alignas(16) {
  short collIds[MAX_LENGTH];
} IdsAlign;

typedef struct alignas(16) {
  char blkCount4Coll[MAX_LENGTH];
} BlkCount4CollAlign;

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

template<typename Q>
__global__ void qCreateKernel(Q *q) {
  if (threadIdx.x == 0) {
    q->length = QLen;
    q->head = 0;
    q->tail = 0;
  }
}
