#include "collectives_ofccl.h"
#include "op128_ofccl.h"
#include "common_ofccl.h" // for CollCtx

#define COPY_ELEM_SIZE 16
#define CHAR_ELEM_SIZE sizeof(char) // CollStatus
#define SHORT_ELEM_SIZE sizeof(short) // ActiveCollIds

#define DevRingBufferGetFrontier(B, frontier) ((B)->buffer + (frontier % (B)->length))

#define DevRingBufferLogicFrontier(B, frontier) (frontier % (B)->length)

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

#ifdef DEBUG_CLOCK
inline __device__ long long int calcDeltaClock(long long int start, long long int end) {
  return end > start ? end - start : end + (0xffffffffffffffff - start);
}
#endif

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
