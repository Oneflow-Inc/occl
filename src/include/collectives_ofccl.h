#ifndef OFCCL_COLLECTIVES_H_
#define OFCCL_COLLECTIVES_H_

#include "collectives.h"
#include "devcomm.h"
#include <pthread.h>

// #define MAX_LENGTH 587 // 587不超0xc000 shmem的限制，588就超了(0xc00c)，这时在启用enqueue_ofccl_dev.cu里边全部3个shared数组的情况下，假如587不够用，可以考虑根据使用频率删除其中的几个；编译器会优化掉没使用的static shared声明，测量时候要注意。
#define MAX_LENGTH 128 // TODO: 先搞小一点，开发之后再优化
// 队列长度搞大些，反正目前也不缺这点显存。就搞得和max collCount一样大，那就不会full了。
#define QLen MAX_LENGTH
#define tempPrintRound 100000

#define RingBuffer_full(B) ((((B)->tail + 1) % (B)->length) == ((B)->head % (B)->length))

#define RingBuffer_empty(B) (((B)->tail % (B)->length) == ((B)->head % (B)->length))

#define RingBuffer_get_head(B) ((B)->buffer + ((B)->head % (B)->length))
#define RingBuffer_get(B, frontier) ((B)->buffer + (frontier% (B)->length))

#define RingBuffer_get_tail(B) ((B)->buffer + ((B)->tail % (B)->length))

#define RingBuffer_logic_head(B) ((B)->head % (B)->length)
#define GetLogicFrontier(B, frontier) (frontier % (B)->length)

#define RingBuffer_logic_tail(B) ((B)->tail % (B)->length)

// 对于device和CPU上的commit分别进行不同的多线程保护。
#define RingBuffer_commit_read(B, A) ((B)->head = ((B)->head + (A)) % (B)->length)

#define RingBuffer_commit_write(B, A) ((B)->tail = ((B)->tail + (A)) % (B)->length)

#define testBlkCnt4Coll(i) i % 2 == 0 ? daemonKernelGridDim.x : daemonKernelGridDim.x - 1

// static thread_local int CPUSleep = 0;
// __device__ static thread_local int GPUSleep = 0;
// static thread_local int CpuSleepUs = 1e6;
// __device__ static thread_local clock_t GpuSpin = 1e9 * 2;
// #define GpuSpin4Bid(i) (1 + i) * 1e9
// #define GpuSpin4BidSmall(i) (6 + i) * 1e6

// SQ read by device, written by host; CQ read by host, written by device;
typedef struct {
  int collId;
  int counter;
  int logicHead;

  const void *sendbuff;
  void *recvbuff;

  bool quit;
} SQE;

typedef struct {
  SQE *buffer;
  int length;
  unsigned long long int head;
  unsigned long long int tail;
  pthread_mutex_t mutex;
} SQ;

typedef struct {
  int collId;
  int counter;
} CQE;

typedef struct {
  CQE *buffer;
  int length;
  unsigned long long int head;
  unsigned long long int tail;
  pthread_mutex_t mutex;
} CQ;

struct DevComm7WorkElem {
  struct ncclDevComm* comm;
  ncclWorkElem first;
};

// 初步设计是一个线程一个这个结构
typedef struct {
  int contextCount;
  int *contexts;
} CollExecContext;

typedef struct {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
} CollCtxGroup;

// sizeof(CollCtx)=42104, sizeof(CollCtxGroup)=248, sizeof(ncclDevComm)=40, sizeof(ncclChannel)=512, sizeof(ncclWork)=512
typedef struct {
  union {
    uint64_t ll128warp[NCCL_LL128_MAX_NTHREADS/WARP_SIZE][NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE]; // 这个占得大，占了40960
    CollCtxGroup groups[NCCL_MAX_GROUPS]; // 这个只占了3968
  };
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  struct ncclDevComm comm;
  struct ncclChannel channel;
  uint64_t pad;
  struct ncclWork work;
  // 代表当前的表项对应的集合通信被调用，还没有执行完成。初始化置0；发现了相应的sqe之后置1；执行完成后置0。
  int executing;
  // int numDoneThrds;
} CollCtx;
static_assert(offsetof(CollCtx, work)%16 == 0, "shmem.work needs to be 16B aligned");

extern __global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx);
// ***** 先不要定义ofccl版本的ncclDevRedOp_t, ncclDevRedOpFull, 这个在其他地方有使用 *****

// ***** 保留FUNC_INDEX *****

// ***** 现在应该只需要个funcName *****
// TODO: 可能需要初始化和真正执行两份function name；也可能只搞一个function，不过通过参数区分是初始化还是真正执行。倾向于后者。
#define OFCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  ofcclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define OFCCL_ONERANK_REDUCE_NAME(devredop, type) \
  ofcclFunction_OneRankReduce_##devredop##_##type

// ***** 我们只需要function，不需要kernel，即只需要device，不需要global *****
#define OFCCL_DECL5(func, algo, proto, devredop, type) \
  extern __device__ void OFCCL_FUNC_NAME(func, algo, proto, devredop, type)();

#define OFCCL_DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, OFCCL_DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, OFCCL_DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, OFCCL_DECL5(func, algo, LL128,  devredop, type))

#define OFCCL_DECL3(func, devredop, type, undef) \
  OFCCL_DECL4(func, RING,    devredop, type, undef) \
  OFCCL_DECL4(func, TREE,    devredop, type, undef) \
  OFCCL_DECL4(func, COLLNET, devredop, type, undef)

#if defined(__CUDA_BF16_TYPES_EXIST__)
#define OFCCL_DECL2(func, devredop, undefForFloat) \
  OFCCL_DECL3(func, devredop, int8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, __nv_bfloat16, /*undef=*/undefForFloat)
#else
#define OFCCL_DECL2(func, devredop, undefForFloat) \
  OFCCL_DECL3(func, devredop, int8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, double, /*undef=*/undefForFloat)
#endif

#define OFCCL_DECL(func) \
  OFCCL_DECL2(func, Sum, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, Prod, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, Min, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, Max, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, SumPostDiv, /*undefForFloat=*/1)

OFCCL_DECL2(Broadcast, Sum, /*undefForFloat=*/0)
OFCCL_DECL(Reduce)
OFCCL_DECL2(AllGather, Sum, /*undefForFloat=*/0)
OFCCL_DECL(ReduceScatter)
OFCCL_DECL(AllReduce)
OFCCL_DECL5(SendRecv, RING, SIMPLE, Sum, int8_t)


extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, half)();
#if defined(__CUDA_BF16_TYPES_EXIST__)
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, __nv_bfloat16)();
#endif
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, float)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, double)();

#endif