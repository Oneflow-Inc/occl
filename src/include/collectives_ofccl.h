#ifndef OFCCL_COLLECTIVES_H_
#define OFCCL_COLLECTIVES_H_

#include "collectives.h"
#include "devcomm.h"
#include <cstdint>
#include <pthread.h>
#include <sys/types.h>

#define MAX_LENGTH 1000LL // 受到0xc000 shmem的限制
// 队列长度搞大些，反正目前也不缺这点显存。就搞得和max collCount一样大，那就不会full了。
#define QLen MAX_LENGTH

#define SHOW_CNT 1

// #define ARRAY_DEBUG 1

#define NUM_BARRIERS 30
#define BARCNT_INNER_SIZE 10
#define PrintTestQNum 10
#define COLL_COUNTER_INNER_SIZE 10

// #define CQE_DEBUG_RANK_X 0

// #define CQE_DEBUG_ALL_RANK 1

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

  const void *sendbuff;
  void *recvbuff;

  bool quit;
} SQE;

typedef struct {
  SQE *buffer;
  unsigned long long int length;
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
  unsigned long long int length;
  unsigned long long int head;
  unsigned long long int tail;
  unsigned long long int frontier;
  pthread_mutex_t mutex;
} CQ;

struct DevComm7WorkElem {
  struct ncclDevComm* comm;
  // TODO: 或许会有扩展性问题。
  ncclWorkElem first;
};

// 初步设计是一个线程一个这个结构
typedef struct {
  int contextCount;
  int *contexts;
} CollExecContext;

typedef struct alignas(16) {
  unsigned long long int sqReadFrontier; // 每个block的0号线程操作
  #ifdef SHOW_CNT
    unsigned long long int totalCtxSaveCnt; // 统计信息，测量绝对性能的时候考虑删掉。
    unsigned long long int totalCtxLoadCnt;
    unsigned long long int totalProgressed7SwithchCnt;
    unsigned long long int totalUnprogressedQuitCnt;
  #endif
  int numActiveColls;

} DynamicBlkStatus;

typedef struct alignas(16) {
  // 加载这个数组时，元素都设成0就好
  char collStatus[MAX_LENGTH]; // 0：没在执行；1：正在执行；2：执行完成；-2：switch且没有progress；-1：switch但有progress
} CollStatusAlign;

typedef struct alignas(16) {
  short activeCollIds[MAX_LENGTH];
} ActiveCollIdsAlign;

typedef struct alignas(16) {
  /* ****** 根据hasQuitted的值，决定重置还是从globalMem里读 ****** */
  DynamicBlkStatus dynamicBlkStatus;


  /* ****** 根据宏单独处理 ****** */ 
  #ifdef ARRAY_DEBUG
    unsigned long long int *barrierCnt;
    unsigned long long int *collCounters;
  #endif


  /* ****** 数组有单独复制 ****** */
  // 这样的好处是以16B复制的时候，没有越界风险
  CollStatusAlign collStatusAlign;
  ActiveCollIdsAlign activeCollIdsAlign;


  /* ****** 固定从globalMem里读 ****** */
  int hasQuitted; // 记录曾经Quit过的状态，一旦被设置，就不再清零。


  /* ****** daemonKernel每次启动需要重置 ****** */
  int quit;
  int currLoadedCollId;
} BlkStatus;

typedef struct {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
} CollCtxGroup;

// sizeof(CollCtx)=42104, sizeof(CollCtxGroup)=248, sizeof(ncclDevComm)=40, sizeof(ncclChannel)=512, sizeof(ncclWork)=512
// 准备抛弃旧的collCtx结构，只保留我们需要的。
typedef struct alignas(16) {
  // TODO: 对LL、LL128的支持

  /* ****** 每次执行需要重置 ****** */
  int saveCtx7Quit;


  /* ****** 每次load需要重置、加载 ****** */
  int progressed;
  struct ncclWorkElem workElem;
  // 来自channel.ring
  int ringPrev;
  int ringNext;
  int ringIndex;
  // 来自channel
  struct ncclPeer* devPeers;
  // 来自comm(devComm, 不是普通comm)
  int rank; // 原来来自于comm.rank，还是放在collCtx而不是blkStatus里，因为在不同的集合通信中，一个设备的rank可能会变，不应该静态保存。
  int nRanks;
  volatile uint32_t *abortFlag;
  // Prims Simple的上下文
  int loadAgain; // 是不是曾经执行了一半，被换出去了，这次是又一次执行。主要用来控制ofccl/src/collectives_ofccl/device/ofccl_prims_simple.h里loadConn时候的roundUp行为，防止异常更新自己的step(head/tail)。正式一点可以搞个issue记录问题，然后在commit里说fix issue。懒得搞了。这个变量是只要曾经被换出去过，就一直是1了，这样每次创建prim，loadConn的时候，才可以都跳过roundUp。
  int slice4SimpleGenericOp;
  int offset4SimpleGenericOp;
  int64_t ctxSwitchThreshold;
  // Ring AllReduce的上下文
  int currentStep4RingAllReduce;
  ssize_t gridOffset4RingAllReduce;
  #if defined(CQE_DEBUG_RANK_X) || defined(CQE_DEBUG_ALL_RANK)
    unsigned long long sqeReadCnt;
    unsigned long long cqeWriteCnt;
    unsigned long long cqePrepareCnt;
  #endif

  /* ****** 不需要load和save ****** */ 
  // 这两个是启动相应的coll的执行之后，Primitive构造函数里填充的
  CollCtxGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  int buffSizes[NCCL_NUM_PROTOCOLS]; // 来自comm(devComm, 不是普通comm)

} CollCtx;

extern __global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, char *globalBlkCount4Coll, int *globalThrdCount4Coll, short *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int *finallyQuit, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters, const int64_t TRAVERSE_TIMES, const int64_t TOLERANT_UNPROGRESSED_CNT, const int64_t BASE_CTX_SWITCH_THRESHOLD, const int64_t BOUNS_SWITCH_4_PROCESSED_COLL);
// ***** 先不要定义ofccl版本的ncclDevRedOp_t, ncclDevRedOpFull, 这个在其他地方有使用 *****

// ***** 保留FUNC_INDEX *****

// ***** 现在应该只需要个funcName *****
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