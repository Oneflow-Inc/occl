#ifndef OFCCL_COLLECTIVES_H_
#define OFCCL_COLLECTIVES_H_

#include "collectives.h"
#include "devcomm.h"
#include "nccl.h"
#include <cstdint>
#include <pthread.h>
#include <sys/types.h>

#define NUM_CQ_SLOT 8
#define RESERVED_GRID_DIM 256
#define INVALID_CQ_SLOT_MASK  0xffffffffffffffff
#define BLOCK_IDX_MASK        0xff00000000000000
#define BLOCK_IDX_BIT         8
#define BLOCK_CNT_MASK        0x00ffffff00000000
#define BLOCK_CNT_BIT         24
#define COLL_ID_MASK          0x00000000ffffffff
#define COLL_ID_BIT           32

extern __constant__ int64_t NUM_TRY_TASKQ_HEAD;
extern __constant__ int64_t RECV_SUCCESS_FACTOR;
extern __constant__ int64_t RECV_SUCCESS_THRESHOLD;

// #define DEBUG_CLOCK 1

// #define DEBUG_CLOCK_TRAIN 1
// #define DEBUG_CLOCK_IO 1
// #define DEBUG_CLOCK_3D 1
// #define DEBUG_CLOCK_3D_HOST 1

// #define DEBUG_CLOCK_CTX 1 // 独立，不依赖DEBUG_CLOCK_IO

#define SHOW_CNT 1
#ifdef SHOW_CNT
extern __constant__ int64_t NUM_ITER_ENV;
#endif

// #define ARRAY_DEBUG 1

#ifdef DEBUG_CLOCK
  #define CLOCK2US_FACTOR 1695.0
  #define NUM_SHMEM_SLOT 1

  #ifdef DEBUG_CLOCK_TRAIN
    #define RECORD_ITER 2
    #define SKIP_WARMUP_ITER 0
    #define MAX_LENGTH 162LL // 受到0xc000 shmem的限制
  #endif
  #ifdef DEBUG_CLOCK_IO
    #define RECORD_ITER 5
    #define SKIP_WARMUP_ITER 3
    #define MAX_LENGTH 2LL // 受到0xc000 shmem的限制
  #endif
  #ifdef DEBUG_CLOCK_CTX
    #define RECORD_ITER 5
    #define MAX_LENGTH 10LL // 受到0xc000 shmem的限制
  #endif
  #ifdef DEBUG_CLOCK_3D
    #define SKIP_WARMUP_ITER 0
    #define MAX_LENGTH 1000LL // 受到0xc000 shmem的限制
    #define NUM_SHMEM_SLOT 1
    #define RESNET_COLL_CNT 161
    #define NUM_ITER 20

    extern __constant__ int *taskQLen4RankBlkIterColl;
    extern __constant__ int *unprogressed7SwitchCnt4RankBlkIterColl;
    extern __constant__ int *progressed7SwitchCnt4RankBlkIterColl;
    extern __constant__ int *unprogressed7SwitchCntTotal4RankBlkIterColl;
    extern __constant__ int *progressed7SwitchCntTotal4RankBlkIterColl;
    extern __constant__ int *collIdInSqe4RankBlkIterColl;
    extern __constant__ int *collId4Cq4RankBlkIterColl;
    extern __constant__ int numColl;

    inline __host__ __device__ int *getSlot(int *ptr, int blk, int iter, int coll_id, int numIter, int collCnt) {
      return ptr + coll_id + iter * collCnt + blk * numIter * collCnt;
    }
    inline int getCollCnt4Blk(int blk) {
      // 2card resnet
      // if (blk == 0) {
      //   return 161;
      // } else if (blk == 1) {
      //   return 52; // 1号block参加52个coll，包括需要2个block的coll和需要4个block的coll
      // } else {
      //   return 46; // 2, 3号block参加46个coll，即需要4个block的coll。
      // }

      // 4card resnet
      if (blk == 0) {
        return 161;
      } else {
        return 53; // 共有53个coll需要2个block，大于36.75K之后。
      }

      // 8card vit
      // if (blk == 0) {
      //   return 85;
      // } else {
      //   return 84;
      // }
    }
  #endif
#else
  #define MAX_LENGTH 2000LL // 受到0xc000 shmem的限制
  #define NUM_SHMEM_SLOT 1
#endif


// 队列长度搞大些，反正目前也不缺这点显存。就搞得和max collCount一样大，那就不会full了。
#define QLen MAX_LENGTH

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
  unsigned long long int *buffer;
  int readSlot;
  unsigned int blockCollCnt[RESERVED_GRID_DIM][MAX_LENGTH]; // 静态分布也就1M，所以就这样吧。
  // pthread_mutex_t mutex;
} CQ;

struct DevComm7WorkElem {
  ncclComm *oriComm;
  struct ncclDevComm* comm;
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
    unsigned long long int totalSwitchCntAfterRecvSuccess;
    unsigned long long int totalSwitchCntBeforeRecvSuccess;
    unsigned long long int totalUnprogressedQuitCnt;
  #endif
  unsigned int cqCnt[MAX_LENGTH];
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
  char collTryCnt[MAX_LENGTH];
} CollTryCntAlign;


typedef struct alignas(16) {
  short collIds[MAX_LENGTH];
} IdsAlign;

typedef struct alignas(16) {
  char blkCount4Coll[MAX_LENGTH];
} BlkCount4CollAlign;

typedef struct alignas(16) {
  /* ****** 根据hasQuitted的值，决定重置还是从globalMem里读 ****** */
  DynamicBlkStatus dynamicBlkStatus;


  /* ****** 根据宏单独处理 ****** */ 
  #ifdef ARRAY_DEBUG
    unsigned long long int *barrierCnt;
    unsigned long long int *collCounters;
  #endif


  #ifdef DEBUG_CLOCK

    #ifdef DEBUG_CLOCK_TRAIN
      int beforeGetSqeIter[MAX_LENGTH];
      long long int beforeGetSqeClock[MAX_LENGTH][RECORD_ITER];

      int getSqeIter[MAX_LENGTH];
      long long int getSqeClock[MAX_LENGTH][RECORD_ITER];

      // int beforePutCqeIter[MAX_LENGTH];
      long long int beforePutCqeClock[MAX_LENGTH][RECORD_ITER];

      // int putCqeIter[MAX_LENGTH];
      long long int putCqeClock[MAX_LENGTH][RECORD_ITER];

      long long int beforeAfterGetSqeDeltaClock[MAX_LENGTH][RECORD_ITER];
      // long long int afterGetSqeAfterPutCqeDeltaClock[MAX_LENGTH][RECORD_ITER];
      long long int afterGetSqeBeforePutCqeDeltaClock[MAX_LENGTH][RECORD_ITER];
      long long int beforeAfterPutCqeDeltaClock[MAX_LENGTH][RECORD_ITER];
      long long int beforeGetSqeAfterPutCqeDeltaClock[MAX_LENGTH][RECORD_ITER];

      int ctxSwitchCnt[MAX_LENGTH];
    #endif

    #ifdef DEBUG_CLOCK_3D
      int switchCntAfterRecvSuccess[MAX_LENGTH];
      int switchCntBeforeRecvSuccess[MAX_LENGTH];
      int switchCntAfterRecvSuccessIterDelta[MAX_LENGTH];
      int switchCntBeforeRecvSuccessIterDelta[MAX_LENGTH];
      int iterCqeCnt;
      int iterNum;
      int iterSqeCnt;
      int iterSqNum;
      int collIdInSqe[RESNET_COLL_CNT];
      int taskQLenAfterGetSqe[RESNET_COLL_CNT];
      int collId4Cq[RESNET_COLL_CNT];

      int totalSqeCnt;
      int totalCqeCnt;
    #endif

    #ifdef DEBUG_CLOCK_IO
      int beforeGetSqeIter;
      long long int beforeGetSqeClock[RECORD_ITER];

      long long int afterReadSqEmptyDeltaClock[RECORD_ITER];
      long long int afterGetSqFrontierDeltaClock[RECORD_ITER];
      long long int afterAddSqFrontierCounterDeltaClock[RECORD_ITER];
      long long int afterUpdateSqHeadDeltaClock[RECORD_ITER];
      long long int afterRecordBuffDeltaClock[RECORD_ITER];

      int getSqeIter;
      long long int getSqeClock[RECORD_ITER];

      long long int beforeOfcclFuncClock[RECORD_ITER];
      long long int afterGetSqeBeforeOfcclFuncDeltaClock[RECORD_ITER];
      long long int afterGetSqeBeforeMaintainSharedCtxDeltaClock[RECORD_ITER];

      long long int beforePutCqeClock[RECORD_ITER];

      long long int putCqeClock[RECORD_ITER];

      long long int beforeAfterGetSqeDeltaClock[RECORD_ITER];
      long long int afterGetSqeBeforePutCqeDeltaClock[RECORD_ITER];
      long long int beforeAfterPutCqeDeltaClock[RECORD_ITER];
      long long int beforeGetSqeAfterPutCqeDeltaClock[RECORD_ITER];

      int sqReadCnt;
      int cqWriteCnt;
    #endif

    #ifdef DEBUG_CLOCK_CTX
      int loadIter;
      int saveIter;
      long long int beforeLoadClock[RECORD_ITER];
      long long int afterLoadDeltaClock[RECORD_ITER];
      long long int beforeSaveClock[RECORD_ITER];
      long long int afterSaveDeltaClock[RECORD_ITER];
    #endif

  #endif


  /* ****** 数组有单独复制 ****** */
  // 这样的好处是以16B复制的时候，没有越界风险
  CollStatusAlign collStatusAlign;
  ActiveCollIdsAlign activeCollIdsAlign;
  CollTryCntAlign collTryCntAllign;


  /* ****** 固定从globalMem里读 ****** */
  int hasQuitted; // 记录曾经Quit过的状态，一旦被设置，就不再清零。


  /* ****** daemonKernel每次启动需要重置 ****** */
  int willingnessToGetSqe;
  int currLoadedCollId;
  char quit;
  char finallyQuit;
} BlkStatus;

typedef struct {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
} CollCtxGroup;

typedef struct alignas(16) {
  struct ncclWorkElem workElem; // sizeof(struct ncclWorkElem)=64
  // 来自channel
  struct ncclPeer* devPeers;
  // 来自channel.ring
  int *ringRanks;
  int ringPrev;
  int ringNext;
  int ringIndex;
  // 来自comm(devComm, 不是普通comm)
  int rank; // 原来来自于comm.rank，还是放在collCtx而不是blkStatus里，因为在不同的集合通信中，一个设备的rank可能会变，不应该静态保存。
  int nRanks;
  volatile uint32_t *abortFlag;
  int collId;
} StaticCollCtx; // sizeof(StaticCollCtx)=

typedef struct alignas(16) {
  // for p2p sendAcceptor
  void *sendConnPtrExchage;
  // Prims Simple的上下文
  int loadAgain; // 是不是曾经执行了一半，被换出去了，这次是又一次执行。主要用来控制ofccl/src/collectives_ofccl/device/ofccl_prims_simple.h里loadConn时候的roundUp行为，防止异常更新自己的step(head/tail)。正式一点可以搞个issue记录问题，然后在commit里说fix issue。懒得搞了。这个变量是只要曾经被换出去过，就一直是1了，这样每次创建prim，loadConn的时候，才可以都跳过roundUp。
  int slice4SimpleGenericOp;
  int offset4SimpleGenericOp;
  // Ring AllReduce的上下文
  int currentStep4RunRing;
  ssize_t gridOffset4RunRing;

} DynamicCollCtx; // sizeof(DynamicCollCtx)=32

// sizeof(CollCtx)=42104, sizeof(CollCtxGroup)=248, sizeof(ncclDevComm)=40, sizeof(ncclChannel)=512, sizeof(ncclWork)=512, sizeof(struct ncclWorkElem)=64, sizeof(struct ncclWorkElemHeader)=4
// 准备抛弃旧的collCtx结构，只保留我们需要的。
typedef struct alignas(16) {
  // TODO: 对LL、LL128的支持


  /* ****** 每次执行前在maintainSharedCollCtx重置 ****** */
  int64_t ctxSwitchThreshold;
  int recvSuccess; // 这个关注在一次ofcclFunc的执行过程中，是否成功recv了peer的数据，用来作为调整黏性的依据。
  int saveCtx7Quit;


  /* ****** 每次load需要重置、加载 ****** */
  // ---- load的时候用常数重置 ----
  int progressed; // 这个主要关注在sharedCollCtx里加载好的collCtx的值有没有变化，即便只是send，那runRing里的step还是改了的。
  // ---- 只需要load，不需要save ----
  StaticCollCtx staticCollCtx;
  // ---- load、save的时候都需要和globalMem发生关系；完成的时候需要用0重置 ----
  DynamicCollCtx dynamicCollCtx;


  /* ****** 不需要load和save ****** */ 
  #if defined(CQE_DEBUG_RANK_X) || defined(CQE_DEBUG_ALL_RANK)
    unsigned long long sqeReadCnt;
    unsigned long long cqeWriteCnt;
    unsigned long long cqePrepareCnt;
  #endif
  // 这两个是启动相应的coll的执行之后，Primitive构造函数里填充的
  CollCtxGroup groups[NCCL_MAX_GROUPS];
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  int buffSizes[NCCL_NUM_PROTOCOLS]; // 来自comm(devComm, 不是普通comm)

} CollCtx;

extern __global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, char *globalBlkCount4Coll, int *globalThrdCount4Coll, short *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int *finallyQuit, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters, const int64_t TOLERANT_UNPROGRESSED_CNT, const int64_t BASE_CTX_SWITCH_THRESHOLD);
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