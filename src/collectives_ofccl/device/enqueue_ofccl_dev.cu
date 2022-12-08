#include "enqueue_ofccl_dev.h"

// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.
inline __device__ void copy16B(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset));
    asm volatile("st.v2.u64 [%0],{%1,%2};" :: "l"((char*)dst + offset), "l"(a), "l"(b));
  }
}

inline __device__ void set16B(int tid, void* dst, void const* src, int bytes) {
  int offset = 16*tid;
  if (offset < bytes) {
    uint64_t a=0, b=0;
    asm("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src));
    asm volatile("st.v2.u64 [%0],{%1,%2};" :: "l"((char*)dst + offset), "l"(a), "l"(b));
  }
}

// share mem用超了。
// TODO: 可以不同的algo、proto使用不同的数据类型，不过可以看看是不是有意义
__shared__ CollCtx sharedCollCtx; // 不能static，primitives要用

__shared__ BlkStatus blkStatus; // 取消static，放到prim里边打印log。

// static __shared__ IdsAlign sharedIdsAlign;
static __shared__ BlkCount4CollAlign sharedBlkCount4CollAlign;
static __shared__ unsigned long long int zeros[2];

__global__ void sqWriteKernel(SQ *sq, SQE *sqe, int thrdCudaDev, int DEV_TRY_ROUND, int *sqWriteRetFlag) {
  if (threadIdx.x == 0) {
    int tryCnt = 0;
    while (tryCnt++ < DEV_TRY_ROUND) {
      if (DevSqFull(sq)) {
        atomicExch(sqWriteRetFlag, -1);
        continue;
      }
      *DevGetQTail<SQ, SQE>(sq) = *sqe;
      __threadfence();
      atomicAdd(&sq->tail, 1);
      atomicExch(sqWriteRetFlag, 1);
      break;
    }
  }
}

__global__ void cqReadKernel(CQ *cq, CQE *target, int thrdCudaDev, int DEV_TRY_ROUND, int *cqReadRetFlag) {
  if (threadIdx.x == 0) {
    int tryCnt = 0;
    while (tryCnt++ < DEV_TRY_ROUND) {
      if (DevCqEmpty(cq)) {
        atomicExch(cqReadRetFlag, -1);
        continue;
      }
      *target = *DevGetQHead<CQ, CQE>(cq);
      __threadfence();
      atomicAdd(&cq->head, 1);
      atomicExch(cqReadRetFlag, 1);
      break;
    }
  }
}

static __device__ int sqRead(SQ *sq, SQE *target, int thrdCudaDev) {

  unsigned long long int currSqFrontier = blkStatus.dynamicBlkStatus.sqReadFrontier;

  // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, enter, sqReadFrontier = %llu, sq->head=%llu, sq->tail=%llu", thrdCudaDev, blockIdx.x, threadIdx.x, DevRingBufferLogicFrontier(sq, currSqFrontier), DevLogicQHead(sq), DevLogicQTail(sq)); // sharedCollCtx.rank是在loadCtx之后才有效的，在此之前想打印sqRead的情况，需要使用thrdCudaDev，不然会搞出乌龙。

  if (DevSqEmpty(sq, currSqFrontier)) {
    return -1;
  }
  // 先读过来，然后再判断，最后更新状态：sqe->counter; 以及在恰当的时候commit read
  *target = *DevRingBufferGetFrontier(sq, currSqFrontier);
  if (target->quit) {
    // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Get quit", thrdCudaDev, bid, threadIdx.x);
    return 0;
  }

  int oldCounter = atomicAdd(&(DevRingBufferGetFrontier(sq, currSqFrontier)->counter), 1); // 将自己读了的sqe的counter加1，代表有人读过了，有一个block不需要再读这个sqe了，后来再有人读这个的时候加完了去判断。

  blkStatus.dynamicBlkStatus.sqReadFrontier++; // 这次读到了，那对于当前这个block来说，下一个可读的位置前进一个。

  // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d>, update counter = %d for coll_id = %d, @ %llu", thrdCudaDev, blockIdx.x, threadIdx.x, oldCounter + 1, DevRingBufferGetFrontier(sq, currSqFrontier)->collId, DevRingBufferLogicFrontier(sq, currSqFrontier));

  __threadfence(); // 保证device上的各个block不要乱序看到。

  unsigned long long int sqHead;
  if (oldCounter + 1 == gridDim.x) {
    do {
      sqHead = atomicCAS(&sq->head, currSqFrontier, currSqFrontier + 1);
    } while (sqHead != currSqFrontier);

    // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d>, update sq->head, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu, sq->head = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier), DevLogicQHead(sq));
  }

  return 0;
}

static __device__ int cqWrite(CQ *cq, CQE *cqe, int thrdCudaDev, unsigned long long int *cqeWriteCnt) {
  if (DevCqFull(cq)) {
    // not an error; caller keeps trying.
    return -1;
  }

  unsigned long long int myCqFrontier = atomicAdd(&(cq->frontier), 1); // 占坑，我就往这里写了，用的是old值，新的cq->tail预期是atomicAdd之后的cq->frontier，也就是myCqFrontier + 1。
  // 两个线程同时调用atomicAdd，是严格保证各自返回的。

  // *(blkStatus.collCounters + 5 + cqe->collId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) = DevRingBufferLogicFrontier(cq, myCqFrontier);
  // *(blkStatus.collCounters + 6 + cqe->collId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) = cq->tail;

  __threadfence();

  DevRingBufferGetFrontier(cq, myCqFrontier)->collId = cqe->collId; // 那这里也应该各自写进去了。

  __threadfence_system();

  // atomicCAS返回地址上的old值，是否修改体现不在返回值上。
  unsigned long long int cqTail;
  do {
    cqTail = atomicCAS(&cq->tail, myCqFrontier, myCqFrontier + 1);
  } while(cqTail != myCqFrontier); // while这里是观察CAS里的条件是否被满足，如果观察到这个条件满足了，那也就可以确定Swap的操作也就完成了。

  // *(blkStatus.collCounters + 1 + cqe->collId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;
  #ifdef CQE_DEBUG_RANK_X
    OFCCL_LOG_RANK_X(OFCCL_CQE, CQE_DEBUG_RANK_X, "Rank<%d> Blk<%d> Thrd<%d>, put %lluth CQE for coll_id = %d @ %llu and update cq->tail", thrdCudaDev, blockIdx.x, threadIdx.x, ++(*cqeWriteCnt), cqe->collId, DevRingBufferLogicFrontier(cq, myCqFrontier));
  #endif
  #ifdef CQE_DEBUG_ALL_RANK
    OFCCL_LOG(OFCCL_CQE, "Rank<%d> Blk<%d> Thrd<%d>, put %lluth CQE for coll_id = %d @ %llu and update cq->tail", thrdCudaDev, blockIdx.x, threadIdx.x, ++(*cqeWriteCnt), cqe->collId, DevRingBufferLogicFrontier(cq, myCqFrontier));
  #endif
  return 0;
}

#ifndef DEBUG_PARA_LD
static __device__ void copyNcclWorkElem (struct ncclWorkElem &dstElem, const struct ncclWorkElem &srcElem) {
  dstElem.header.funcIndex = srcElem.header.funcIndex;
  dstElem.header.type = srcElem.header.type;
  dstElem.header.nWarps = srcElem.header.nWarps;
  dstElem.header.isLast = srcElem.header.isLast;

  dstElem.regUsed = srcElem.regUsed;
  dstElem.direct = srcElem.direct;
  dstElem.sendbuff = srcElem.sendbuff;
  dstElem.recvbuff = srcElem.recvbuff;
  dstElem.count = srcElem.count;
  dstElem.lastChunkSize = srcElem.lastChunkSize;
  dstElem.root = srcElem.root;
  dstElem.nChannels = srcElem.nChannels;
  dstElem.redOpArg = srcElem.redOpArg;
}
#endif

static __device__ int blockInit(int thrdCudaDev, int collCount, char *globalBlkCount4Coll, int *globalThrdCount4Coll, short *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters, int turn) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  

  ONE_THRD_DO
    #ifdef ARRAY_DEBUG
      blkStatus.barrierCnt = barrierCnt;
      blkStatus.collCounters = collCounters;
    #endif

    blkStatus.quit = 0;
    blkStatus.currLoadedCollId = -1;
    sharedCollCtx.buffSizes[NCCL_PROTO_SIMPLE] = (1 << 22); // TODO: 目前只考虑simple

    zeros[0] = zeros[1] = 0llu;
  ONE_THRD_DO_END

  BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;
  int hasQuitted = myGlobalBlkStatus->hasQuitted; // 每个线程都读。

  if (hasQuitted == 0) {
    set16B(tid, &blkStatus.dynamicBlkStatus, &zeros, sizeof(DynamicBlkStatus));
  } else {
    copy16B(tid, &blkStatus.dynamicBlkStatus, &myGlobalBlkStatus->dynamicBlkStatus, sizeof(DynamicBlkStatus));
  }

  int bcTotalBytes = roundUp(collCount * CHAR_ELEM_SIZE, COPY_ELEM_SIZE);
  int bcDoneBytes = 0;
  while (bcDoneBytes < bcTotalBytes) {
    int targetBytes = min(nthreads * COPY_ELEM_SIZE, bcTotalBytes - bcDoneBytes);
    copy16B(tid, (char *)(sharedBlkCount4CollAlign.blkCount4Coll) + bcDoneBytes, (char *)globalBlkCount4Coll + bcDoneBytes, targetBytes);
    bcDoneBytes += targetBytes;
  }

  ofcclBarrier(1); // 为了下边读取blkStatus.dynamicBlkStatus.numActiveColls

  int aciTotalBytes = roundUp(blkStatus.dynamicBlkStatus.numActiveColls * SHORT_ELEM_SIZE, COPY_ELEM_SIZE);
  int aciDoneBytes = 0;
  while (aciDoneBytes < aciTotalBytes) {
    int targetBytes = min(nthreads * COPY_ELEM_SIZE, aciTotalBytes - aciDoneBytes);
    copy16B(tid, (char *)(blkStatus.activeCollIdsAlign.activeCollIds) + aciDoneBytes, (char *)(&myGlobalBlkStatus->activeCollIdsAlign.activeCollIds) + aciDoneBytes, targetBytes);
    aciDoneBytes += targetBytes;
  }
  return turn;
}

#if defined(CQE_DEBUG_RANK_X) || defined(CQE_DEBUG_ALL_RANK)
static __device__ void logTaskQ(int caller, int thrdCudaDev, int rank=-1) {
  if (rank == -1) {
    rank = thrdCudaDev;
  }
  OFCCL_LOG_RANK_X(OFCCL_CQE, rank, "Rank<%d> Blk<%d> Thrd<%d>, caller = %d, numActiveColls=%d, TaskQ: [%d-%d-%d-%d-%d-%d-%d-%d-%d-%d]", thrdCudaDev, blockIdx.x, threadIdx.x, caller, blkStatus.dynamicBlkStatus.numActiveColls, blkStatus.activeCollIdsAlign.activeCollIds[0], blkStatus.activeCollIdsAlign.activeCollIds[1], blkStatus.activeCollIdsAlign.activeCollIds[2], blkStatus.activeCollIdsAlign.activeCollIds[3], blkStatus.activeCollIdsAlign.activeCollIds[4], blkStatus.activeCollIdsAlign.activeCollIds[5], blkStatus.activeCollIdsAlign.activeCollIds[6], blkStatus.activeCollIdsAlign.activeCollIds[7], blkStatus.activeCollIdsAlign.activeCollIds[8], blkStatus.activeCollIdsAlign.activeCollIds[9]);
}
#endif

// 为了初步实现按需启停，增加一个“空read计数，读不到新的，增加计数”
static __device__ void checkSQ7TidyTaskQ(int thrdCudaDev, SQ *sq, CollCtx *globalBlk2CollId2CollCtx, int *finallyQuit, int *unprogressedCnt) {
  int bid = blockIdx.x;

  SQE target;

  // 能读到，假如是正常SQE，把信息在任务列表里记录一下；假如是quit，那也记录一下
  // 读不到新东西那就算了

  if (sqRead(sq, &target, thrdCudaDev) == -1) {
    *unprogressedCnt += 1;
    if (blkStatus.dynamicBlkStatus.numActiveColls > 0) {
      
      // 没读到新的，应该不用处理taskQ了，因为每次遍历一次taskQ，都会处理。 
    }
    return;
  } else {
    if (target.quit) {
      blkStatus.quit = 1; // TODO: 从鲁棒性的角度来说，这里应该有机制保证看到这个quit sqe的时候，taskQ里的所有sqe也应该都处理完，才能退出。（不过目前可以先不管，可以由用户程序间接保证）；一个简单的保证方法是，加一个check。
      // if (bid == 0) {
        *finallyQuit = 1; // TODO: 为了最后每个block都保证打印统计信息，挺不优雅的
      // }
      return;
    }

    // 正常读到了SQE的话，需要往global的globalBlk2CollId2CollCtx表项里边写入，更新blkStatus.numActiveColls
    int newActiveCollId = target.collId;
    int blkLimit = sharedBlkCount4CollAlign.blkCount4Coll[newActiveCollId]; // 需要参与新读到的coll的block才会进行后续操作。

    *unprogressedCnt = 0;
    // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d>, read SQE for coll_id = %d, reset *unprogressedCnt = 0", thrdCudaDev, blockIdx.x, threadIdx.x, newActiveCollId);

    if (bid < blkLimit) {
      CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + newActiveCollId;
      // if (blkStatus.collStatusAlign.collStatus[newActiveCollId] != 0) { // 应该没有重入的风险。重入指一个正在执行的集合通信又被提起请求。
      //   OFCCL_LOG(OFCCL_FATAL, "Rank<%d> Blk<%d> Thrd<%d> globalCollCtx4Blk7Coll->executing should be 0! sq->head = %llu, sq->tail = %llu, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu", thrdCudaDev, bid, threadIdx.x, DevLogicQHead(sq), DevLogicQTail(sq), DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier));
      // }

      blkStatus.collStatusAlign.collStatus[newActiveCollId] = 1;
      
      #ifdef CQE_DEBUG_RANK_X
        OFCCL_LOG_RANK_X(OFCCL_CQE, CQE_DEBUG_RANK_X, "Rank<%d> Blk<%d> Thrd<%d>, read %lluth SQE for coll_id = %d, sq->head = %llu, sq->tail = %llu, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->sqeReadCnt), newActiveCollId, DevLogicQHead(sq), DevLogicQTail(sq), DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier));
      #endif
      #ifdef CQE_DEBUG_ALL_RANK
        OFCCL_LOG(OFCCL_CQE, "Rank<%d> Blk<%d> Thrd<%d>, read %lluth SQE for coll_id = %d, sq->head = %llu, sq->tail = %llu, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->sqeReadCnt), newActiveCollId, DevLogicQHead(sq), DevLogicQTail(sq), DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier));
      #endif
      
      globalCollCtx4Blk7Coll->staticCollCtx.workElem.sendbuff = target.sendbuff;
      globalCollCtx4Blk7Coll->staticCollCtx.workElem.recvbuff = target.recvbuff;

      // maintain the taskQ here.
      // 新加入的集合通信放在末位，最后执行。如果新加入的集合通信存在于当前的blkStatus.activeCollIds里边，也不必强行放到末位。
      int new_numActiveColls = 0;
      bool newActiveCollId_in_taskQ = false;
      // TODO: 考虑循环展开的优化。
      for (int i = 0; i < blkStatus.dynamicBlkStatus.numActiveColls; ++i) {
        int collIdInTaskQ = blkStatus.activeCollIdsAlign.activeCollIds[i];
        if (collIdInTaskQ == newActiveCollId) {
          newActiveCollId_in_taskQ = true;
        }
        if (blkStatus.collStatusAlign.collStatus[collIdInTaskQ] != 0) { // 1_新加入、-2_switch、-1_switch但有progress 都算在执行中，要保留在任务列表中，应该不会有2
          // 在同一个数组上就地操作。new_numActiveColls一定是<=i的，所以不会有问题。
          blkStatus.activeCollIdsAlign.activeCollIds[new_numActiveColls++] = collIdInTaskQ;
        }
      }
      if (!newActiveCollId_in_taskQ) {
        blkStatus.activeCollIdsAlign.activeCollIds[new_numActiveColls++] = newActiveCollId;
      }

      blkStatus.dynamicBlkStatus.numActiveColls = new_numActiveColls;
      #ifdef CQE_DEBUG_ALL_RANK
        logTaskQ(0, thrdCudaDev, -1);
      #elif defined(CQE_DEBUG_RANK_X)
        logTaskQ(0, thrdCudaDev, CQE_DEBUG_RANK_X);
      #endif
    }
  }
}

#ifdef DEBUG_PARA_LD
static __device__ int loadCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId, int turn, int64_t BASE_CTX_SWITCH_THRESHOLD) {
  int tid = threadIdx.x;

  // TODO: 考虑让所有线程都执行常数初始化。
  ONE_THRD_DO
    #ifdef SHOW_CNT
      blkStatus.dynamicBlkStatus.totalCtxLoadCnt++;
    #endif
    blkStatus.currLoadedCollId = collId;

    sharedCollCtx.progressed = 0;
    sharedCollCtx.ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD;
  ONE_THRD_DO_END

  copy16B(tid, &sharedCollCtx.dynamicCollCtx, &globalCollCtx4Blk7Coll->dynamicCollCtx, sizeof(DynamicCollCtx));
  copy16B(tid, &sharedCollCtx.staticCollCtx, &globalCollCtx4Blk7Coll->staticCollCtx, sizeof(StaticCollCtx));

  return turn;
}
#else
static __device__ int loadCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId, int turn, int64_t BASE_CTX_SWITCH_THRESHOLD) {
  int tid = threadIdx.x;

  if (tid == 0) { // 31线程不需要参加下边copy16B的执行，稍稍提高点效率
    #ifdef SHOW_CNT
      blkStatus.dynamicBlkStatus.totalCtxLoadCnt++;
    #endif
    blkStatus.currLoadedCollId = collId;

    sharedCollCtx.progressed = 0;
    sharedCollCtx.ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD;

    sharedCollCtx.staticCollCtx.ringPrev = globalCollCtx4Blk7Coll->staticCollCtx.ringPrev;
    sharedCollCtx.staticCollCtx.ringNext = globalCollCtx4Blk7Coll->staticCollCtx.ringNext;
    sharedCollCtx.staticCollCtx.ringIndex = globalCollCtx4Blk7Coll->staticCollCtx.ringIndex;
    sharedCollCtx.staticCollCtx.devPeers = globalCollCtx4Blk7Coll->staticCollCtx.devPeers;

    sharedCollCtx.staticCollCtx.rank = globalCollCtx4Blk7Coll->staticCollCtx.rank;
    sharedCollCtx.staticCollCtx.nRanks = globalCollCtx4Blk7Coll->staticCollCtx.nRanks;
    sharedCollCtx.staticCollCtx.abortFlag = globalCollCtx4Blk7Coll->staticCollCtx.abortFlag;    
    sharedCollCtx.dynamicCollCtx.loadAgain = globalCollCtx4Blk7Coll->dynamicCollCtx.loadAgain;
    sharedCollCtx.dynamicCollCtx.slice4SimpleGenericOp = globalCollCtx4Blk7Coll->dynamicCollCtx.slice4SimpleGenericOp;
    sharedCollCtx.dynamicCollCtx.offset4SimpleGenericOp = globalCollCtx4Blk7Coll->dynamicCollCtx.offset4SimpleGenericOp;
    sharedCollCtx.dynamicCollCtx.currentStep4RingAllReduce = globalCollCtx4Blk7Coll->dynamicCollCtx.currentStep4RingAllReduce;
    sharedCollCtx.dynamicCollCtx.gridOffset4RingAllReduce = globalCollCtx4Blk7Coll->dynamicCollCtx.gridOffset4RingAllReduce;

    copyNcclWorkElem(sharedCollCtx.staticCollCtx.workElem, globalCollCtx4Blk7Coll->staticCollCtx.workElem);
    // __threadfence_block();
  }
  return turn;
}
#endif

#ifdef DEBUG_PARA_SV
static __device__ void saveExcutingCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId) {
  int tid = threadIdx.x;
  #ifdef SHOW_CNT
    ONE_THRD_DO
      blkStatus.dynamicBlkStatus.totalCtxSaveCnt++;
    ONE_THRD_DO_END
  #endif
  copy16B(tid, &globalCollCtx4Blk7Coll->dynamicCollCtx, &sharedCollCtx.dynamicCollCtx, sizeof(DynamicCollCtx));
}
#else
static __device__ void saveExcutingCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId) {
  if(threadIdx.x == 0) {
    globalCollCtx4Blk7Coll->dynamicCollCtx.loadAgain = sharedCollCtx.dynamicCollCtx.loadAgain;
    globalCollCtx4Blk7Coll->dynamicCollCtx.slice4SimpleGenericOp = sharedCollCtx.dynamicCollCtx.slice4SimpleGenericOp;
    globalCollCtx4Blk7Coll->dynamicCollCtx.offset4SimpleGenericOp = sharedCollCtx.dynamicCollCtx.offset4SimpleGenericOp;
  
    globalCollCtx4Blk7Coll->dynamicCollCtx.currentStep4RingAllReduce = sharedCollCtx.dynamicCollCtx.currentStep4RingAllReduce;
    globalCollCtx4Blk7Coll->dynamicCollCtx.gridOffset4RingAllReduce = sharedCollCtx.dynamicCollCtx.gridOffset4RingAllReduce;
  
    #ifdef SHOW_CNT
      blkStatus.dynamicBlkStatus.totalCtxSaveCnt++;
    #endif
  }
}
#endif

static __device__ void manipulateCQ7ResetDoneColl(int thrdCudaDev, int doneCollId, CQ *cq, CQE *globalCqes, CollCtx *globalCollCtx4Blk7Coll, CollCtx *globalBlk2CollId2CollCtx) {
  // 协调所有blk，发现所有blk都完成，最后一个blk发送CQE
  int old_counter = atomicAdd(&(globalCqes[doneCollId].counter), 1);
  __threadfence(); // cqes在global memory里边，全部block关心。

  // *(blkStatus.collCounters + 0 + doneCollId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;

  // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prepare %lluth CQE for coll_id = %d", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->cqePrepareCnt), doneCollId);

  if (old_counter + 1 == sharedBlkCount4CollAlign.blkCount4Coll[doneCollId]) {
    atomicExch(&globalCqes[doneCollId].counter, 0);

    #if defined(CQE_DEBUG_RANK_X) || defined(CQE_DEBUG_ALL_RANK)
      CollCtx *globalCollCtx4Blk_0_7Coll = globalBlk2CollId2CollCtx + 0 * MAX_LENGTH + doneCollId;
      unsigned long long int *cqeWriteCnt = &globalCollCtx4Blk_0_7Coll->cqeWriteCnt;
      while (cqWrite(cq, globalCqes + doneCollId, thrdCudaDev, cqeWriteCnt) == -1) {
      }
    #else
      while (cqWrite(cq, globalCqes + doneCollId, thrdCudaDev, nullptr) == -1) {
      }
    #endif
    // *(blkStatus.collCounters + 1 + doneCollId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;
    __threadfence();
  }

  // bugfix: manipulateCQ7ResetDoneColl的调用时机，是traverseTaskQ外边，又遍历一次taskQ，所以要判断一下。这也是可以带来性能优化的地方。不判断会导致另一个coll从上次save的地方重新load，重新做已经完成的搬运，但是peer rank未必能配合，就会卡住。
  if (doneCollId == blkStatus.currLoadedCollId) {
    blkStatus.currLoadedCollId = -1;
  }

  blkStatus.collStatusAlign.collStatus[doneCollId] = 0;

  // ResetDoneColl
  globalCollCtx4Blk7Coll->dynamicCollCtx.loadAgain = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.slice4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.offset4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.currentStep4RingAllReduce = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.gridOffset4RingAllReduce = 0;
}

static __device__ int maintainSharedCollCtx(int thrdCudaDev, CollCtx *globalBlk2CollId2CollCtx, int collId, int turn, int64_t BASE_CTX_SWITCH_THRESHOLD, int64_t BOUNS_SWITCH_4_PROCESSED_COLL, int *unprogressedCnt) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, old blkStatus.currLoadedCollId=%d", thrdCudaDev, blockIdx.x, threadIdx.x, collId, blkStatus.currLoadedCollId);
  
  bool noLoadedColl = (blkStatus.currLoadedCollId == -1);
  bool sameLoadedColl = (collId == blkStatus.currLoadedCollId); // 这个条件成立的情况不止一种。

  // bool loadedCollSaveCtx7Quit = !noLoadedColl && (blkStatus.collStatusAlign.collStatus[blkStatus.currLoadedCollId] < 0);
  // bool needSave = !sameLoadedColl && loadedCollSaveCtx7Quit;

  // TODO: 只有progressed，才需要save。
  bool loadedCollProgressed7SaveCtx7Quit = !noLoadedColl && (blkStatus.collStatusAlign.collStatus[blkStatus.currLoadedCollId] == -1);
  bool needSave = !sameLoadedColl && loadedCollProgressed7SaveCtx7Quit;

  bool needLoad = noLoadedColl || !sameLoadedColl;
    
  if (needSave) {
    // bugfix: save的时候，不应该save到即将load的coll的global collCtx副本里。
    CollCtx *globalCollCtx4Blk7OldColl = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + blkStatus.currLoadedCollId;

    saveExcutingCollCtx(thrdCudaDev, globalCollCtx4Blk7OldColl, blkStatus.currLoadedCollId);

    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> save ctx for coll_id = %d, sharedCollCtx.dynamicCollCtx.slice4SimpleGenericOp=%d, sharedCollCtx.dynamicCollCtx.offset4SimpleGenericOp=%d, sharedCollCtx.dynamicCollCtx.currentStep4RingAllReduce=%d, sharedCollCtx.dynamicCollCtx.gridOffset4RingAllReduce=%ld", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.currLoadedCollId, sharedCollCtx.dynamicCollCtx.slice4SimpleGenericOp, sharedCollCtx.dynamicCollCtx.offset4SimpleGenericOp, sharedCollCtx.dynamicCollCtx.currentStep4RingAllReduce, sharedCollCtx.dynamicCollCtx.gridOffset4RingAllReduce);
  }

  if (needLoad) {
    CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collId;
    turn = loadCollCtx(thrdCudaDev, globalCollCtx4Blk7Coll, collId, turn, BASE_CTX_SWITCH_THRESHOLD);

    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> load ctx for coll_id = %d, sharedCollCtx.dynamicCollCtx.loadAgain=%d, sharedCollCtx.dynamicCollCtx.slice4SimpleGenericOp=%d, sharedCollCtx.dynamicCollCtx.offset4SimpleGenericOp=%d, sharedCollCtx.dynamicCollCtx.currentStep4RingAllReduce=%d, sharedCollCtx.dynamicCollCtx.gridOffset4RingAllReduce=%ld", thrdCudaDev, blockIdx.x, threadIdx.x, collId, sharedCollCtx.dynamicCollCtx.loadAgain, sharedCollCtx.dynamicCollCtx.slice4SimpleGenericOp, sharedCollCtx.dynamicCollCtx.offset4SimpleGenericOp, sharedCollCtx.dynamicCollCtx.currentStep4RingAllReduce, sharedCollCtx.dynamicCollCtx.gridOffset4RingAllReduce);
  }

  if (tid == 0) {
    if (blkStatus.collStatusAlign.collStatus[collId] == -1) {
      // bugfix: 防止一个不需要load的coll，无休止增加下去。
      sharedCollCtx.ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD + BOUNS_SWITCH_4_PROCESSED_COLL;
      // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, blkStatus.collStatusAlign.collStatus is %d, sharedCollCtx.ctxSwitchThreshold = %ld", thrdCudaDev, blockIdx.x, threadIdx.x, collId, blkStatus.collStatusAlign.collStatus[collId], sharedCollCtx.ctxSwitchThreshold);
      *unprogressedCnt = 0; // 这表明有coll前进了，只不过没跑完。
      
      #ifdef SHOW_CNT
        blkStatus.dynamicBlkStatus.totalProgressed7SwithchCnt++;
      #endif
    } else if (blkStatus.collStatusAlign.collStatus[collId] == -2) {
      sharedCollCtx.ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD;
      *unprogressedCnt += 1;
    }

    blkStatus.collStatusAlign.collStatus[collId] = 1; // 每次准备执行的时候，重置为正常执行状态。新的coll已经是1，不过不要浪费if了。 
    sharedCollCtx.saveCtx7Quit = 0; // 重置。
  }

  ofcclBarrier(4);
  return turn;
}

static __device__ int traverseTaskQ(int thrdCudaDev, CollCtx *globalBlk2CollId2CollCtx, int collCount, CQ *cq, CQE *globalCqes, int turn, int *unprogressedCnt, int64_t BASE_CTX_SWITCH_THRESHOLD, int64_t BOUNS_SWITCH_4_PROCESSED_COLL) {
  int bid = blockIdx.x;

  #if defined(ARRAY_DEBUG)
    *(blkStatus.barrierCnt + 0 + 11 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    if (blkStatus.dynamicBlkStatus.numActiveColls == 0) {
      *(blkStatus.barrierCnt + 1 + 11 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      return turn;
    }
  #else 
    if (blkStatus.dynamicBlkStatus.numActiveColls == 0) {
      return turn;
    }
  #endif

  // TODO: 循环展开的优化？
  int i = 0;
  for (; i < blkStatus.dynamicBlkStatus.numActiveColls; i++) {

    // 下边这三个量是不变的。
    int collId = blkStatus.activeCollIdsAlign.activeCollIds[i];
    int blkLimit = sharedBlkCount4CollAlign.blkCount4Coll[collId];

    // *(blkStatus.barrierCnt + 0 + 10 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    // *(blkStatus.barrierCnt + 2 + 10 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = collId;

    // 这里不需要再判断blkStatus.collStatus[collId]了，因为这一次循环里只会遍历taskQ一次，出去之后就更新taskQ了。
    if (bid < blkLimit) { // blk天然分化，保留这个条件 // TODO: 如果节省if判断对性能有提升，可以改变处理方法，让所有block处理所有的集合通信。不过好像也省不了。。。总得判断。

      // ***** 先准备好sharedCollCtx，全部线程都参与 *****
      turn = maintainSharedCollCtx(thrdCudaDev, globalBlk2CollId2CollCtx, collId, turn, BASE_CTX_SWITCH_THRESHOLD, BOUNS_SWITCH_4_PROCESSED_COLL, unprogressedCnt);

      // *(blkStatus.barrierCnt + 0 + 15 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

      // ***** 然后调用ofcclFunc *****
      int wid = threadIdx.x / WARP_SIZE;
      if (wid < sharedCollCtx.staticCollCtx.workElem.header.nWarps) {
        ofcclFuncs[sharedCollCtx.staticCollCtx.workElem.header.funcIndex](); // 这里边的调用里不涉及__syncthreads().
      }

      // *(blkStatus.barrierCnt + 1 + 15 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

      // *(blkStatus.barrierCnt + 0 + 13 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      ofcclBarrier(3); // 跑完一个集合通信，同步一下。
      // *(blkStatus.barrierCnt + 1 + 13 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    }

    // *(blkStatus.barrierCnt + 1 + 10 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  }
  #if defined(ARRAY_DEBUG)
    *(blkStatus.barrierCnt + 2 + 11 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  #endif

  return turn;
}

__global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, char *globalBlkCount4Coll, int *globalThrdCount4Coll, short *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int *finallyQuit, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters, const int64_t TRAVERSE_TIMES, const int64_t TOLERANT_UNPROGRESSED_CNT, const int64_t BASE_CTX_SWITCH_THRESHOLD, const int64_t BOUNS_SWITCH_4_PROCESSED_COLL) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel starts, blkStatus.dynamicBlkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, blkStatus.dynamicBlkStatus.numActiveColls);
  // OFCCL_LOG_THRD_0(OFCCL_CQE, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel starts", thrdCudaDev, blockIdx.x, tid);
  // __syncwarp(); // ！！！！！！为了打印log加的！！！！

  // int tempRound = 0;
  int turn = 0;

  turn = blockInit(thrdCudaDev, collCount, globalBlkCount4Coll, globalThrdCount4Coll, globalCollIds, globalDevComm7WorkElems, globalBlk2CollId2CollCtx, globalBlkStatus, barrierCnt, collCounters, turn);

  #ifdef ARRAY_DEBUG
    if (tid == 0) {
      *(blkStatus.barrierCnt + 0 + 5 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    }
  #endif

  ofcclBarrier(5);

  int unprogressedCnt = 0;
  while (true) {

    for (int i = 0; i < TRAVERSE_TIMES; i++) {
      if (blkStatus.dynamicBlkStatus.numActiveColls == 0) {
        break;
      }
      turn = traverseTaskQ(thrdCudaDev, globalBlk2CollId2CollCtx, collCount, cq, globalCqes, turn, &unprogressedCnt, BASE_CTX_SWITCH_THRESHOLD, BOUNS_SWITCH_4_PROCESSED_COLL);

      if (tid == 0) { // 遍历完一次之后，当前activeColl的后续工作，
        // 只有完成一个集合通信，才有必要操作taskQ
        int new_numActiveColls = 0;
        for (int i = 0; i < blkStatus.dynamicBlkStatus.numActiveColls; ++i) {
          int collIdInTaskQ = blkStatus.activeCollIdsAlign.activeCollIds[i];
          // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, blkStatus.collStatusAlign.collStatus is %d", thrdCudaDev, blockIdx.x, threadIdx.x, collIdInTaskQ, blkStatus.collStatusAlign.collStatus[collIdInTaskQ]);
          if (blkStatus.collStatusAlign.collStatus[collIdInTaskQ] < 0) { // 不应该有1 的存在了，只有-1, -2或者2
            blkStatus.activeCollIdsAlign.activeCollIds[new_numActiveColls++] = collIdInTaskQ; // 小于0，就要继续放在taskQ里

          } else if (blkStatus.collStatusAlign.collStatus[collIdInTaskQ] == 2) {
            unprogressedCnt = 0;
            // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, blkStatus.collStatusAlign.collStatus is %d, done, reset nprogressedCnt = 0", thrdCudaDev, blockIdx.x, threadIdx.x, collIdInTaskQ, blkStatus.collStatusAlign.collStatus[collIdInTaskQ]);

            CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collIdInTaskQ;
            manipulateCQ7ResetDoneColl(thrdCudaDev, collIdInTaskQ, cq, globalCqes, globalCollCtx4Blk7Coll, globalBlk2CollId2CollCtx);
            // 对于完成执行的集合通信应该不用把shmem里的collCtx写回到global mem里边，sendbuff/recvbuff等下次的SQE传过来，剩下的其他都是些静态配置项。
          }
        }
        blkStatus.dynamicBlkStatus.numActiveColls = new_numActiveColls;
        #ifdef CQE_DEBUG_ALL_RANK
          logTaskQ(1, thrdCudaDev, -1);
        #elif defined(CQE_DEBUG_RANK_X)
          logTaskQ(1, thrdCudaDev, CQE_DEBUG_RANK_X);
        #endif
      }
      // *(blkStatus.barrierCnt + 0 + 18 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      ofcclBarrier(10);
      // *(blkStatus.barrierCnt + 1 + 18 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    }

    if (tid == 0) {
    
      checkSQ7TidyTaskQ(thrdCudaDev, sq, globalBlk2CollId2CollCtx, finallyQuit, &unprogressedCnt);

      // 只有0号线程才会执行checkSQ7TidyTaskQ，自然只有0号线程才会更改checkSQ7TidyTaskQFailCnt，并且进行相应调整。

      if (unprogressedCnt >= TOLERANT_UNPROGRESSED_CNT && blkStatus.quit != 1) {
        BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;

        // bugfix: 如果需要，应该save coll的上下文，但是按理说，如果有个coll是-1，不可能主动退出的。留下来吧。
        bool noLoadedColl = (blkStatus.currLoadedCollId == -1);
        // bool needSave = (!noLoadedColl) && (blkStatus.collStatusAlign.collStatus[blkStatus.currLoadedCollId] < 0);
        bool needSave = (!noLoadedColl) && (blkStatus.collStatusAlign.collStatus[blkStatus.currLoadedCollId] == -1);
        if (needSave) {
          CollCtx *globalCollCtx4Blk7OldColl = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + blkStatus.currLoadedCollId;
          saveExcutingCollCtx(thrdCudaDev, globalCollCtx4Blk7OldColl, blkStatus.currLoadedCollId);
        }

        // 保存blkstatus
        myGlobalBlkStatus->hasQuitted = 1;
        blkStatus.quit = 1;

        #ifdef SHOW_CNT
          ++blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt;
        #endif
      }
    }

    // *(blkStatus.barrierCnt + 0 + 9 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    ofcclBarrier(7); // prims_simple里用的是8和15。
    // *(blkStatus.barrierCnt + 1 + 9 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

    // // daemonKernel一开始这个数组用不上，可以用来记点其他信息

    #ifdef ARRAY_DEBUG
      if (tid == 0) {
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 33 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.totalCtxSaveCnt;
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 34 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.totalCtxLoadCnt;
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 35 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.totalProgressed7SwithchCnt;
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 36 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.numActiveColls;
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 37 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = unprogressedCnt;
      }
    #endif

    // 记录数组的前10项，未必都是有效的。所有线程都做，看到的应该是一样的。
    // for (int i = 0; i < PrintTestQNum; i++) {
    //   *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + (38 + i) * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.activeCollIdsAlign.activeCollIds[i];
    // }

    if (blkStatus.quit == 1) {

      if (*finallyQuit == 1) {
        #ifdef SHOW_CNT
          OFCCL_LOG_THRD_0(OFCCL_FINAL_QUIT, "Rank<%d> Blk<%d> Thrd<%d> totalCtxSaveCnt=%llu, totalCtxLoadCnt=%llu, totalProgressed7SwithchCnt=%llu, totalUnprogressedQuitCnt=%llu", thrdCudaDev, bid, tid, blkStatus.dynamicBlkStatus.totalCtxSaveCnt, blkStatus.dynamicBlkStatus.totalCtxLoadCnt, blkStatus.dynamicBlkStatus.totalProgressed7SwithchCnt, blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt);
        #endif
      } else {
        int aciTotalBytes = roundUp(blkStatus.dynamicBlkStatus.numActiveColls * SHORT_ELEM_SIZE, COPY_ELEM_SIZE);
        int aciDoneBytes = 0;
        BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;
        int nthreads = blockDim.x;
        while (aciDoneBytes < aciTotalBytes) {
          int targetBytes = min(nthreads * COPY_ELEM_SIZE, aciTotalBytes - aciDoneBytes);
          copy16B(tid, (char *)(&myGlobalBlkStatus->activeCollIdsAlign.activeCollIds) + aciDoneBytes, (char *)(blkStatus.activeCollIdsAlign.activeCollIds) + aciDoneBytes, targetBytes);
          aciDoneBytes += targetBytes;
        }
        copy16B(tid, &myGlobalBlkStatus->dynamicBlkStatus, &blkStatus.dynamicBlkStatus, sizeof(DynamicBlkStatus));
      }


      // OFCCL_LOG_THRD_0(OFCCL_CQE, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel quits", thrdCudaDev, blockIdx.x, tid);
      #ifdef ARRAY_DEBUG
        if (tid == 0) {
          *(blkStatus.barrierCnt + 1 + 5 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #ifdef SHOW_CNT
            *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 66 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt;
          #endif
        }
      #endif
      return;
    }
  }
}

__global__ void sqCreateKernel(SQ *sq) {
  qCreateDev(sq);
}

__global__ void cqCreateKernel(CQ *cq) {
  qCreateDev(cq);
}