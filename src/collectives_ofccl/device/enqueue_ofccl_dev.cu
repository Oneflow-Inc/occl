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

inline __device__ void copy16BLoop(int tid, void* dst, void const* src, int totolBytes) {
  int offset = 16*tid;
  while (offset < totolBytes) {
    uint64_t a=0, b=0;
    asm("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src + offset));
    asm volatile("st.v2.u64 [%0],{%1,%2};" :: "l"((char*)dst + offset), "l"(a), "l"(b));
    offset += 16*blockDim.x;
  }
}

inline __device__ void set16BLoop(int tid, void* dst, void const* src, int totolBytes) {
  int offset = 16*tid;
  while (offset < totolBytes) {
    uint64_t a=0, b=0;
    asm("ld.v2.u64 {%0,%1},[%2];" : "=l"(a),"=l"(b) : "l"((char const*)src));
    asm volatile("st.v2.u64 [%0],{%1,%2};" :: "l"((char*)dst + offset), "l"(a), "l"(b));
    offset += 16*blockDim.x;
  }
}

// share mem用超了。
// TODO: 可以不同的algo、proto使用不同的数据类型，不过可以看看是不是有意义
__shared__ CollCtx sharedCollCtx[NUM_SHMEM_SLOT]; // 不能static，primitives要用

__shared__ BlkStatus blkStatus; // 取消static，放到prim里边打印log。

// static __shared__ IdsAlign sharedIdsAlign;
static __shared__ BlkCount4CollAlign sharedBlkCount4CollAlign;
static __shared__ unsigned long long int zeros[2];
static __shared__ int cqWriteSlot;

#ifdef DEBUG_CLOCK_3D
__constant__ int *taskQLen4RankBlkIterColl;
__constant__ int *unprogressed7SwitchCnt4RankBlkIterColl;
__constant__ int *progressed7SwitchCnt4RankBlkIterColl;
__constant__ int *collIdInSqe4RankBlkIterColl;
__constant__ int *collId4Cq4RankBlkIterColl;
__constant__ int *unprogressed7SwitchCntTotal4RankBlkIterColl;
__constant__ int *progressed7SwitchCntTotal4RankBlkIterColl;
__constant__ int numColl;
#endif

__constant__ int64_t NUM_TRY_TASKQ_HEAD;
__constant__ int64_t RECV_SUCCESS_FACTOR;
__constant__ int64_t RECV_SUCCESS_THRESHOLD;

#ifdef SHOW_CNT
__constant__ int64_t NUM_ITER_ENV;
#endif

inline __device__ int getTryNum(int posInTaskQ) {
  if (posInTaskQ == 0) {
    return int(NUM_TRY_TASKQ_HEAD);
  }
  return 1;
}

inline __device__ void getInitSwitchThreshold(int collId, int64_t BASE_CTX_SWITCH_THRESHOLD) {
  // 这个行为到底有没有意义？或许可以讨论下。
  int tryCnt =  min(int(blkStatus.collTryCntAllign.collTryCnt[collId]), int(NUM_TRY_TASKQ_HEAD));
  sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD * tryCnt;
}

static __device__ int sqRead(SQ *sq, SQE *target, int thrdCudaDev) {

  unsigned long long int currSqFrontier = blkStatus.dynamicBlkStatus.sqReadFrontier;

  // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, enter, sqReadFrontier = %llu, sq->head=%llu, sq->tail=%llu", thrdCudaDev, blockIdx.x, threadIdx.x, DevRingBufferLogicFrontier(sq, currSqFrontier), DevLogicSqHead(sq), DevLogicSqTail(sq)); // sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].rank是在loadCtx之后才有效的，在此之前想打印sqRead的情况，需要使用thrdCudaDev，不然会搞出乌龙。
  #ifdef DEBUG_CLOCK
    #ifdef DEBUG_CLOCK_IO
      ++blkStatus.sqReadCnt;
      long long int localBeforeGetSqeClock = clock64();
      long long int afterReadSqEmptyClock;
      long long int afterGetSqFrontierClock;
      long long int afterAddSqFrontierCounterClock;
      long long int afterUpdateSqHeadClock;
    #endif
  #endif

  if (DevSqEmpty(sq, currSqFrontier)) {
    return -1;
  }
  #ifdef DEBUG_CLOCK
    #ifdef DEBUG_CLOCK_IO

      if (blkStatus.beforeGetSqeIter >= SKIP_WARMUP_ITER) {
        int iter = (blkStatus.beforeGetSqeIter - SKIP_WARMUP_ITER) % RECORD_ITER;
        blkStatus.beforeGetSqeClock[iter] = localBeforeGetSqeClock; // 只有真的读到sqe才记录这个值。
      }
      ++blkStatus.beforeGetSqeIter;

      if (blkStatus.beforeGetSqeIter >= SKIP_WARMUP_ITER + 1) {
        int iter = (blkStatus.beforeGetSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
        afterReadSqEmptyClock = clock64();
        blkStatus.afterReadSqEmptyDeltaClock[iter] = calcDeltaClock(blkStatus.beforeGetSqeClock[iter], afterReadSqEmptyClock);
      }
    #endif
  #endif
  // 先读过来，然后再判断，最后更新状态：sqe->counter; 以及在恰当的时候commit read
  *target = *DevRingBufferGetFrontier(sq, currSqFrontier);
  if (target->quit) {
    // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Get quit", thrdCudaDev, bid, threadIdx.x);
    return 0;
  }
  #ifdef DEBUG_CLOCK
    #ifdef DEBUG_CLOCK_IO
      if (blkStatus.beforeGetSqeIter >= SKIP_WARMUP_ITER + 1) {
        int iter = (blkStatus.beforeGetSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
        afterGetSqFrontierClock = clock64();
        blkStatus.afterGetSqFrontierDeltaClock[iter] = calcDeltaClock(afterReadSqEmptyClock, afterGetSqFrontierClock);
      }
    #endif
  #endif

  int oldCounter = atomicAdd(&(DevRingBufferGetFrontier(sq, currSqFrontier)->counter), 1); // 将自己读了的sqe的counter加1，代表有人读过了，有一个block不需要再读这个sqe了，后来再有人读这个的时候加完了去判断。

  blkStatus.dynamicBlkStatus.sqReadFrontier++; // 这次读到了，那对于当前这个block来说，下一个可读的位置前进一个。
  #ifdef DEBUG_CLOCK
    #ifdef DEBUG_CLOCK_IO
      if (blkStatus.beforeGetSqeIter >= SKIP_WARMUP_ITER + 1) {
        int iter = (blkStatus.beforeGetSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
        afterAddSqFrontierCounterClock = clock64();
        blkStatus.afterAddSqFrontierCounterDeltaClock[iter] = calcDeltaClock(afterGetSqFrontierClock, afterAddSqFrontierCounterClock);
      }
    #endif
  #endif

  // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d>, update counter = %d for coll_id = %d, @ %llu", thrdCudaDev, blockIdx.x, threadIdx.x, oldCounter + 1, DevRingBufferGetFrontier(sq, currSqFrontier)->collId, DevRingBufferLogicFrontier(sq, currSqFrontier));

  // 会被device上其他block看到的都是原子操作了
  // __threadfence(); // 保证device上的各个block不要乱序看到。

  unsigned long long int sqHead;
  if (oldCounter + 1 == gridDim.x) {
    do {
      sqHead = atomicCAS(&sq->head, currSqFrontier, currSqFrontier + 1);
    } while (sqHead != currSqFrontier);

    // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d>, update sq->head, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu, sq->head = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier), DevLogicSqHead(sq));
    #ifdef DEBUG_CLOCK
      #ifdef DEBUG_CLOCK_IO
        if (blkStatus.beforeGetSqeIter >= SKIP_WARMUP_ITER + 1) {
          int iter = (blkStatus.beforeGetSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
          afterUpdateSqHeadClock = clock64();
          blkStatus.afterUpdateSqHeadDeltaClock[iter] = calcDeltaClock(afterAddSqFrontierCounterClock, afterUpdateSqHeadClock);
        }
      #endif
    #endif
  }

  return 0;
}

static __device__ int cqWrite(CQ *cq, int doneCollId, int thrdCudaDev, unsigned long long int *cqeWriteCnt) {
  #ifdef DEBUG_CLOCK
    #ifdef DEBUG_CLOCK_IO
      ++blkStatus.cqWriteCnt;
    #endif
  #endif

  cqWriteSlot %= NUM_CQ_SLOT; // 原来把取模放在for循环初始那里，事实上会在写失败的时候，一直反复循环，而不是返回。其实是不好的。
  unsigned long long int cqSlot = 0llu;
  cqSlot |= doneCollId;
  cqSlot |= BLOCK_CNT_MASK & ((unsigned long long int)(blkStatus.dynamicBlkStatus.cqCnt[doneCollId]) << COLL_ID_BIT);
  cqSlot |= (unsigned long long int)blockIdx.x << (BLOCK_CNT_BIT + COLL_ID_BIT);
  for (; cqWriteSlot < NUM_CQ_SLOT; ++cqWriteSlot) {
    
    unsigned long long int oldSlot = atomicCAS_system(cq->buffer + cqWriteSlot, INVALID_CQ_SLOT_MASK, cqSlot);
    // OFCCL_LOG(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after CAS oldSlot = 0x%llx, cqCnt = %u, doneCollId = %d", thrdCudaDev, blockIdx.x, threadIdx.x, oldSlot, blkStatus.dynamicBlkStatus.cqCnt[doneCollId], doneCollId);

    if (oldSlot == INVALID_CQ_SLOT_MASK) {
      ++blkStatus.dynamicBlkStatus.cqCnt[doneCollId]; // 写成功才更新。
      return 0;
    }
  }

  // 纯粹的单坑：

  return -1;
}

static __device__ void blockInit(int thrdCudaDev, int collCount, char *globalBlkCount4Coll, int *globalThrdCount4Coll, short *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  // int nthreads = blockDim.x;
  
  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 1", thrdCudaDev, blockIdx.x, threadIdx.x);
  ONE_THRD_DO
    #ifdef ARRAY_DEBUG
      blkStatus.barrierCnt = barrierCnt;
      blkStatus.collCounters = collCounters;
    #endif

    blkStatus.quit = 0;
    blkStatus.finallyQuit = 0;
    blkStatus.currLoadedCollId = -1;
    for (int i = 0; i < NUM_SHMEM_SLOT; ++i) {
      // TODO: 可以并行优化，每次循环的增量是blockDim.x
      sharedCollCtx[i].buffSizes[NCCL_PROTO_SIMPLE] = (1 << 22); // TODO: 目前只考虑simple
      sharedCollCtx[i].staticCollCtx.collId = -1; // 作用类似于设置blkStatus.currLoadedCollId = -1。可以使得不读保存collStatus的值，通过这里的重置以及下边恢复activeCollIds，保证在maintainSharedCollCtx里有正确的行为
    }

    zeros[0] = zeros[1] = 0llu;
    cqWriteSlot = 0;
  ONE_THRD_DO_END
  
  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 2", thrdCudaDev, blockIdx.x, threadIdx.x);
  // 不需要初始化DEBUG_CLOCK里的数组，因为这些数组使用的时候都是直接赋值的。

  BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;
  int hasQuitted = myGlobalBlkStatus->hasQuitted; // 每个线程都读。

  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 3", thrdCudaDev, blockIdx.x, threadIdx.x);

  // 第一次启动之前，rankCtx->hostBlkStatus是calloc的，然后复制到globalMem上，所以blkStatus.collStatusAlign.collStatus应该是全0，但是之后的启动可能导致collStatus数组是混乱的，还是重置一下。
  int csTotalBytes = roundUp(MAX_LENGTH * CHAR_ELEM_SIZE, COPY_ELEM_SIZE);
  set16BLoop(tid, blkStatus.collStatusAlign.collStatus, zeros, csTotalBytes);
  
  // int csDoneBytes = 0;
  // while (csDoneBytes < csTotalBytes) {
  //   int targetBytes = min(nthreads * COPY_ELEM_SIZE, csTotalBytes - csDoneBytes);
  //   set16B(tid, (char *)(blkStatus.collStatusAlign.collStatus) + csDoneBytes, zeros, targetBytes);
  //   csDoneBytes += targetBytes;
  // }

  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 4", thrdCudaDev, blockIdx.x, threadIdx.x);
  
  // 每次kernel启动的时候，都把各个coll的尝试次数重置。
  int ctcTotalBytes = roundUp(MAX_LENGTH * CHAR_ELEM_SIZE, COPY_ELEM_SIZE);
  set16BLoop(tid, blkStatus.collTryCntAllign.collTryCnt, zeros, ctcTotalBytes);
  // int ctcDoneBytes = 0;
  // while (ctcDoneBytes < ctcTotalBytes) {
  //   int targetBytes = min(nthreads * COPY_ELEM_SIZE, ctcTotalBytes - ctcDoneBytes);
  //   set16B(tid, (char *)(blkStatus.collTryCntAllign.collTryCnt) + ctcDoneBytes, zeros, targetBytes);
  //   ctcDoneBytes += targetBytes;
  // }
  
  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 5, blkStatus.dynamicBlkStatus.numActiveColls=%d, sizeof(DynamicBlkStatus)=%lu, 16*blockDim.x=%d, sizeof(StaticCollCtx)=%lu, sizeof(DynamicCollCtx)=%lu, hasQuitted=%d, &zeros=%p, zeros=%p, zeros+1=%p, zeros[0]=%llu, zeros[1]=%llu", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.dynamicBlkStatus.numActiveColls, sizeof(DynamicBlkStatus), 16*blockDim.x, sizeof(StaticCollCtx), sizeof(DynamicCollCtx), hasQuitted, &zeros, zeros, zeros+1, *zeros, *(zeros+1));

  if (hasQuitted == 0) {
    set16BLoop(tid, &blkStatus.dynamicBlkStatus, zeros, sizeof(DynamicBlkStatus));
    // blkStatus.dynamicBlkStatus.numActiveColls = 0;
      
    // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 6, blkStatus.dynamicBlkStatus.numActiveColls=%d", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.dynamicBlkStatus.numActiveColls);

    #ifdef DEBUG_CLOCK
      // 可以并行优化，看看有没有必要吧，每次循环的增量是blockDim.x
      ONE_THRD_DO
        #ifdef DEBUG_CLOCK_TRAIN
          for (int i = 0; i < collCount; ++i) {
            int collId = globalCollIds[i];
            blkStatus.beforeGetSqeIter[collId] = 0;
            blkStatus.getSqeIter[collId] = 0;
            // blkStatus.beforePutCqeIter[collId] = 0;
            // blkStatus.putCqeIter[collId] = 0;
            blkStatus.ctxSwitchCnt[collId] = 0;
            for (int j = 0; j < RECORD_ITER; ++j) {
              blkStatus.beforeGetSqeClock[collId][j] = 0;
              blkStatus.getSqeClock[collId][j] = 0;
              blkStatus.beforePutCqeClock[collId][j] = 0;
              blkStatus.putCqeClock[collId][j] = 0;
              blkStatus.beforeAfterGetSqeDeltaClock[collId][j] = 0;
              blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][j] = 0;
              blkStatus.beforeAfterPutCqeDeltaClock[collId][j] = 0;
              blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][j] = 0;
            }
          }
        #endif

        #ifdef DEBUG_CLOCK_3D
          for (int i = 0; i < collCount; ++i) {
            int collId = globalCollIds[i];
            blkStatus.switchCntAfterRecvSuccess[collId] = 0;
            blkStatus.switchCntBeforeRecvSuccess[collId] = 0;
            blkStatus.switchCntAfterRecvSuccessIterDelta[collId] = 0;
            blkStatus.switchCntBeforeRecvSuccessIterDelta[collId] = 0;
            // 这两个数组的下标就代表index，而不是collId。
            blkStatus.collIdInSqe[i] = 0;
            blkStatus.taskQLenAfterGetSqe[i] = 0;
            blkStatus.collId4Cq[i] = 0;
          }
          blkStatus.iterCqeCnt = 0;
          blkStatus.iterSqeCnt = 0;
          blkStatus.iterNum = 0;
          blkStatus.iterSqNum = 0;
          blkStatus.totalSqeCnt = 0;
          blkStatus.totalCqeCnt = 0;
        #endif
          
        #ifdef DEBUG_CLOCK_IO
          blkStatus.beforeGetSqeIter = 0;
          blkStatus.getSqeIter = 0;
          // blkStatus.beforePutCqeIter = 0;
          // blkStatus.putCqeIter = 0;
          for (int j = 0; j < RECORD_ITER; ++j) {
            blkStatus.beforeGetSqeClock[j] = 0;
            blkStatus.getSqeClock[j] = 0;
            blkStatus.beforePutCqeClock[j] = 0;
            blkStatus.putCqeClock[j] = 0;
            blkStatus.beforeAfterGetSqeDeltaClock[j] = 0;
            blkStatus.afterGetSqeBeforePutCqeDeltaClock[j] = 0;
            blkStatus.beforeAfterPutCqeDeltaClock[j] = 0;
            blkStatus.beforeGetSqeAfterPutCqeDeltaClock[j] = 0;

            blkStatus.afterReadSqEmptyDeltaClock[j] = 0;
            blkStatus.afterGetSqFrontierDeltaClock[j] = 0;
            blkStatus.afterAddSqFrontierCounterDeltaClock[j] = 0;
            blkStatus.afterUpdateSqHeadDeltaClock[j] = 0;
            blkStatus.afterRecordBuffDeltaClock[j] = 0;

            blkStatus.beforeOfcclFuncClock[j] = 0;
            blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[j] = 0;
            blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[j] = 0;
          }
          blkStatus.sqReadCnt = 0;
          blkStatus.cqWriteCnt = 0;
        #endif

        #ifdef DEBUG_CLOCK_CTX
          for (int j = 0; j < RECORD_ITER; ++j) {
            blkStatus.beforeLoadClock[j] = 0;
            blkStatus.afterLoadDeltaClock[j] = 0;
            blkStatus.beforeSaveClock[j] = 0;
            blkStatus.afterSaveDeltaClock[j] = 0;
          }
          blkStatus.loadIter = 0;
          blkStatus.saveIter = 0;
        #endif

      ONE_THRD_DO_END
    #endif

  } else {
    copy16BLoop(tid, &blkStatus.dynamicBlkStatus, &myGlobalBlkStatus->dynamicBlkStatus, sizeof(DynamicBlkStatus));
      
    #ifdef DEBUG_CLOCK
      // 可以并行优化，看看有没有必要吧，每次循环的增量是blockDim.x
        ONE_THRD_DO
        #ifdef DEBUG_CLOCK_TRAIN
            for (int i = 0; i < collCount; ++i) {
              int collId = globalCollIds[i];
              for (int j = 0; j < RECORD_ITER; ++j) {
                blkStatus.beforeGetSqeClock[collId][j] = myGlobalBlkStatus->beforeGetSqeClock[collId][j];
                blkStatus.getSqeClock[collId][j] = myGlobalBlkStatus->getSqeClock[collId][j];
                blkStatus.beforePutCqeClock[collId][j] = myGlobalBlkStatus->beforePutCqeClock[collId][j];
                blkStatus.putCqeClock[collId][j] = myGlobalBlkStatus->putCqeClock[collId][j];
                
                blkStatus.beforeAfterGetSqeDeltaClock[collId][j] = myGlobalBlkStatus->beforeAfterGetSqeDeltaClock[collId][j];
                blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][j] = myGlobalBlkStatus->afterGetSqeBeforePutCqeDeltaClock[collId][j];
                blkStatus.beforeAfterPutCqeDeltaClock[collId][j] = myGlobalBlkStatus->beforeAfterPutCqeDeltaClock[collId][j];
                blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][j] = myGlobalBlkStatus->beforeGetSqeAfterPutCqeDeltaClock[collId][j];
              }
              blkStatus.beforeGetSqeIter[collId] = myGlobalBlkStatus->beforeGetSqeIter[collId];
              blkStatus.getSqeIter[collId] = myGlobalBlkStatus->getSqeIter[collId];

              blkStatus.ctxSwitchCnt[collId] = myGlobalBlkStatus->ctxSwitchCnt[collId];
              // blkStatus.beforePutCqeIter[collId] = myGlobalBlkStatus->beforePutCqeIter[collId];
              // blkStatus.putCqeIter[collId] = myGlobalBlkStatus->putCqeIter[collId];
            }
          #endif

          #ifdef DEBUG_CLOCK_3D
            for (int i = 0; i < collCount; ++i) {
              int collId = globalCollIds[i];
              blkStatus.switchCntAfterRecvSuccess[collId] = myGlobalBlkStatus->switchCntAfterRecvSuccess[collId];
              blkStatus.switchCntBeforeRecvSuccess[collId] = myGlobalBlkStatus->switchCntBeforeRecvSuccess[collId];
              blkStatus.switchCntAfterRecvSuccessIterDelta[collId] = myGlobalBlkStatus->switchCntAfterRecvSuccessIterDelta[collId];
              blkStatus.switchCntBeforeRecvSuccessIterDelta[collId] = myGlobalBlkStatus->switchCntBeforeRecvSuccessIterDelta[collId];
              
              blkStatus.collIdInSqe[i] = myGlobalBlkStatus->collIdInSqe[i];
              blkStatus.taskQLenAfterGetSqe[i] = myGlobalBlkStatus->taskQLenAfterGetSqe[i];
              blkStatus.collId4Cq[i] = myGlobalBlkStatus->collId4Cq[i];
            }
            blkStatus.iterCqeCnt = myGlobalBlkStatus->iterCqeCnt;
            blkStatus.iterSqeCnt = myGlobalBlkStatus->iterSqeCnt;
            blkStatus.iterNum = myGlobalBlkStatus->iterNum;
            blkStatus.iterSqNum = myGlobalBlkStatus->iterSqNum;
            blkStatus.totalSqeCnt = myGlobalBlkStatus->totalSqeCnt;
            blkStatus.totalCqeCnt = myGlobalBlkStatus->totalCqeCnt;
          #endif
          #ifdef DEBUG_CLOCK_IO
            for (int j = 0; j < RECORD_ITER; ++j) {
              blkStatus.beforeGetSqeClock[j] = myGlobalBlkStatus->beforeGetSqeClock[j];
              blkStatus.getSqeClock[j] = myGlobalBlkStatus->getSqeClock[j];
              blkStatus.beforePutCqeClock[j] = myGlobalBlkStatus->beforePutCqeClock[j];
              blkStatus.putCqeClock[j] = myGlobalBlkStatus->putCqeClock[j];
              
              blkStatus.beforeAfterGetSqeDeltaClock[j] = myGlobalBlkStatus->beforeAfterGetSqeDeltaClock[j];
              blkStatus.afterGetSqeBeforePutCqeDeltaClock[j] = myGlobalBlkStatus->afterGetSqeBeforePutCqeDeltaClock[j];
              blkStatus.beforeAfterPutCqeDeltaClock[j] = myGlobalBlkStatus->beforeAfterPutCqeDeltaClock[j];
              blkStatus.beforeGetSqeAfterPutCqeDeltaClock[j] = myGlobalBlkStatus->beforeGetSqeAfterPutCqeDeltaClock[j];

              blkStatus.afterReadSqEmptyDeltaClock[j] = myGlobalBlkStatus->afterReadSqEmptyDeltaClock[j];
              blkStatus.afterGetSqFrontierDeltaClock[j] = myGlobalBlkStatus->afterGetSqFrontierDeltaClock[j];
              blkStatus.afterAddSqFrontierCounterDeltaClock[j] = myGlobalBlkStatus->afterAddSqFrontierCounterDeltaClock[j];
              blkStatus.afterUpdateSqHeadDeltaClock[j] = myGlobalBlkStatus->afterUpdateSqHeadDeltaClock[j];
              blkStatus.afterRecordBuffDeltaClock[j] = myGlobalBlkStatus->afterRecordBuffDeltaClock[j];

              blkStatus.beforeOfcclFuncClock[j] = myGlobalBlkStatus->beforeOfcclFuncClock[j];
              blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[j] = myGlobalBlkStatus->afterGetSqeBeforeOfcclFuncDeltaClock[j];
              blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[j] = myGlobalBlkStatus->afterGetSqeBeforeMaintainSharedCtxDeltaClock[j];
            }
            blkStatus.beforeGetSqeIter = myGlobalBlkStatus->beforeGetSqeIter;
            blkStatus.getSqeIter = myGlobalBlkStatus->getSqeIter;

            blkStatus.sqReadCnt = myGlobalBlkStatus->sqReadCnt;
            blkStatus.cqWriteCnt = myGlobalBlkStatus->cqWriteCnt;
          #endif
          #ifdef DEBUG_CLOCK_CTX
            for (int j = 0; j < RECORD_ITER; ++j) {
              blkStatus.beforeLoadClock[j] = myGlobalBlkStatus->beforeLoadClock[j];
              blkStatus.afterLoadDeltaClock[j] = myGlobalBlkStatus->afterLoadDeltaClock[j];
              blkStatus.beforeSaveClock[j] = myGlobalBlkStatus->beforeSaveClock[j];
              blkStatus.afterSaveDeltaClock[j] = myGlobalBlkStatus->afterSaveDeltaClock[j];
            }
            blkStatus.loadIter = myGlobalBlkStatus->loadIter;
            blkStatus.saveIter = myGlobalBlkStatus->saveIter;
          #endif
        ONE_THRD_DO_END
    #endif
  }

  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 7", thrdCudaDev, blockIdx.x, threadIdx.x);

  int bcTotalBytes = roundUp(MAX_LENGTH * CHAR_ELEM_SIZE, COPY_ELEM_SIZE); // 这里不应该用collCount，因为blkCount4Coll相当于是数组模拟的map，我们不应该假设coll_id连续增长。
  copy16BLoop(tid, sharedBlkCount4CollAlign.blkCount4Coll, globalBlkCount4Coll, bcTotalBytes);
  // int bcDoneBytes = 0;
  // while (bcDoneBytes < bcTotalBytes) {
  //   int targetBytes = min(nthreads * COPY_ELEM_SIZE, bcTotalBytes - bcDoneBytes);
  //   copy16B(tid, (char *)(sharedBlkCount4CollAlign.blkCount4Coll) + bcDoneBytes, (char *)globalBlkCount4Coll + bcDoneBytes, targetBytes);
  //   bcDoneBytes += targetBytes;
  // }

  // OFCCL_LOG(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, before ofcclBarrier(1)", thrdCudaDev, blockIdx.x, threadIdx.x);

  ofcclBarrier(1); // 为了下边读取blkStatus.dynamicBlkStatus.numActiveColls

  // OFCCL_LOG(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, after ofcclBarrier(1)", thrdCudaDev, blockIdx.x, threadIdx.x);

  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 8, blkStatus.dynamicBlkStatus.numActiveColls=%d", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.dynamicBlkStatus.numActiveColls);
  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 8, blkStatus.dynamicBlkStatus.numActiveColls=%d", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.dynamicBlkStatus.numActiveColls);

  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 9", thrdCudaDev, blockIdx.x, threadIdx.x);

  int acTotalBytes = roundUp(blkStatus.dynamicBlkStatus.numActiveColls * SHORT_ELEM_SIZE, COPY_ELEM_SIZE);
  copy16BLoop(tid, blkStatus.activeCollIdsAlign.activeCollIds, myGlobalBlkStatus->activeCollIdsAlign.activeCollIds, acTotalBytes);
  // OFCCL_LOG(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 10, acTotalBytes=%d", thrdCudaDev, blockIdx.x, threadIdx.x, acTotalBytes);
  // int acDoneBytes = 0;
  // // 这个要不要复制，需要读取numActiveColls，所以必须得上边做完，加一个barrier之后才可以。
  // while (acDoneBytes < acTotalBytes) {
  //   int targetBytes = min(nthreads * COPY_ELEM_SIZE, acTotalBytes - acDoneBytes);
  //   copy16B(tid, (char *)(blkStatus.activeCollIdsAlign.activeCollIds) + acDoneBytes, (char *)(&myGlobalBlkStatus->activeCollIdsAlign.activeCollIds) + acDoneBytes, targetBytes);
  //   acDoneBytes += targetBytes;
  // }

  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, flag 11", thrdCudaDev, blockIdx.x, threadIdx.x);
  return;
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
  #ifdef DEBUG_CLOCK
    #ifdef DEBUG_CLOCK_TRAIN
      // int tid = threadIdx.x;
      long long int beforeGetSqeClock = clock64();
    #endif
  #endif

  SQE target;

  // 能读到，假如是正常SQE，把信息在任务列表里记录一下；假如是quit，那也记录一下
  // 读不到新东西那就算了

  if (sqRead(sq, &target, thrdCudaDev) == -1) {
    *unprogressedCnt += 1;
    // if (blkStatus.dynamicBlkStatus.numActiveColls > 0) {
      
    //   // 没读到新的，应该不用处理taskQ了，因为每次遍历一次taskQ，都会处理。 
    // }
    return;
  } else {
    if (target.quit) {
      blkStatus.quit = 1; // TODO: 从鲁棒性的角度来说，这里应该有机制保证看到这个quit sqe的时候，taskQ里的所有sqe也应该都处理完，才能退出。（不过目前可以先不管，可以由用户程序间接保证）；一个简单的保证方法是，加一个check。
      blkStatus.finallyQuit = 1;
      // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, read quit SQE", thrdCudaDev, blockIdx.x, threadIdx.x);
      // if (bid == 0) {
        *finallyQuit = 1; // TODO: 为了最后每个block都保证打印统计信息，挺不优雅的
      // }
      return;
    }

    #ifdef ARRAY_DEBUG
      if (threadIdx.x == 0) {
        *(blkStatus.barrierCnt + 2 + 5 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      }
    #endif

    // 正常读到了SQE的话，需要往global的globalBlk2CollId2CollCtx表项里边写入，更新blkStatus.numActiveColls
    int newActiveCollId = target.collId;
    int blkLimit = sharedBlkCount4CollAlign.blkCount4Coll[newActiveCollId]; // 需要参与新读到的coll的block才会进行后续操作。

    #ifdef DEBUG_CLOCK
      #ifdef DEBUG_CLOCK_TRAIN
        if (blkStatus.beforeGetSqeIter[newActiveCollId] >= SKIP_WARMUP_ITER) {
          int iter = (blkStatus.beforeGetSqeIter[newActiveCollId] - SKIP_WARMUP_ITER) % RECORD_ITER;
          blkStatus.beforeGetSqeClock[newActiveCollId][iter] = beforeGetSqeClock; // 只有真的读到sqe才记录这个值。
        }
        ++blkStatus.beforeGetSqeIter[newActiveCollId];

        if (blkStatus.getSqeIter[newActiveCollId] >= SKIP_WARMUP_ITER) {
          int iter = (blkStatus.getSqeIter[newActiveCollId] - SKIP_WARMUP_ITER) % RECORD_ITER;
          blkStatus.getSqeClock[newActiveCollId][iter] = clock64();

          blkStatus.beforeAfterGetSqeDeltaClock[newActiveCollId][iter] = calcDeltaClock(blkStatus.beforeGetSqeClock[newActiveCollId][iter], blkStatus.getSqeClock[newActiveCollId][iter]);
        }
        ++blkStatus.getSqeIter[newActiveCollId];
      #endif
      
      #ifdef DEBUG_CLOCK_IO
        if (blkStatus.getSqeIter >= SKIP_WARMUP_ITER) {
          int iter = (blkStatus.getSqeIter - SKIP_WARMUP_ITER) % RECORD_ITER;
          blkStatus.getSqeClock[iter] = clock64();

          blkStatus.beforeAfterGetSqeDeltaClock[iter] = calcDeltaClock(blkStatus.beforeGetSqeClock[iter], blkStatus.getSqeClock[iter]);
        }
        ++blkStatus.getSqeIter;
      #endif
      
    #endif

    *unprogressedCnt = 0;
    // OFCCL_LOG(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d>, read SQE for coll_id = %d, reset *unprogressedCnt = 0", thrdCudaDev, blockIdx.x, threadIdx.x, newActiveCollId);

    if (bid < blkLimit) {
      CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + newActiveCollId;
      if (blkStatus.collStatusAlign.collStatus[newActiveCollId] != 0) { // 应该没有重入的风险。重入指一个正在执行的集合通信又被提起请求。
        OFCCL_LOG(OFCCL_FATAL, "Rank<%d> Blk<%d> Thrd<%d> globalCollCtx4Blk7Coll->executing should be 0! sq->head = %llu, sq->tail = %llu, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu", thrdCudaDev, bid, threadIdx.x, DevLogicSqHead(sq), DevLogicSqTail(sq), DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier));
      }

      blkStatus.collStatusAlign.collStatus[newActiveCollId] = 1;
      
      #ifdef CQE_DEBUG_RANK_X
        OFCCL_LOG_RANK_X(OFCCL_CQE, CQE_DEBUG_RANK_X, "Rank<%d> Blk<%d> Thrd<%d>, read %lluth SQE for coll_id = %d, sq->head = %llu, sq->tail = %llu, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->sqeReadCnt), newActiveCollId, DevLogicSqHead(sq), DevLogicSqTail(sq), DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier));
      #endif
      #ifdef CQE_DEBUG_ALL_RANK
        OFCCL_LOG(OFCCL_CQE, "Rank<%d> Blk<%d> Thrd<%d>, read %lluth SQE for coll_id = %d, sq->head = %llu, sq->tail = %llu, blkStatus.dynamicBlkStatus.sqReadFrontier = %llu", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->sqeReadCnt), newActiveCollId, DevLogicSqHead(sq), DevLogicSqTail(sq), DevRingBufferLogicFrontier(sq, blkStatus.dynamicBlkStatus.sqReadFrontier));
      #endif
      
      globalCollCtx4Blk7Coll->staticCollCtx.workElem.sendbuff = target.sendbuff;
      globalCollCtx4Blk7Coll->staticCollCtx.workElem.recvbuff = target.recvbuff;

      #ifdef DEBUG_CLOCK
        #ifdef DEBUG_CLOCK_IO
          if (blkStatus.getSqeIter >= SKIP_WARMUP_ITER + 1) {
            int iter = (blkStatus.getSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
            long long int afterRecordBuffClock = clock64();
            blkStatus.afterRecordBuffDeltaClock[iter] = calcDeltaClock(blkStatus.getSqeClock[iter], afterRecordBuffClock);
          }
        #endif
      #endif

      // maintain the taskQ here.
      // 新加入的集合通信放在末位，最后执行。如果新加入的集合通信存在于当前的blkStatus.activeCollIds里边，也不必强行放到末位。
      int new_numActiveColls = 0;
      bool newActiveCollId_in_taskQ = false;
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
      if (!newActiveCollId_in_taskQ) { // TODO: newActiveCollId应该放在队头
        blkStatus.activeCollIdsAlign.activeCollIds[new_numActiveColls++] = newActiveCollId;
      }

      blkStatus.dynamicBlkStatus.numActiveColls = new_numActiveColls;

      #ifdef DEBUG_CLOCK_3D
        blkStatus.collIdInSqe[blkStatus.iterSqeCnt] = newActiveCollId;
        blkStatus.taskQLenAfterGetSqe[blkStatus.iterSqeCnt] = new_numActiveColls;
        ++blkStatus.iterSqeCnt;
        ++blkStatus.totalSqeCnt;
      #endif

      #ifdef CQE_DEBUG_ALL_RANK
        logTaskQ(0, thrdCudaDev, -1);
      #elif defined(CQE_DEBUG_RANK_X)
        logTaskQ(0, thrdCudaDev, CQE_DEBUG_RANK_X);
      #endif
    }
  }
}

static __device__ void loadCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId, int64_t BASE_CTX_SWITCH_THRESHOLD) {
  int tid = threadIdx.x;
  // #ifdef ARRAY_DEBUG
  //   *(blkStatus.barrierCnt + 0 + 16 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  // #endif

  // TODO: 考虑让所有线程都执行常数初始化。
  ONE_THRD_DO
    #ifdef SHOW_CNT
      blkStatus.dynamicBlkStatus.totalCtxLoadCnt++;
    #endif

    sharedCollCtx[collId % NUM_SHMEM_SLOT].progressed = 0;
  ONE_THRD_DO_END

  copy16BLoop(tid, &sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx, &globalCollCtx4Blk7Coll->dynamicCollCtx, sizeof(DynamicCollCtx));
  copy16BLoop(tid, &sharedCollCtx[collId % NUM_SHMEM_SLOT].staticCollCtx, &globalCollCtx4Blk7Coll->staticCollCtx, sizeof(StaticCollCtx));

  // #ifdef ARRAY_DEBUG
  //   *(blkStatus.barrierCnt + 1 + 16 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  // #endif
  return;
}

#ifdef DEBUG_PARA_SV
static __device__ void saveExcutingCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId) {
  int tid = threadIdx.x;
  #ifdef SHOW_CNT
    ONE_THRD_DO
      blkStatus.dynamicBlkStatus.totalCtxSaveCnt++;
    ONE_THRD_DO_END
  #endif
  copy16BLoop(tid, &globalCollCtx4Blk7Coll->dynamicCollCtx, &sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx, sizeof(DynamicCollCtx));
}
#else
static __device__ void saveExcutingCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId) {
  if(threadIdx.x == 0) {
    globalCollCtx4Blk7Coll->dynamicCollCtx.loadAgain = sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.loadAgain;
    globalCollCtx4Blk7Coll->dynamicCollCtx.slice4SimpleGenericOp = sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.slice4SimpleGenericOp;
    globalCollCtx4Blk7Coll->dynamicCollCtx.offset4SimpleGenericOp = sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.offset4SimpleGenericOp;
  
    globalCollCtx4Blk7Coll->dynamicCollCtx.currentStep4RunRing = sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.currentStep4RunRing;
    globalCollCtx4Blk7Coll->dynamicCollCtx.gridOffset4RunRing = sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.gridOffset4RunRing;
  
    #ifdef SHOW_CNT
      blkStatus.dynamicBlkStatus.totalCtxSaveCnt++;
    #endif
  }
}
#endif

static __device__ void maintainSharedCollCtx(int thrdCudaDev, CollCtx *globalBlk2CollId2CollCtx, int collId, int64_t BASE_CTX_SWITCH_THRESHOLD, int *unprogressedCnt) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int slotIdxToUse = collId % NUM_SHMEM_SLOT;
  int collIdOfThatSlot = sharedCollCtx[slotIdxToUse].staticCollCtx.collId;
  
  bool noLoadedColl = (collIdOfThatSlot == -1);
  bool sameLoadedColl = (collId == collIdOfThatSlot); // 这个条件成立的情况不止一种。
  bool loadedCollProgressed7SaveCtx7Quit = !noLoadedColl && (blkStatus.collStatusAlign.collStatus[collIdOfThatSlot] == -1); // 只有progressed，才需要save。

  bool needSave = !sameLoadedColl && loadedCollProgressed7SaveCtx7Quit;
  bool needLoad = noLoadedColl || !sameLoadedColl;

  // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> before run coll_id = %d, numActiveColls=%d, collIdOfThatSlot=%d, loadedCollProgressed7SaveCtx7Quit=%d, needSave=%d, needLoad=%d", thrdCudaDev, blockIdx.x, threadIdx.x, collId, blkStatus.dynamicBlkStatus.numActiveColls, collIdOfThatSlot, loadedCollProgressed7SaveCtx7Quit, needSave, needLoad);
  
  if (needSave) {
    // bugfix: save的时候，不应该save到即将load的coll的global collCtx副本里。
    CollCtx *globalCollCtx4Blk7OldColl = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collIdOfThatSlot;

    #ifdef DEBUG_CLOCK_CTX
      if (threadIdx.x == 0) {
        int iter = (blkStatus.saveIter) % RECORD_ITER;
        blkStatus.beforeSaveClock[iter] = clock64();
      }
    #endif

    saveExcutingCollCtx(thrdCudaDev, globalCollCtx4Blk7OldColl, collIdOfThatSlot);

    #ifdef DEBUG_CLOCK_CTX
      if (threadIdx.x == 0) {
        int iter = (blkStatus.saveIter) % RECORD_ITER;
        long long int afterSaveClock = clock64();
        blkStatus.afterSaveDeltaClock[iter] = calcDeltaClock(blkStatus.beforeSaveClock[iter], afterSaveClock);
        ++blkStatus.saveIter;
      }
    #endif

    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> save ctx for coll_id = %d, slice4SimpleGenericOp=%d, offset4SimpleGenericOp=%d, currentStep4RunRing=%d, gridOffset4RunRing=%ld", thrdCudaDev, blockIdx.x, threadIdx.x, collIdOfThatSlot, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.slice4SimpleGenericOp, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.offset4SimpleGenericOp, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.currentStep4RunRing, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.gridOffset4RunRing);
  }

  if (needLoad) {
    #ifdef DEBUG_CLOCK_CTX
      if (threadIdx.x == 0) {
        int iter = (blkStatus.loadIter) % RECORD_ITER;
        blkStatus.beforeLoadClock[iter] = clock64();
      }
    #endif

    CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collId;
    loadCollCtx(thrdCudaDev, globalCollCtx4Blk7Coll, collId, BASE_CTX_SWITCH_THRESHOLD);

    #ifdef DEBUG_CLOCK_CTX
      if (threadIdx.x == 0) {
        int iter = (blkStatus.loadIter) % RECORD_ITER;
        long long int afterLoadClock = clock64();
        blkStatus.afterLoadDeltaClock[iter] = calcDeltaClock(blkStatus.beforeLoadClock[iter], afterLoadClock);
        ++blkStatus.loadIter;
      }
    #endif

    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> load ctx for coll_id = %d, loadAgain=%d, slice4SimpleGenericOp=%d, offset4SimpleGenericOp=%d, currentStep4RunRing=%d, gridOffset4RunRing=%ld", thrdCudaDev, blockIdx.x, threadIdx.x, collId, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.loadAgain, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.slice4SimpleGenericOp, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.offset4SimpleGenericOp, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.currentStep4RunRing, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.gridOffset4RunRing);
  }

  if (tid == 0) {
    blkStatus.currLoadedCollId = collId; // 这个变量只起一个传递信息的作用了，不再标记shmem是否valid
    
    ++blkStatus.collTryCntAllign.collTryCnt[collId]; // tryCnt增加是不应该受干扰的行为。
    getInitSwitchThreshold(collId, BASE_CTX_SWITCH_THRESHOLD); // 设置了sharedCollCtx[].ctxSwitchThreshold
    sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].saveCtx7Quit = 0;
    sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].recvSuccess = 0;

    if (blkStatus.collStatusAlign.collStatus[collId] == -1) {
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, blkStatus.collStatusAlign.collStatus is %d, sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].ctxSwitchThreshold = %ld", thrdCudaDev, blockIdx.x, threadIdx.x, collId, blkStatus.collStatusAlign.collStatus[collId], sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].ctxSwitchThreshold);
      *unprogressedCnt = 0; // 这表明有coll前进了，只不过没跑完。
      
    } else if (blkStatus.collStatusAlign.collStatus[collId] == -2) {
      *unprogressedCnt += 1;
    }

    #ifdef DEBUG_CLOCK_TRAIN
      if (blkStatus.collStatusAlign.collStatus[collId] < 0) {
        ++blkStatus.ctxSwitchCnt[collId];
      }
    #endif

    blkStatus.collStatusAlign.collStatus[collId] = 1; // 每次准备执行的时候，重置为正常执行状态。新的coll已经是1，不过不要浪费if了。 
  }

  ofcclBarrier(4);
  return;
}

static __device__ void manipulateCQ7ResetDoneColl(int thrdCudaDev, int doneCollId, CQ *cq, CQE *globalCqes, CollCtx *globalBlk2CollId2CollCtx) {

  CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + blockIdx.x * MAX_LENGTH + doneCollId;
  // 放在这里，让每个完成了coll的block都打印自己的情况，而不是只有最终写cqe的那个block才报告。
  #ifdef DEBUG_CLOCK_3D
    ++blkStatus.iterCqeCnt;
    ++blkStatus.totalCqeCnt;
    blkStatus.collId4Cq[blkStatus.iterCqeCnt] = doneCollId;
    int collCnt4Blk = getCollCnt4Blk();
    if (blkStatus.iterCqeCnt % collCnt4Blk == 0) {
      // 完成了一个iter所需的集合通信，打印到目前为止的总的switch数，以及这个iter的switch数，并且清零iterCqeCnt和这个iter的switch的数。
      for (int i = 0; i < collCnt4Blk; ++i) {
        int sqeCollId = blkStatus.collIdInSqe[i];
        int taskQLen = blkStatus.taskQLenAfterGetSqe[i];
        #ifdef DEBUG_CLOCK_3D_HOST
          int blk = blockIdx.x, iter = blkStatus.iterNum;
          *getSlot(taskQLen4RankBlkIterColl, blk, iter, sqeCollId, NUM_ITER, numColl) = taskQLen;
          *getSlot(collIdInSqe4RankBlkIterColl, blk, iter, i, NUM_ITER, numColl) = sqeCollId;
        #else
          if (thrdCudaDev == 1) {
            OFCCL_LOG(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> in %dth iter, after get sqe for coll_id = %d (%d), taskQLen = %d", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.iterNum, sqeCollId, sharedBlkCount4CollAlign.blkCount4Coll[sqeCollId], taskQLen);
          }
        #endif

        // 数组元素不用置零，反正下次再用也是赋值。
      }
      // if (thrdCudaDev == 1) {
      //   OFCCL_LOG(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> in %dth iter, totalSqeCnt = %d, totalCqeCnt = %d, iterSqeCnt = %d, iterCqeCnt = %d", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.iterNum, blkStatus.totalSqeCnt, blkStatus.totalCqeCnt, blkStatus.iterSqeCnt, blkStatus.iterCqeCnt);
      // }
      blkStatus.iterSqeCnt = 0;
      for (int i = 0; i < collCnt4Blk; ++i) {
        int collId = blkStatus.collId4Cq[i];
        #ifdef DEBUG_CLOCK_3D_HOST
          int blk = blockIdx.x, iter = blkStatus.iterNum;
          *getSlot(unprogressed7SwitchCnt4RankBlkIterColl, blk, iter, collId, NUM_ITER, numColl) = blkStatus.switchCntBeforeRecvSuccessIterDelta[collId];
          *getSlot(progressed7SwitchCnt4RankBlkIterColl, blk, iter, collId, NUM_ITER, numColl) = blkStatus.switchCntAfterRecvSuccessIterDelta[collId];
          *getSlot(unprogressed7SwitchCntTotal4RankBlkIterColl, blk, iter, collId, NUM_ITER, numColl) = blkStatus.switchCntBeforeRecvSuccess[collId];
          *getSlot(progressed7SwitchCntTotal4RankBlkIterColl, blk, iter, collId, NUM_ITER, numColl) = blkStatus.switchCntAfterRecvSuccess[collId];
          *getSlot(collId4Cq4RankBlkIterColl, blk, iter, i, NUM_ITER, numColl) = collId;
        #else
          if (thrdCudaDev == 1) {
            OFCCL_LOG(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> done %dth iter, coll_id = %d (%d), switchCntAfterRecvSuccessIterDelta = %d, switchCntBeforeRecvSuccessIterDelta = %d, switchCntAfterRecvSuccess = %d, switchCntBeforeRecvSuccess = %d", thrdCudaDev, blockIdx.x, threadIdx.x, blkStatus.iterNum, collId, sharedBlkCount4CollAlign.blkCount4Coll[collId], blkStatus.switchCntAfterRecvSuccessIterDelta[collId], blkStatus.switchCntBeforeRecvSuccessIterDelta[collId], blkStatus.switchCntAfterRecvSuccess[collId], blkStatus.switchCntBeforeRecvSuccess[collId]);
          }
        #endif
        blkStatus.switchCntBeforeRecvSuccessIterDelta[collId] = 0;  
        blkStatus.switchCntAfterRecvSuccessIterDelta[collId] = 0;
      }
      blkStatus.iterCqeCnt = 0;
      ++blkStatus.iterNum;
    }
  #endif
  
  // 协调所有blk，发现所有blk都完成，最后一个blk发送CQE
  int old_counter = atomicAdd(&(globalCqes[doneCollId].counter), 1);
  __threadfence(); // cqes在global memory里边，全部block关心。

  // *(blkStatus.collCounters + 0 + doneCollId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;

  // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prepare %lluth CQE for coll_id = %d", thrdCudaDev, blockIdx.x, threadIdx.x, ++(globalCollCtx4Blk7Coll->cqePrepareCnt), doneCollId);

  if (old_counter + 1 == sharedBlkCount4CollAlign.blkCount4Coll[doneCollId]) {
    atomicExch(&globalCqes[doneCollId].counter, 0);

    #ifdef DEBUG_CLOCK
      #ifdef DEBUG_CLOCK_TRAIN
        if (blkStatus.getSqeIter[doneCollId] >= SKIP_WARMUP_ITER + 1) {
          int iter = (blkStatus.getSqeIter[doneCollId] - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
          blkStatus.beforePutCqeClock[doneCollId][iter] = clock64();
        }
      #endif
      #ifdef DEBUG_CLOCK_IO
        if (blkStatus.getSqeIter >= SKIP_WARMUP_ITER + 1) {
          int iter = (blkStatus.getSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
          blkStatus.beforePutCqeClock[iter] = clock64();
        }
      #endif
      
    #endif

    #if defined(CQE_DEBUG_RANK_X) || defined(CQE_DEBUG_ALL_RANK)
      CollCtx *globalCollCtx4Blk_0_7Coll = globalBlk2CollId2CollCtx + 0 * MAX_LENGTH + doneCollId;
      unsigned long long int *cqeWriteCnt = &globalCollCtx4Blk_0_7Coll->cqeWriteCnt;
      while (cqWrite(cq, doneCollId, thrdCudaDev, cqeWriteCnt) == -1) {
      }
    #else
      while (cqWrite(cq, doneCollId, thrdCudaDev, nullptr) == -1) {
      }
    #endif


    // *(blkStatus.collCounters + 1 + doneCollId * COLL_COUNTER_INNER_SIZE + blockIdx.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;

    #ifdef DEBUG_CLOCK
      #ifdef DEBUG_CLOCK_TRAIN
        if (blkStatus.getSqeIter[doneCollId] >= SKIP_WARMUP_ITER + 1) {
          int iter = (blkStatus.getSqeIter[doneCollId] - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
          blkStatus.putCqeClock[doneCollId][iter] = clock64();

          blkStatus.afterGetSqeBeforePutCqeDeltaClock[doneCollId][iter] = calcDeltaClock(blkStatus.getSqeClock[doneCollId][iter], blkStatus.beforePutCqeClock[doneCollId][iter]);
          blkStatus.beforeAfterPutCqeDeltaClock[doneCollId][iter] = calcDeltaClock(blkStatus.beforePutCqeClock[doneCollId][iter], blkStatus.putCqeClock[doneCollId][iter]);
          // blkStatus.afterGetSqeAfterPutCqeDeltaClock[doneCollId][iter] = calcDeltaClock(blkStatus.getSqeClock[doneCollId][iter], blkStatus.putCqeClock[doneCollId][iter]);
          blkStatus.beforeGetSqeAfterPutCqeDeltaClock[doneCollId][iter] = calcDeltaClock(blkStatus.beforeGetSqeClock[doneCollId][iter], blkStatus.putCqeClock[doneCollId][iter]);
        }
      #endif

      #ifdef DEBUG_CLOCK_IO
        if (blkStatus.getSqeIter >= SKIP_WARMUP_ITER + 1) {
          int iter = (blkStatus.getSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
          blkStatus.putCqeClock[iter] = clock64();

          blkStatus.afterGetSqeBeforePutCqeDeltaClock[iter] = calcDeltaClock(blkStatus.getSqeClock[iter], blkStatus.beforePutCqeClock[iter]);
          blkStatus.beforeAfterPutCqeDeltaClock[iter] = calcDeltaClock(blkStatus.beforePutCqeClock[iter], blkStatus.putCqeClock[iter]);
          // blkStatus.afterGetSqeAfterPutCqeDeltaClock[iter] = calcDeltaClock(blkStatus.getSqeClock[iter], blkStatus.putCqeClock[iter]);
          blkStatus.beforeGetSqeAfterPutCqeDeltaClock[iter] = calcDeltaClock(blkStatus.beforeGetSqeClock[iter], blkStatus.putCqeClock[iter]);
        }
      #endif
      
    #endif

    __threadfence();
  }

  // bugfix: manipulateCQ7ResetDoneColl的调用时机，是traverseTaskQ外边，又遍历一次taskQ，所以要判断一下。不判断会导致另一个coll从上次save的地方重新load，重新做已经完成的搬运，但是peer rank未必能配合，就会卡住。
  if (doneCollId == blkStatus.currLoadedCollId) {
    blkStatus.currLoadedCollId = -1;
  }
  // bugfix: 启用了slot之后，对于完成的coll，要把其占用的slot给invalid掉
  // bugfix: 但是也得判断那个slot装的是不是自己。
  if (sharedCollCtx[doneCollId % NUM_SHMEM_SLOT].staticCollCtx.collId == doneCollId) {
    sharedCollCtx[doneCollId % NUM_SHMEM_SLOT].staticCollCtx.collId = -1;

    int group = 0; // this->group = group & (uint16_t)0xFFFF; // 还是 0. ring的情况下，group一直是0。
    int index = 0; // TODO: 这样的指定或许是有问题的。// index = tid % ThreadPerSync; // 当前线程在 “g” 里的位置 // Peer index I'm responsible for
    void **slot = sharedCollCtx[doneCollId % NUM_SHMEM_SLOT].groups[group].sendConns[index]->ptrExchange;
    if (slot != nullptr) {
      *slot = nullptr; // bugfix：应该把这个重置放到coll完成的时候，来实现一种peer间的同步。
    }
  }

  blkStatus.collStatusAlign.collStatus[doneCollId] = 0;

  // ResetDoneColl
  globalCollCtx4Blk7Coll->dynamicCollCtx.loadAgain = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.slice4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.offset4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.currentStep4RunRing = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.gridOffset4RunRing = 0;
}

static __device__ void traverseTaskQ(int thrdCudaDev, CollCtx *globalBlk2CollId2CollCtx, int collCount, CQ *cq, CQE *globalCqes, int *unprogressedCnt, int64_t BASE_CTX_SWITCH_THRESHOLD) {
  int bid = blockIdx.x;

  #if defined(ARRAY_DEBUG)
    *(blkStatus.barrierCnt + 0 + 11 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  #endif

  for (int i = 0; i < blkStatus.dynamicBlkStatus.numActiveColls; i++) {

    int collId = blkStatus.activeCollIdsAlign.activeCollIds[i];
    int blkLimit = sharedBlkCount4CollAlign.blkCount4Coll[collId];

    if (bid < blkLimit) { // blk天然分化，保留这个条件
      int try_cnt = getTryNum(i); // 只有队头循环多次。
      for (int tryCnt = 0; tryCnt < try_cnt; ++tryCnt) {

        // ***** 先准备好sharedCollCtx，全部线程都参与 *****
        #ifdef DEBUG_CLOCK_IO
          if (threadIdx.x == 0) {
            if (blkStatus.beforeGetSqeIter >= SKIP_WARMUP_ITER + 1) {
              int iter = (blkStatus.beforeGetSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
              blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[iter] = calcDeltaClock(blkStatus.getSqeClock[iter], clock64());
            }
          }
        #endif
        maintainSharedCollCtx(thrdCudaDev, globalBlk2CollId2CollCtx, collId, BASE_CTX_SWITCH_THRESHOLD, unprogressedCnt);

        // ***** 然后调用ofcclFunc *****
        int wid = threadIdx.x / WARP_SIZE;
        if (wid < sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.workElem.header.nWarps) {
          
          // OFCCL_LOG_THRD_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d>, before running coll_id = %d", thrdCudaDev, blockIdx.x, threadIdx.x, collId);

          #ifdef DEBUG_CLOCK_IO
            if (threadIdx.x == 0) {
              if (blkStatus.beforeGetSqeIter >= SKIP_WARMUP_ITER + 1) {
                int iter = (blkStatus.beforeGetSqeIter - SKIP_WARMUP_ITER - 1) % RECORD_ITER;
                blkStatus.beforeOfcclFuncClock[iter] = clock64();
                blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[iter] = calcDeltaClock(blkStatus.getSqeClock[iter], blkStatus.beforeOfcclFuncClock[iter]);
              }
            }
          #endif

          ofcclFuncs[sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.workElem.header.funcIndex](); // 这里边的调用里不涉及__syncthreads().

          // OFCCL_LOG_THRD_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d>, ofcclFuncs of coll_id = %d returns, blkStatus.collStatusAlign.collStatus[collId] = %d", thrdCudaDev, blockIdx.x, threadIdx.x, collId, blkStatus.collStatusAlign.collStatus[collId]);
        }
        ofcclBarrier(3);  // 跑完一个集合通信，同步一下。
        #if defined(SHOW_CNT) || defined(DEBUG_CLOCK_3D)
          if (threadIdx.x == 0) {
            int currUsedSlotId = collId % NUM_SHMEM_SLOT;
            if (blkStatus.collStatusAlign.collStatus[collId] < 0) {
              if (sharedCollCtx[currUsedSlotId].recvSuccess) {
                #if defined(SHOW_CNT)
                  blkStatus.dynamicBlkStatus.totalSwitchCntAfterRecvSuccess++;
                #endif
                #if defined(DEBUG_CLOCK_3D)
                  blkStatus.switchCntAfterRecvSuccess[collId]++;
                  blkStatus.switchCntAfterRecvSuccessIterDelta[collId]++;
                #endif
              } else {
                #if defined(SHOW_CNT)
                  blkStatus.dynamicBlkStatus.totalSwitchCntBeforeRecvSuccess++;
                #endif
                #if defined(DEBUG_CLOCK_3D)
                  blkStatus.switchCntBeforeRecvSuccess[collId]++;
                  blkStatus.switchCntBeforeRecvSuccessIterDelta[collId]++;
                #endif
              }
            }
          }
        #endif
        if (blkStatus.collStatusAlign.collStatus[collId] == 2) {
          if (threadIdx.x == 8) { // bugfix: 调整为8号线程写cqe，为的是在里边把ptrExchange处理了。
            *unprogressedCnt = 0;
  
            blkStatus.collTryCntAllign.collTryCnt[collId] = 0;
  
            // OFCCL_LOG_THRD_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d>, manipulateCQ7ResetDoneColl of coll_id = %d", thrdCudaDev, blockIdx.x, threadIdx.x, collId);
            manipulateCQ7ResetDoneColl(thrdCudaDev, collId, cq, globalCqes, globalBlk2CollId2CollCtx);
            // 对于完成执行的集合通信应该不用把shmem里的collCtx写回到global mem里边，sendbuff/recvbuff等下次的SQE传过来，剩下的其他都是些静态配置项。
          }
          
          #ifdef ARRAY_DEBUG
            if (threadIdx.x == 0) {
              *(blkStatus.barrierCnt + 3 + 5 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
            }
          #endif

          break;
        }
      }
      ofcclBarrier(3); // 放在这里不太好看，但是应该没啥问题。
    }
  }

  #if defined(ARRAY_DEBUG)
    *(blkStatus.barrierCnt + 2 + 11 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  #endif

  return;
}

// TODO: 考虑在按需启停的场景下，会多次启动，执行上会不会有什么变化。
__global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, char *globalBlkCount4Coll, int *globalThrdCount4Coll, short *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems, CollCtx *globalBlk2CollId2CollCtx, int *finallyQuit, BlkStatus *globalBlkStatus, unsigned long long int *barrierCnt, unsigned long long int *collCounters, const int64_t TOLERANT_UNPROGRESSED_CNT, const int64_t BASE_CTX_SWITCH_THRESHOLD) {

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel starts, blkStatus.dynamicBlkStatus.numActiveColls = %d", thrdCudaDev, blockIdx.x, tid, blkStatus.dynamicBlkStatus.numActiveColls);
  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel starts", thrdCudaDev, blockIdx.x, tid);
  // __syncwarp(); // ！！！！！！为了打印log加的！！！！

  // int tempRound = 0;
  
  // OFCCL_LOG_RANK_X_THRD_0(OFCCL_MPI, 0, "Rank<%d> Blk<%d> send conn info @ %p, send conn tail(RolePostSend) @ %p, send conn head(RoleWaitSend) @ %p,", thrdCudaDev, bid, &(globalBlk2CollId2CollCtx + bid * MAX_LENGTH + 0)->staticCollCtx.devPeers[(thrdCudaDev + 1) % 2].send[0].conn, (globalBlk2CollId2CollCtx + bid * MAX_LENGTH + 0)->staticCollCtx.devPeers[(thrdCudaDev + 1) % 2].send[0].conn.tail, (globalBlk2CollId2CollCtx + bid * MAX_LENGTH + 0)->staticCollCtx.devPeers[(thrdCudaDev + 1) % 2].send[0].conn.head);
  // OFCCL_LOG_RANK_X_THRD_0(OFCCL_MPI, 0, "Rank<%d> Blk<%d> recv conn info @ %p, recv conn head(RolePostRecv) @ %p, recv conn tail(RoleWaitRecv) @ %p,", thrdCudaDev, bid, &(globalBlk2CollId2CollCtx + bid * MAX_LENGTH + 0)->staticCollCtx.devPeers[(thrdCudaDev + 1) % 2].recv[0].conn, (globalBlk2CollId2CollCtx + bid * MAX_LENGTH + 0)->staticCollCtx.devPeers[(thrdCudaDev + 1) % 2].recv[0].conn.head, (globalBlk2CollId2CollCtx + bid * MAX_LENGTH + 0)->staticCollCtx.devPeers[(thrdCudaDev + 1) % 2].recv[0].conn.tail);

  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, before blockInit", thrdCudaDev, blockIdx.x, threadIdx.x);
  blockInit(thrdCudaDev, collCount, globalBlkCount4Coll, globalThrdCount4Coll, globalCollIds, globalDevComm7WorkElems, globalBlk2CollId2CollCtx, globalBlkStatus, barrierCnt, collCounters);
  // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, after blockInit", thrdCudaDev, blockIdx.x, threadIdx.x);

  #ifdef ARRAY_DEBUG
    if (tid == 0) {
      *(blkStatus.barrierCnt + 0 + 5 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    }
  #endif

  ofcclBarrier(5);

  int unprogressedCnt = 0;
  while (true) {

    while (true) {
      if (blkStatus.dynamicBlkStatus.numActiveColls == 0) {
        break;
      }
      traverseTaskQ(thrdCudaDev, globalBlk2CollId2CollCtx, collCount, cq, globalCqes, &unprogressedCnt, BASE_CTX_SWITCH_THRESHOLD);

      if (tid == 0) { // 遍历完一次之后，当前activeColl的后续工作，
        // 只有完成一个集合通信，才有必要操作taskQ
        int new_numActiveColls = 0;
        int numStuckColls = 0;
        for (int i = 0; i < blkStatus.dynamicBlkStatus.numActiveColls; ++i) {
          int collIdInTaskQ = blkStatus.activeCollIdsAlign.activeCollIds[i];
          // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, blkStatus.collStatusAlign.collStatus is %d", thrdCudaDev, blockIdx.x, threadIdx.x, collIdInTaskQ, blkStatus.collStatusAlign.collStatus[collIdInTaskQ]);
          if (blkStatus.collStatusAlign.collStatus[collIdInTaskQ] < 0) { // 不应该有1 的存在了，只有-1, -2或者2
            blkStatus.activeCollIdsAlign.activeCollIds[new_numActiveColls++] = collIdInTaskQ; // 小于0，就要继续放在taskQ里

            if (blkStatus.collTryCntAllign.collTryCnt[collIdInTaskQ] >= getTryNum(i)) { // TODO: 做一个新的函数判断stuck，现在这样循环一次任务列表肯定就算stuck了。 // 不过目前不太好完全解耦。
              ++numStuckColls;
            }

          }

          // 全部重置？还是只重置完成的？
          // blkStatus.collTryCntAllign.collTryCnt[collIdInTaskQ] = 0;
        }
        blkStatus.dynamicBlkStatus.numActiveColls = new_numActiveColls;
        if (numStuckColls == new_numActiveColls) {
          blkStatus.willingnessToGetSqe = 1; // TODO: 目前不太好完全解耦。
        }
        #ifdef CQE_DEBUG_ALL_RANK
          logTaskQ(1, thrdCudaDev, -1);
        #elif defined(CQE_DEBUG_RANK_X)
          logTaskQ(1, thrdCudaDev, CQE_DEBUG_RANK_X);
        #endif
      }
      // *(blkStatus.barrierCnt + 0 + 18 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      ofcclBarrier(10);
      // *(blkStatus.barrierCnt + 1 + 18 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

      if (blkStatus.willingnessToGetSqe == 1) {
        // 在blkStatus里通过处理任务列表，设置这个标记位，barrier之后，所有线程根据这个标记位决定是否退出遍历任务列表。
        break;
      }
    }

    if (tid == 0) {
      blkStatus.willingnessToGetSqe = 0;
    
      checkSQ7TidyTaskQ(thrdCudaDev, sq, globalBlk2CollId2CollCtx, finallyQuit, &unprogressedCnt);

      // 只有0号线程才会执行checkSQ7TidyTaskQ，自然只有0号线程才会更改checkSQ7TidyTaskQFailCnt，并且进行相应调整。

      if (unprogressedCnt >= TOLERANT_UNPROGRESSED_CNT && blkStatus.quit != 1) {
        BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;

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
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 35 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.totalSwitchCntAfterRecvSuccess;
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 36 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.numActiveColls;
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 37 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = unprogressedCnt;
        *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 38 * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt;
      }
    #endif

    // 记录数组的前10项，未必都是有效的。所有线程都做，看到的应该是一样的。
    // for (int i = 0; i < PrintTestQNum; i++) {
    //   *(blkStatus.barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + (38 + i) * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = blkStatus.activeCollIdsAlign.activeCollIds[i];
    // }

    if (blkStatus.quit == 1) {

      if (blkStatus.finallyQuit == 1) { // TODO: 还是不要在这里读host mem
        #ifdef SHOW_CNT
          // OFCCL_LOG_THRD_0(OFCCL_FINAL_QUIT, "Rank<%d> Blk<%d> Thrd<%d> totalCtxSaveCnt=%llu (avg totalCtxSaveCnt: %llu), totalCtxLoadCnt=%llu (avg totalCtxLoadCnt: %llu), totalSwitchCntAfterRecvSuccess=%llu (avg totalSwitchCntAfterRecvSuccess: %llu), totalSwitchCntBeforeRecvSuccess=%llu (avg totalSwitchCntBeforeRecvSuccess: %llu), totalUnprogressedQuitCnt=%llu (avg totalUnprogressedQuitCnt: %llu)", thrdCudaDev, bid, tid, blkStatus.dynamicBlkStatus.totalCtxSaveCnt, blkStatus.dynamicBlkStatus.totalCtxSaveCnt / NUM_ITER_ENV, blkStatus.dynamicBlkStatus.totalCtxLoadCnt, blkStatus.dynamicBlkStatus.totalCtxLoadCnt / NUM_ITER_ENV, blkStatus.dynamicBlkStatus.totalSwitchCntAfterRecvSuccess, blkStatus.dynamicBlkStatus.totalSwitchCntAfterRecvSuccess / NUM_ITER_ENV, blkStatus.dynamicBlkStatus.totalSwitchCntBeforeRecvSuccess, blkStatus.dynamicBlkStatus.totalSwitchCntBeforeRecvSuccess / NUM_ITER_ENV, blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt, blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt / NUM_ITER_ENV);

          OFCCL_LOG_THRD_0(OFCCL_FINAL_QUIT, "Rank<%d> Blk<%d> Thrd<%d> totalCtxSaveCnt=%llu, totalCtxLoadCnt=%llu, totalSwitchCntAfterRecvSuccess=%llu, totalSwitchCntBeforeRecvSuccess=%llu, totalUnprogressedQuitCnt=%llu", thrdCudaDev, bid, tid, blkStatus.dynamicBlkStatus.totalCtxSaveCnt, blkStatus.dynamicBlkStatus.totalCtxLoadCnt, blkStatus.dynamicBlkStatus.totalSwitchCntAfterRecvSuccess, blkStatus.dynamicBlkStatus.totalSwitchCntBeforeRecvSuccess, blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt);
        #endif

        #ifdef DEBUG_CLOCK
          if (tid == 0) {
            #ifdef DEBUG_CLOCK_TRAIN
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              for (int i = 0; i < collCount; ++i) {
                int collId = globalCollIds[i];
                if (bid < sharedBlkCount4CollAlign.blkCount4Coll[collId]) {
                  long long int totalDeltaClock = 0;
                  for (int j = 0; j < RECORD_ITER; ++j) {
                    totalDeltaClock += blkStatus.beforeAfterGetSqeDeltaClock[collId][j];
                  }
                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, before get sqe clock = %lld\t%lld\t%lld\t%lld", thrdCudaDev, bid, tid, collId, blkStatus.beforeGetSqeClock[collId][0], blkStatus.beforeGetSqeClock[collId][1], blkStatus.beforeGetSqeClock[collId][2], blkStatus.beforeGetSqeClock[collId][3]);
                  
                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, after get sqe clock = %lld\t%lld\t%lld\t%lld", thrdCudaDev, bid, tid, collId, blkStatus.getSqeClock[collId][0], blkStatus.getSqeClock[collId][1], blkStatus.getSqeClock[collId][2], blkStatus.getSqeClock[collId][3]);
                  
                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, before put cqe clock = %lld\t%lld\t%lld\t%lld", thrdCudaDev, bid, tid, collId, blkStatus.beforePutCqeClock[collId][0], blkStatus.beforePutCqeClock[collId][1], blkStatus.beforePutCqeClock[collId][2], blkStatus.beforePutCqeClock[collId][3]);
                  
                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, after put cqe clock = %lld\t%lld\t%lld\t%lld", thrdCudaDev, bid, tid, collId, blkStatus.putCqeClock[collId][0], blkStatus.putCqeClock[collId][1], blkStatus.putCqeClock[collId][2], blkStatus.putCqeClock[collId][3]);

                  // OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, before after get sqe = %.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, collId, blkStatus.beforeAfterGetSqeDeltaClock[collId][0]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[collId][1]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[collId][2]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[collId][3]/CLOCK2US_FACTOR);
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, before after get sqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, collId, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);
                }
              }

              int putCqeCnt;
              int putCqeCnt_adjust;

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              for (int i = 0; i < collCount; ++i) {
                int collId = globalCollIds[i];
                if (bid < sharedBlkCount4CollAlign.blkCount4Coll[collId]) {
                  putCqeCnt = 0;
                  long long int totalDeltaClock = 0;
                  for (int j = 0; j < RECORD_ITER; ++j) {
                    totalDeltaClock += blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][j];
                    if (blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][j] > 0.0) {
                      putCqeCnt++;
                    }
                  }
                  putCqeCnt_adjust = (putCqeCnt == 0) ? 1 : putCqeCnt; // 防止除0的bug。
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, AfterSqe TO BeforeCqe = %.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, collId, blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][0]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][1]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][2]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][3]/CLOCK2US_FACTOR);
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, AfterSqe TO BeforeCqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, collId, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);
                }
              }
              
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              for (int i = 0; i < collCount; ++i) {
                int collId = globalCollIds[i];
                if (bid < sharedBlkCount4CollAlign.blkCount4Coll[collId]) {
                  long long int totalDeltaClock = 0;
                  for (int j = 0; j < RECORD_ITER; ++j) {
                    totalDeltaClock += blkStatus.beforeAfterPutCqeDeltaClock[collId][j];
                  }
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, before after put cqe = %.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, collId, blkStatus.beforeAfterPutCqeDeltaClock[collId][0]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[collId][1]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[collId][2]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[collId][3]/CLOCK2US_FACTOR);
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, before after put cqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, collId, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);
                }
              }
              
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              for (int i = 0; i < collCount; ++i) {
                int collId = globalCollIds[i];
                if (bid < sharedBlkCount4CollAlign.blkCount4Coll[collId]) {
                  long long int totalDeltaClock = 0;
                  for (int j = 0; j < RECORD_ITER; ++j) {
                    totalDeltaClock += blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][j];
                  }
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, beforeSqe TO afterCqe = %.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, collId, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][0]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][1]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][2]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][3]/CLOCK2US_FACTOR);
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, beforeSqe TO afterCqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, collId, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);
                }
              }

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              for (int i = 0; i < collCount; ++i) {
                int collId = globalCollIds[i];
                if (bid < sharedBlkCount4CollAlign.blkCount4Coll[collId]) {
                  OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, ctxSwitchCnt = %d", thrdCudaDev, bid, tid, collId, blkStatus.ctxSwitchCnt[i]);
                }
              }

            #endif

            #ifdef DEBUG_CLOCK_3D
              if (thrdCudaDev == 1) {
                OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>, totalSqeCnt = %d, totalCqeCnt = %d", thrdCudaDev, bid, tid, blkStatus.totalSqeCnt, blkStatus.totalCqeCnt);
              }
            #endif

            #ifdef DEBUG_CLOCK_CTX
            long long int totalDeltaClock = 0;
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>, loadIter=%d", thrdCudaDev, bid, tid, blkStatus.loadIter);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterLoadDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, LoadDeltaClock = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterLoadDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterLoadDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterLoadDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterLoadDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterLoadDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, LoadDeltaClock AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterSaveDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, SaveDeltaClock = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterSaveDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterSaveDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterSaveDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterSaveDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterSaveDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, SaveDeltaClock AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);
              
            #endif

            #ifdef DEBUG_CLOCK_IO
              long long int totalDeltaClock = 0;
              int commitSqeCnt;
              int commitSqeCnt_adjust;

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterReadSqEmptyDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, ReadSqEmptyDelta = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterReadSqEmptyDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, ReadSqEmptyDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterGetSqFrontierDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, GetSqFrontierDelta = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterGetSqFrontierDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, GetSqFrontierDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterAddSqFrontierCounterDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, AddSqFrontierCounterDelta = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterAddSqFrontierCounterDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, AddSqFrontierCounterDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              commitSqeCnt = 0;
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterUpdateSqHeadDeltaClock[j];
                if (blkStatus.afterUpdateSqHeadDeltaClock[j] > 0.0) {
                  commitSqeCnt++;
                }
              }
              commitSqeCnt_adjust = (commitSqeCnt == 0) ? 1 : commitSqeCnt; // 防止除0的bug。
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, UpdateSqHeadDelta = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterUpdateSqHeadDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, UpdateSqHeadDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/commitSqeCnt_adjust/CLOCK2US_FACTOR, commitSqeCnt);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.beforeAfterGetSqeDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after get sqe = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.beforeAfterGetSqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after get sqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterRecordBuffDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, RecordBuffDelta = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterRecordBuffDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, RecordBuffDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, afterGetSqeBeforeMaintainSharedCtxDeltaClock = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, afterGetSqeBeforeMaintainSharedCtxDeltaClock AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, afterGetSqeBeforeOfcclFuncDeltaClock = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, afterGetSqeBeforeOfcclFuncDeltaClock AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              int putCqeCnt;
              int putCqeCnt_adjust;

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              putCqeCnt = 0;
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterGetSqeBeforePutCqeDeltaClock[j];
                if (blkStatus.afterGetSqeBeforePutCqeDeltaClock[j] > 0.0) {
                  putCqeCnt++;
                }
              }
              putCqeCnt_adjust = (putCqeCnt == 0) ? 1 : putCqeCnt; // 防止除0的bug。
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, AfterSqe TO BeforeCqe = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.afterGetSqeBeforePutCqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, AfterSqe TO BeforeCqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);
              
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.beforeAfterPutCqeDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after put cqe = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.beforeAfterPutCqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after put cqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);
              
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.beforeGetSqeAfterPutCqeDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, beforeSqe TO afterCqe = %.2lf, %.2lf, %.2lf, %.2lf, %.2lf", thrdCudaDev, bid, tid, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, beforeSqe TO afterCqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, sqReadCnt = %d, cqWriteCnt = %d", thrdCudaDev, bid, tid, blkStatus.sqReadCnt, blkStatus.cqWriteCnt);
            #endif
          }
        #endif

      } else {
        int acTotalBytes = roundUp(blkStatus.dynamicBlkStatus.numActiveColls * SHORT_ELEM_SIZE, COPY_ELEM_SIZE);
        BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;
        copy16BLoop(tid, myGlobalBlkStatus->activeCollIdsAlign.activeCollIds, blkStatus.activeCollIdsAlign.activeCollIds, acTotalBytes);
        // int acDoneBytes = 0;
        // int nthreads = blockDim.x;
        // while (acDoneBytes < acTotalBytes) {
        //   int targetBytes = min(nthreads * COPY_ELEM_SIZE, acTotalBytes - acDoneBytes);
        //   copy16B(tid, (char *)(&myGlobalBlkStatus->activeCollIdsAlign.activeCollIds) + acDoneBytes, (char *)(blkStatus.activeCollIdsAlign.activeCollIds) + acDoneBytes, targetBytes);
        //   acDoneBytes += targetBytes;
        // }
        copy16BLoop(tid, &myGlobalBlkStatus->dynamicBlkStatus, &blkStatus.dynamicBlkStatus, sizeof(DynamicBlkStatus));

        #ifdef DEBUG_CLOCK
          // 可以并行优化，看看有没有必要吧，每次循环的增量是blockDim.x
          ONE_THRD_DO
            #ifdef DEBUG_CLOCK_TRAIN
              for (int i = 0; i < collCount; ++i) {
                int collId = globalCollIds[i];
                for (int j = 0; j < RECORD_ITER; ++j) {
                  myGlobalBlkStatus->beforeGetSqeClock[collId][j] = blkStatus.beforeGetSqeClock[collId][j];
                  myGlobalBlkStatus->getSqeClock[collId][j] = blkStatus.getSqeClock[collId][j];
                  myGlobalBlkStatus->beforePutCqeClock[collId][j] = blkStatus.beforePutCqeClock[collId][j];
                  myGlobalBlkStatus->putCqeClock[collId][j] = blkStatus.putCqeClock[collId][j];
                  
                  myGlobalBlkStatus->beforeAfterGetSqeDeltaClock[collId][j] = blkStatus.beforeAfterGetSqeDeltaClock[collId][j];
                  myGlobalBlkStatus->afterGetSqeBeforePutCqeDeltaClock[collId][j] = blkStatus.afterGetSqeBeforePutCqeDeltaClock[collId][j];
                  myGlobalBlkStatus->beforeAfterPutCqeDeltaClock[collId][j] = blkStatus.beforeAfterPutCqeDeltaClock[collId][j];
                  myGlobalBlkStatus->beforeGetSqeAfterPutCqeDeltaClock[collId][j] = blkStatus.beforeGetSqeAfterPutCqeDeltaClock[collId][j];
                }
                myGlobalBlkStatus->beforeGetSqeIter[collId] = blkStatus.beforeGetSqeIter[collId];
                myGlobalBlkStatus->getSqeIter[collId] = blkStatus.getSqeIter[collId];
                // myGlobalBlkStatus->beforePutCqeIter[collId] = blkStatus.beforePutCqeIter[collId];
                // myGlobalBlkStatus->putCqeIter[collId] = blkStatus.putCqeIter[collId];
                
                myGlobalBlkStatus->ctxSwitchCnt[collId] = blkStatus.ctxSwitchCnt[collId];
              }
            #endif
            #ifdef DEBUG_CLOCK_3D
              for (int i = 0; i < collCount; ++i) {
                int collId = globalCollIds[i];
                myGlobalBlkStatus->switchCntAfterRecvSuccess[collId] = blkStatus.switchCntAfterRecvSuccess[collId];
                myGlobalBlkStatus->switchCntBeforeRecvSuccess[collId] = blkStatus.switchCntBeforeRecvSuccess[collId];
                myGlobalBlkStatus->switchCntAfterRecvSuccessIterDelta[collId] = blkStatus.switchCntAfterRecvSuccessIterDelta[collId];
                myGlobalBlkStatus->switchCntBeforeRecvSuccessIterDelta[collId] = blkStatus.switchCntBeforeRecvSuccessIterDelta[collId];
              
                myGlobalBlkStatus->collIdInSqe[i] = blkStatus.collIdInSqe[i];
                myGlobalBlkStatus->taskQLenAfterGetSqe[i] = blkStatus.taskQLenAfterGetSqe[i];
                myGlobalBlkStatus->collId4Cq[i] = blkStatus.collId4Cq[i];
              }
              myGlobalBlkStatus->iterCqeCnt = blkStatus.iterCqeCnt;
              myGlobalBlkStatus->iterSqeCnt = blkStatus.iterSqeCnt;
              myGlobalBlkStatus->iterNum = blkStatus.iterNum;
              myGlobalBlkStatus->iterSqNum = blkStatus.iterSqNum;
              myGlobalBlkStatus->totalSqeCnt = blkStatus.totalSqeCnt;
              myGlobalBlkStatus->totalCqeCnt = blkStatus.totalCqeCnt;
            #endif
            #ifdef DEBUG_CLOCK_IO
              for (int j = 0; j < RECORD_ITER; ++j) {
                myGlobalBlkStatus->beforeGetSqeClock[j] = blkStatus.beforeGetSqeClock[j];
                myGlobalBlkStatus->getSqeClock[j] = blkStatus.getSqeClock[j];
                myGlobalBlkStatus->beforePutCqeClock[j] = blkStatus.beforePutCqeClock[j];
                myGlobalBlkStatus->putCqeClock[j] = blkStatus.putCqeClock[j];
                
                myGlobalBlkStatus->beforeAfterGetSqeDeltaClock[j] = blkStatus.beforeAfterGetSqeDeltaClock[j];
                myGlobalBlkStatus->afterGetSqeBeforePutCqeDeltaClock[j] = blkStatus.afterGetSqeBeforePutCqeDeltaClock[j];
                myGlobalBlkStatus->beforeAfterPutCqeDeltaClock[j] = blkStatus.beforeAfterPutCqeDeltaClock[j];
                myGlobalBlkStatus->beforeGetSqeAfterPutCqeDeltaClock[j] = blkStatus.beforeGetSqeAfterPutCqeDeltaClock[j];

                myGlobalBlkStatus->afterReadSqEmptyDeltaClock[j] = blkStatus.afterReadSqEmptyDeltaClock[j];
                myGlobalBlkStatus->afterGetSqFrontierDeltaClock[j] = blkStatus.afterGetSqFrontierDeltaClock[j];
                myGlobalBlkStatus->afterAddSqFrontierCounterDeltaClock[j] = blkStatus.afterAddSqFrontierCounterDeltaClock[j];
                myGlobalBlkStatus->afterUpdateSqHeadDeltaClock[j] = blkStatus.afterUpdateSqHeadDeltaClock[j];
                myGlobalBlkStatus->afterRecordBuffDeltaClock[j] = blkStatus.afterRecordBuffDeltaClock[j];

                myGlobalBlkStatus->beforeOfcclFuncClock[j] = blkStatus.beforeOfcclFuncClock[j];
                myGlobalBlkStatus->afterGetSqeBeforeOfcclFuncDeltaClock[j] = blkStatus.afterGetSqeBeforeOfcclFuncDeltaClock[j];
                myGlobalBlkStatus->afterGetSqeBeforeMaintainSharedCtxDeltaClock[j] = blkStatus.afterGetSqeBeforeMaintainSharedCtxDeltaClock[j];
              }
              myGlobalBlkStatus->beforeGetSqeIter = blkStatus.beforeGetSqeIter;
              myGlobalBlkStatus->getSqeIter = blkStatus.getSqeIter;

              myGlobalBlkStatus->sqReadCnt = blkStatus.sqReadCnt;
              myGlobalBlkStatus->cqWriteCnt = blkStatus.cqWriteCnt;
            #endif

            #ifdef DEBUG_CLOCK_CTX
              for (int j = 0; j < RECORD_ITER; ++j) {
                myGlobalBlkStatus->beforeLoadClock[j] = blkStatus.beforeLoadClock[j];
                myGlobalBlkStatus->afterLoadDeltaClock[j] = blkStatus.afterLoadDeltaClock[j];
                myGlobalBlkStatus->beforeSaveClock[j] = blkStatus.beforeSaveClock[j];
                myGlobalBlkStatus->afterSaveDeltaClock[j] = blkStatus.afterSaveDeltaClock[j];
              }
              myGlobalBlkStatus->loadIter = blkStatus.loadIter;
              myGlobalBlkStatus->saveIter = blkStatus.saveIter;
            #endif
          ONE_THRD_DO_END
        #endif
      }

      // OFCCL_LOG_THRD_0(OFCCL_RESNET, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel quits, blkStatus.dynamicBlkStatus.numActiveColls=%d", thrdCudaDev, blockIdx.x, tid, blkStatus.dynamicBlkStatus.numActiveColls);
      #ifdef ARRAY_DEBUG
        if (tid == 0) {
          *(blkStatus.barrierCnt + 1 + 5 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        }
      #endif
      return;
    }
  }
}