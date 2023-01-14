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
__shared__ CollCtx sharedCollCtx[NUM_SHMEM_SLOT]; // 不能static，primitives要用

__shared__ BlkStatus blkStatus; // 取消static，放到prim里边打印log。

// static __shared__ IdsAlign sharedIdsAlign;
static __shared__ BlkCount4CollAlign sharedBlkCount4CollAlign;
static __shared__ unsigned long long int zeros[2];
static __shared__ int cqWriteSlot;

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
    // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, after CAS oldSlot = 0x%llx, cqCnt = %u, doneCollId = %d", thrdCudaDev, blockIdx.x, threadIdx.x, oldSlot, blkStatus.dynamicBlkStatus.cqCnt[doneCollId], doneCollId);

    if (oldSlot == INVALID_CQ_SLOT_MASK) {
      ++blkStatus.dynamicBlkStatus.cqCnt[doneCollId]; // 写成功才更新。
      return 0;
    }
  }

  // 纯粹的单坑：

  return -1;
}

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
  
  // 不需要初始化DEBUG_CLOCK里的数组，因为这些数组使用的时候都是直接赋值的。

  BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;
  int hasQuitted = myGlobalBlkStatus->hasQuitted; // 每个线程都读。

  // 第一次启动之前，rankCtx->hostBlkStatus是calloc的，然后复制到globalMem上，所以blkStatus.collStatusAlign.collStatus应该是全0，但是之后的启动可能导致collStatus数组是混乱的，还是重置一下。
  int csTotalBytes = roundUp(MAX_LENGTH * CHAR_ELEM_SIZE, COPY_ELEM_SIZE);
  int csDoneBytes = 0;
  while (csDoneBytes < csTotalBytes) {
    int targetBytes = min(nthreads * COPY_ELEM_SIZE, csTotalBytes - csDoneBytes);
    set16B(tid, (char *)(blkStatus.collStatusAlign.collStatus) + csDoneBytes, &zeros, targetBytes);
    csDoneBytes += targetBytes;
  }

  if (hasQuitted == 0) {
    set16B(tid, &blkStatus.dynamicBlkStatus, &zeros, sizeof(DynamicBlkStatus));
      
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
          }
          blkStatus.sqReadCnt = 0;
          blkStatus.cqWriteCnt = 0;
        #endif
      ONE_THRD_DO_END
    #endif

  } else {
    copy16B(tid, &blkStatus.dynamicBlkStatus, &myGlobalBlkStatus->dynamicBlkStatus, sizeof(DynamicBlkStatus));
      
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
            }
            blkStatus.beforeGetSqeIter = myGlobalBlkStatus->beforeGetSqeIter;
            blkStatus.getSqeIter = myGlobalBlkStatus->getSqeIter;

            blkStatus.sqReadCnt = myGlobalBlkStatus->sqReadCnt;
            blkStatus.cqWriteCnt = myGlobalBlkStatus->cqWriteCnt;
          #endif
        ONE_THRD_DO_END
    #endif
  }

  int bcTotalBytes = roundUp(MAX_LENGTH * CHAR_ELEM_SIZE, COPY_ELEM_SIZE); // 这里不应该用collCount，因为blkCount4Coll相当于是数组模拟的map，我们不应该假设coll_id连续增长。
  int bcDoneBytes = 0;
  while (bcDoneBytes < bcTotalBytes) {
    int targetBytes = min(nthreads * COPY_ELEM_SIZE, bcTotalBytes - bcDoneBytes);
    copy16B(tid, (char *)(sharedBlkCount4CollAlign.blkCount4Coll) + bcDoneBytes, (char *)globalBlkCount4Coll + bcDoneBytes, targetBytes);
    bcDoneBytes += targetBytes;
  }

  ofcclBarrier(1); // 为了下边读取blkStatus.dynamicBlkStatus.numActiveColls

  int acTotalBytes = roundUp(blkStatus.dynamicBlkStatus.numActiveColls * SHORT_ELEM_SIZE, COPY_ELEM_SIZE);
  int acDoneBytes = 0;
  // 这个要不要复制，需要读取numActiveColls，所以必须得上边做完，加一个barrier之后才可以。
  while (acDoneBytes < acTotalBytes) {
    int targetBytes = min(nthreads * COPY_ELEM_SIZE, acTotalBytes - acDoneBytes);
    copy16B(tid, (char *)(blkStatus.activeCollIdsAlign.activeCollIds) + acDoneBytes, (char *)(&myGlobalBlkStatus->activeCollIdsAlign.activeCollIds) + acDoneBytes, targetBytes);
    acDoneBytes += targetBytes;
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
    // OFCCL_LOG_RANK_X(OFCCL, 0, "Rank<%d> Blk<%d> Thrd<%d>, read SQE for coll_id = %d, reset *unprogressedCnt = 0", thrdCudaDev, blockIdx.x, threadIdx.x, newActiveCollId);

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

static __device__ int loadCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId, int turn, int64_t BASE_CTX_SWITCH_THRESHOLD) {
  int tid = threadIdx.x;

  // TODO: 考虑让所有线程都执行常数初始化。
  ONE_THRD_DO
    #ifdef SHOW_CNT
      blkStatus.dynamicBlkStatus.totalCtxLoadCnt++;
    #endif

    sharedCollCtx[collId % NUM_SHMEM_SLOT].progressed = 0;
    sharedCollCtx[collId % NUM_SHMEM_SLOT].ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD;
  ONE_THRD_DO_END

  copy16B(tid, &sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx, &globalCollCtx4Blk7Coll->dynamicCollCtx, sizeof(DynamicCollCtx));
  copy16B(tid, &sharedCollCtx[collId % NUM_SHMEM_SLOT].staticCollCtx, &globalCollCtx4Blk7Coll->staticCollCtx, sizeof(StaticCollCtx));

  return turn;
}

#ifdef DEBUG_PARA_SV
static __device__ void saveExcutingCollCtx(int thrdCudaDev, CollCtx *globalCollCtx4Blk7Coll, int collId) {
  int tid = threadIdx.x;
  #ifdef SHOW_CNT
    ONE_THRD_DO
      blkStatus.dynamicBlkStatus.totalCtxSaveCnt++;
    ONE_THRD_DO_END
  #endif
  copy16B(tid, &globalCollCtx4Blk7Coll->dynamicCollCtx, &sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx, sizeof(DynamicCollCtx));
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

static __device__ int maintainSharedCollCtx(int thrdCudaDev, CollCtx *globalBlk2CollId2CollCtx, int collId, int turn, int64_t BASE_CTX_SWITCH_THRESHOLD, int64_t BOUNS_SWITCH_4_PROCESSED_COLL, int *unprogressedCnt) {
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

    saveExcutingCollCtx(thrdCudaDev, globalCollCtx4Blk7OldColl, collIdOfThatSlot);

    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> save ctx for coll_id = %d, slice4SimpleGenericOp=%d, offset4SimpleGenericOp=%d, currentStep4RunRing=%d, gridOffset4RunRing=%ld", thrdCudaDev, blockIdx.x, threadIdx.x, collIdOfThatSlot, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.slice4SimpleGenericOp, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.offset4SimpleGenericOp, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.currentStep4RunRing, sharedCollCtx[collIdOfThatSlot % NUM_SHMEM_SLOT].dynamicCollCtx.gridOffset4RunRing);
  }

  if (needLoad) {
    CollCtx *globalCollCtx4Blk7Coll = globalBlk2CollId2CollCtx + bid * MAX_LENGTH + collId;
    turn = loadCollCtx(thrdCudaDev, globalCollCtx4Blk7Coll, collId, turn, BASE_CTX_SWITCH_THRESHOLD);

    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> load ctx for coll_id = %d, loadAgain=%d, slice4SimpleGenericOp=%d, offset4SimpleGenericOp=%d, currentStep4RunRing=%d, gridOffset4RunRing=%ld", thrdCudaDev, blockIdx.x, threadIdx.x, collId, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.loadAgain, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.slice4SimpleGenericOp, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.offset4SimpleGenericOp, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.currentStep4RunRing, sharedCollCtx[collId % NUM_SHMEM_SLOT].dynamicCollCtx.gridOffset4RunRing);
  }

  if (tid == 0) {
    blkStatus.currLoadedCollId = collId; // 这个变量只起一个传递信息的作用了，不再标记shmem是否valid
    if (blkStatus.collStatusAlign.collStatus[collId] == -1) {
      // bugfix: 防止一个不需要load的coll，无休止增加下去。
      sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD + BOUNS_SWITCH_4_PROCESSED_COLL;
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> coll_id = %d, blkStatus.collStatusAlign.collStatus is %d, sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].ctxSwitchThreshold = %ld", thrdCudaDev, blockIdx.x, threadIdx.x, collId, blkStatus.collStatusAlign.collStatus[collId], sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].ctxSwitchThreshold);
      *unprogressedCnt = 0; // 这表明有coll前进了，只不过没跑完。
      
      #ifdef SHOW_CNT
        blkStatus.dynamicBlkStatus.totalProgressed7SwithchCnt++;
      #endif
    } else if (blkStatus.collStatusAlign.collStatus[collId] == -2) {
      sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].ctxSwitchThreshold = BASE_CTX_SWITCH_THRESHOLD;
      *unprogressedCnt += 1;
    }

    #ifdef DEBUG_CLOCK_TRAIN
      if (blkStatus.collStatusAlign.collStatus[collId] < 0) {
        ++blkStatus.ctxSwitchCnt[collId];
      }
    #endif

    blkStatus.collStatusAlign.collStatus[collId] = 1; // 每次准备执行的时候，重置为正常执行状态。新的coll已经是1，不过不要浪费if了。 
    sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].saveCtx7Quit = 0; // 重置。
  }

  ofcclBarrier(4);
  return turn;
}

static __device__ void manipulateCQ7ResetDoneColl(int thrdCudaDev, int doneCollId, CQ *cq, CQE *globalCqes, CollCtx *globalCollCtx4Blk7Coll, CollCtx *globalBlk2CollId2CollCtx) {
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

  // 多个slot之后这条失效了，不过可以重置一下。bugfix: manipulateCQ7ResetDoneColl的调用时机，是traverseTaskQ外边，又遍历一次taskQ，所以要判断一下。这也是可以带来性能优化的地方。不判断会导致另一个coll从上次save的地方重新load，重新做已经完成的搬运，但是peer rank未必能配合，就会卡住。
  if (doneCollId == blkStatus.currLoadedCollId) {
    blkStatus.currLoadedCollId = -1;
  }
  // bugfix: 启用了slot之后，对于完成的coll，要把其占用的slot给invalid掉
  // bugfix: 但是也得判断那个slot装的是不是自己。
  if (sharedCollCtx[doneCollId % NUM_SHMEM_SLOT].staticCollCtx.collId == doneCollId) {
    sharedCollCtx[doneCollId % NUM_SHMEM_SLOT].staticCollCtx.collId = -1;
  }

  blkStatus.collStatusAlign.collStatus[doneCollId] = 0;

  // ResetDoneColl
  globalCollCtx4Blk7Coll->dynamicCollCtx.loadAgain = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.slice4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.offset4SimpleGenericOp = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.currentStep4RunRing = 0;
  globalCollCtx4Blk7Coll->dynamicCollCtx.gridOffset4RunRing = 0;
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
      if (wid < sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.workElem.header.nWarps) {
        ofcclFuncs[sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.workElem.header.funcIndex](); // 这里边的调用里不涉及__syncthreads().
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

// TODO: 考虑在按需启停的场景下，会多次启动，执行上会不会有什么变化。
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
          OFCCL_LOG_THRD_0(OFCCL_FINAL_QUIT, "Rank<%d> Blk<%d> Thrd<%d> totalCtxSaveCnt=%llu, totalCtxLoadCnt=%llu, totalProgressed7SwithchCnt=%llu, totalUnprogressedQuitCnt=%llu", thrdCudaDev, bid, tid, blkStatus.dynamicBlkStatus.totalCtxSaveCnt, blkStatus.dynamicBlkStatus.totalCtxLoadCnt, blkStatus.dynamicBlkStatus.totalProgressed7SwithchCnt, blkStatus.dynamicBlkStatus.totalUnprogressedQuitCnt);
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

            #ifdef DEBUG_CLOCK_IO
              long long int totalDeltaClock = 0;
              int commitSqeCnt;
              int commitSqeCnt_adjust;

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterReadSqEmptyDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, ReadSqEmptyDelta = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterReadSqEmptyDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterReadSqEmptyDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, ReadSqEmptyDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterGetSqFrontierDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, GetSqFrontierDelta = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterGetSqFrontierDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterGetSqFrontierDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, GetSqFrontierDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterAddSqFrontierCounterDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, AddSqFrontierCounterDelta = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterAddSqFrontierCounterDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterAddSqFrontierCounterDeltaClock[4]/CLOCK2US_FACTOR);
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
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, UpdateSqHeadDelta = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterUpdateSqHeadDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterUpdateSqHeadDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, UpdateSqHeadDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/commitSqeCnt_adjust/CLOCK2US_FACTOR, commitSqeCnt);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.beforeAfterGetSqeDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after get sqe = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.beforeAfterGetSqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.beforeAfterGetSqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after get sqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.afterRecordBuffDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, RecordBuffDelta = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterRecordBuffDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterRecordBuffDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, RecordBuffDelta AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/RECORD_ITER/CLOCK2US_FACTOR, RECORD_ITER);


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
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, AfterSqe TO BeforeCqe = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.afterGetSqeBeforePutCqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.afterGetSqeBeforePutCqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, AfterSqe TO BeforeCqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);
              
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.beforeAfterPutCqeDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after put cqe = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.beforeAfterPutCqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.beforeAfterPutCqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, before after put cqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);
              
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              totalDeltaClock = 0;
              for (int j = 0; j < RECORD_ITER; ++j) {
                totalDeltaClock += blkStatus.beforeGetSqeAfterPutCqeDeltaClock[j];
              }
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, beforeSqe TO afterCqe = %.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf", thrdCudaDev, bid, tid, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[0]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[1]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[2]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[3]/CLOCK2US_FACTOR, blkStatus.beforeGetSqeAfterPutCqeDeltaClock[4]/CLOCK2US_FACTOR);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, beforeSqe TO afterCqe AVG = %.2lf us, weight = %d", thrdCudaDev, bid, tid, totalDeltaClock/putCqeCnt_adjust/CLOCK2US_FACTOR, putCqeCnt);

              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d>", thrdCudaDev, bid, tid);
              OFCCL_LOG_RANK_0(OFCCL_DEBUG_TIME, "Rank<%d> Blk<%d> Thrd<%d> coll_id = 0, sqReadCnt = %d, cqWriteCnt = %d", thrdCudaDev, bid, tid, blkStatus.sqReadCnt, blkStatus.cqWriteCnt);
            #endif
          }
        #endif

      } else {
        int acTotalBytes = roundUp(blkStatus.dynamicBlkStatus.numActiveColls * SHORT_ELEM_SIZE, COPY_ELEM_SIZE);
        int acDoneBytes = 0;
        BlkStatus *myGlobalBlkStatus = globalBlkStatus + bid;
        int nthreads = blockDim.x;
        while (acDoneBytes < acTotalBytes) {
          int targetBytes = min(nthreads * COPY_ELEM_SIZE, acTotalBytes - acDoneBytes);
          copy16B(tid, (char *)(&myGlobalBlkStatus->activeCollIdsAlign.activeCollIds) + acDoneBytes, (char *)(blkStatus.activeCollIdsAlign.activeCollIds) + acDoneBytes, targetBytes);
          acDoneBytes += targetBytes;
        }
        copy16B(tid, &myGlobalBlkStatus->dynamicBlkStatus, &blkStatus.dynamicBlkStatus, sizeof(DynamicBlkStatus));

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
              }
              myGlobalBlkStatus->beforeGetSqeIter = blkStatus.beforeGetSqeIter;
              myGlobalBlkStatus->getSqeIter = blkStatus.getSqeIter;

              myGlobalBlkStatus->sqReadCnt = blkStatus.sqReadCnt;
              myGlobalBlkStatus->cqWriteCnt = blkStatus.cqWriteCnt;
            #endif
          ONE_THRD_DO_END
        #endif
      }

      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, daemonKernel quits", thrdCudaDev, blockIdx.x, tid);
      #ifdef ARRAY_DEBUG
        if (tid == 0) {
          *(blkStatus.barrierCnt + 1 + 5 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        }
      #endif
      return;
    }
  }
}