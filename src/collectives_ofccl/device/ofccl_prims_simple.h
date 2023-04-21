#include "collectives_ofccl.h"
#include "common_ofccl.h"
#include "debug.h"
#include <cstdint>

//                                       NCCL_STEPS(8)/2=4     NCCL_STEPS/4=2         2
// ringAllreduce里边，Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
// 对应                       ProtoSimple<          SlicePerChunk,                    StepPerSlice, Unroll>
//                                                SlicePerChunk=2                     StepPerSlice=2
// Unroll 在primitives.h 里有默认值 COLL_UNROLL=4
// P2p=0
template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int P2p>
class Primitives<
    T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll>, P2p
  > {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       OffsFifoEnabled = 0x80,
                       SizesFifoEnabled = 0x100,
                       DirectWrite = 0x200,
                       DirectRead = 0x400,
                       ThreadsSynced = 0x800;
  const int tid;
  int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  uint64_t step;
  int *connOffsFifoPtr;   // (flags & OffsFifoEnabled)
  union { // 每个线程各自设置，这一组是选本地buff，还是conn里buff
    T *userBuff;            // (flags & (RoleInput|RoleOutput))
    T *connEltsFifo;        // !(flags & (RoleInput|RoleOutput))
  };
  union { // 每个线程各自设置，这一组是选proxy（conn相关），还是direct buff
    int volatile *connSizesFifoPtr; //  (flags & SizesFifoEnabled)
    T *directBuff;                  // !(flags & SizesFifoEnabled)
  };
  uint64_t volatile *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)

  uint64_t genericOpExecCnt; // 每个线程的本地变量，标记genericOp跑了几次，和ctx.loadAgain配合判断是否是恢复上下文后的第一次执行，如果是，那么genericOp里的slice和offset要从ctx里读。runRing会反复调用genericOp，所以这个变量计数了当前这个primitive实例生命周期内genericOp被调用了几次
  const int currUsedSlotId;
  
  // Don't use barrier 0 as it's used by the final sync
  inline __device__ void barrier() {
    if (nthreads == WARP_SIZE)
      __syncwarp();
    else
      asm volatile("bar.sync %0, %1;" :: "r"(15-group), "r"(nthreads));
    flags |= ThreadsSynced;
  }
  inline __device__ void subBarrier() {
    if (nworkers == nthreads)
      barrier();
    else
      asm volatile("bar.sync %0, %1;" :: "r"(8-group), "r"(nworkers));
  }

  inline __device__ bool checkAbort(int &spins) {
    spins++;
    if (!(flags & Aborted) && spins == OFCCL_SPINS_BEFORE_CHECK_ABORT) {
      flags |= *sharedCollCtx[currUsedSlotId].staticCollCtx.abortFlag ? Aborted : 0;
      spins = 0;
    }
    return flags & Aborted;
  }

  template <int Recv>
  __device__ __forceinline__ void udpateSwitchThreshold(int64_t ctxSwitchCounter) {
    if (sharedCollCtx[currUsedSlotId].recvSuccess) {
      sharedCollCtx[currUsedSlotId].ctxSwitchThreshold = (RECV_SUCCESS_FACTOR * ctxSwitchCounter > sharedCollCtx[currUsedSlotId].ctxSwitchThreshold) ? RECV_SUCCESS_FACTOR * ctxSwitchCounter : sharedCollCtx[currUsedSlotId].ctxSwitchThreshold;
    }

    // sharedCollCtx[currUsedSlotId].ctxSwitchThreshold = (sharedCollCtx[currUsedSlotId].progressed == 1) ?10000 : sharedCollCtx[currUsedSlotId].ctxSwitchThreshold;

    // sharedCollCtx[currUsedSlotId].ctxSwitchThreshold = (sharedCollCtx[currUsedSlotId].recvSuccess == 1) ?10000 : sharedCollCtx[currUsedSlotId].ctxSwitchThreshold;
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t dstIx, intptr_t remoteIx, int offset, int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    const bool noRecvWait = DirectRecv && Src && (flags & DirectRead);        // no wait when directly reading from remote input
    const bool noSendWait = DirectSend && (flags & (DirectRead|DirectWrite)); // no wait in empty send (e.g. directScatter) or direct remote write
    
    // if ((flags & (Recv*RoleWaitRecv))) {
    //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, DirectRecv = %d, noRecvWait = %d, DirectSend = %d, noSendWait = %d, enter waitPeer, coll_id = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, DirectRecv, noRecvWait, DirectSend, noSendWait, blkStatus.currLoadedCollId);
    // }
    // if ((flags & (Send*RoleWaitSend))) {
    //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, DirectRecv = %d, noRecvWait = %d, DirectSend = %d, noSendWait = %d, enter waitPeer, coll_id = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, DirectRecv, noRecvWait, DirectSend, noSendWait, blkStatus.currLoadedCollId);
    // }
    
    int64_t ctxSwitchCounter = 0;
    if (((flags & (Recv*RoleWaitRecv)) && !noRecvWait) || // 0 * 8 + 0 号线程是 RoleWaitRecv(0x04) 0
        ((flags & (Send*RoleWaitSend)) && !noSendWait)) { // 1 * 8 + 0 号线程是 RoleWaitSend 8 
      // int spins = 0;

      // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, enter busy loop, connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) = %llu, step + StepPerSlice = %llu", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, connStepCache + (isSendNotRecv ? NCCL_STEPS : 0), step + StepPerSlice);
      
      // if ((flags & (Recv*RoleWaitRecv))) {
      //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, DirectRecv = %d, noRecvWait = %d, DirectSend = %d, noSendWait = %d, enter busy loop, coll_id = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, DirectRecv, noRecvWait, DirectSend, noSendWait, blkStatus.currLoadedCollId);
      // }
      // if ((flags & (Send*RoleWaitSend))) {
      //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, DirectRecv = %d, noRecvWait = %d, DirectSend = %d, noSendWait = %d, enter busy loop, coll_id = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, DirectRecv, noRecvWait, DirectSend, noSendWait, blkStatus.currLoadedCollId);
      // }

      // 目前RingAllReduce的send在这里等待条件会放宽，fall into while的条件是connStepCache + NCCL_STEPS < step + StepPerSlice)，即connStepCache + 8 < step + 2)，所以send更容易执行
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        
        connStepCache = *connStepPtr;
        // if (checkAbort(spins)) break; // nccl自己有个退出机制，不过没有保留上下文的功能，最终会设置到comm里的一个flag，用来告知用户abort了，去自行处理。
        //if (spins == 0) printf("r=%d b=%d t=%d SPUN OUT got=%d want=%d\n", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, threadIdx.x, int(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), int(step+StepPerSlice));
        
        if (ctxSwitchCounter++ >= sharedCollCtx[currUsedSlotId].ctxSwitchThreshold) {
          // 这里虽然可能会有两个线程都进来，但是进来之后都会设置saveCtx7Quit = 1，然后退出，所以不会有问题；再一个一般来说，也只有waitRecv线程会进来。
          sharedCollCtx[currUsedSlotId].saveCtx7Quit = 1;
          sharedCollCtx[currUsedSlotId].dynamicCollCtx.loadAgain = 1;
          

          // if ((flags & (Recv*RoleWaitRecv))) {
          //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, SHOULD RETURN!! connStepCache(tail from Rank[%d], connStepPtr = %p) = %llu, step + StepPerSlice = %llu, step = %llu, ctxSwitchCounter = %llu", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, (sharedCollCtx[currUsedSlotId].staticCollCtx.rank - 1 + sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks) % sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks, connStepPtr, connStepCache, step + StepPerSlice, step, ctxSwitchCounter);
          // }
          // if ((flags & (Send*RoleWaitSend))) {
          //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, coll_id = %d, SHOULD RETURN!! connStepCache(head from Rank[%d], connStepPtr = %p) + NCCL_STEPS = %llu, step + StepPerSlice = %llu, isSendNotRecv = %d, ctxSwitchCounter = %llu", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, (sharedCollCtx[currUsedSlotId].staticCollCtx.rank + 1) % sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks, connStepPtr, connStepCache + NCCL_STEPS, step + StepPerSlice, isSendNotRecv, ctxSwitchCounter);
          // }
          // __syncwarp(); // ！！！！！！为了打印log加的！
          
          return; // 使用return，而不是break。
        }
      }
    }

    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) { // 这里还是只有特殊线程在做

      // wait成功，在subBarrier和barrier之前，标记一下，保证后续的可见性。
      atomicOr(&sharedCollCtx[currUsedSlotId].progressed, (((flags & RoleWaitRecv) != 0) || ((!Recv) && ((flags & RoleWaitSend) != 0))));
      // 若RoleWaitRecv线程wait成功，那可以设置；若RoleWaitSend线程wait成功，只有在不需要recv的时候才可以设置；并且一旦设成1，就不再清零了。

      atomicOr(&sharedCollCtx[currUsedSlotId].recvSuccess, (flags & (Recv*RoleWaitRecv)) != 0);

      udpateSwitchThreshold<Recv>(ctxSwitchCounter);

      // if ((flags & (Recv*RoleWaitRecv))) {
      //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, progressed = %d, Send = %d, Recv = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, sharedCollCtx[currUsedSlotId].progressed, Send, Recv);
      // }
      // if ((flags & (Send*RoleWaitSend))) {
      //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, coll_id = %d, progressed = %d, Send = %d, Recv = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, sharedCollCtx[currUsedSlotId].progressed, Send, Recv);
      // }

      if (isSendNotRecv && (flags & SizesFifoEnabled)) // proxy 相关，不用考虑
        connSizesFifoPtr[step%NCCL_STEPS] = nelts*sizeof(T);

      void **ptrs = isSendNotRecv ? (sharedCollCtx[currUsedSlotId].groups[group].dsts + Dst)
                                  : (sharedCollCtx[currUsedSlotId].groups[group].srcs + Src);
      if (flags & OffsFifoEnabled) // proxy 相关，不用考虑
        ptrs[index] = connEltsFifo + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
      else if (isSendNotRecv && DirectSend) {
        if (flags & DirectWrite) {
          ptrs[index] = directBuff + remoteIx + offset;
        } else if (flags & DirectRead) {  // empty send
          ptrs[index] = nullptr;
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
        }
      } else if (!isSendNotRecv && DirectRecv) { 
        if (flags & DirectRead) {
          ptrs[index] = directBuff + remoteIx + offset;
        } else if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;  // send to next from my output buffer
          // OFCCL_LOG(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, update srcs[0]=%p; directBuff=%p, dstIx=%lx, offset=%x", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, sharedCollCtx[currUsedSlotId].groups[group].srcs[0], directBuff, dstIx, offset);
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
        }
      }
      else {
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
      }
      step += StepPerSlice;
    }
    
    // if ((flags & (Recv*RoleWaitRecv))) {
    //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, waitPeer success, tail from Rank[%d] = %llu, new step %llu", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, (sharedCollCtx[currUsedSlotId].staticCollCtx.rank - 1 + sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks) % sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks, connStepCache, step);
    // }
    // if ((flags & (Send*RoleWaitSend))) {
    //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, coll_id = %d, waitPeer success connStepCache(head from Rank[%d]) + NCCL_STEPS = %llu, new step %llu, isSendNotRecv = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, (sharedCollCtx[currUsedSlotId].staticCollCtx.rank + 1) % sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks, connStepCache + NCCL_STEPS, step, isSendNotRecv);
    // }
    // __syncwarp(); // ！！！！！！为了打印log加的！
  }

  template<int Recv, int Send>
  inline __device__ void postPeer() {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      *connStepPtr = step;
    }
    
    // if ((flags & (Recv*RolePostRecv))) {
    //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RolePostRecv>, coll_id = %d, postPeer update head: *connStepPtr = %llu, connStepPtr = %p, to Rank[%d]", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, *connStepPtr, connStepPtr, (sharedCollCtx[currUsedSlotId].staticCollCtx.rank - 1 + sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks) % sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks);
    // }
    // if ((flags & (Send*RolePostSend))) {
    //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RolePostSend>, coll_id = %d, postPeer update tail = *connStepPtr = %llu, connStepPtr = %p, to Rank[%d]", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, *connStepPtr, connStepPtr, (sharedCollCtx[currUsedSlotId].staticCollCtx.rank + 1) % sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks);
    // }
    // __syncwarp(); // ！！！！！！为了打印log加的！
  }

  // 后两个模板参数的常见取值：static constexpr int Input=0, Output=1;
  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, intptr_t remoteIx, int nelem, bool postOp
    ) {
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 0 + 12 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
    // 下边这两个由模板参数决定
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    // 只要传了那个静态变量，不是-1，下边的值就是true
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    // stepSize(sharedCollCtx[currUsedSlotId].comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T))
    // SlicePerChunk = 2, StepPerSlice = 2
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);

    // bugfix: ！！！！！！！！！！！！！不应该每次都从ctx里边恢复！！！！！！！！！！！！！！
    // 之后刚恢复回来那次，才应该读保存的ctx。
    int slice = 0;
    int offset = 0;
    if (genericOpExecCnt++ == 0 && sharedCollCtx[currUsedSlotId].dynamicCollCtx.loadAgain == 1) {
      slice = sharedCollCtx[currUsedSlotId].dynamicCollCtx.slice4SimpleGenericOp;
      offset = sharedCollCtx[currUsedSlotId].dynamicCollCtx.offset4SimpleGenericOp;
    }
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 1 + 12 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif

    if (tid < nworkers && offset < nelem) {
      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
        // The RTL representation of the code for a function is a doubly-linked chain of objects called insns. Insns are expressions with special codes that are used for no other purpose. Some insns are actual instructions; others represent dispatch tables for switch statements; others represent labels to jump to or various sorts of declarative information. https://gcc.gnu.org/onlinedocs/gccint/Insns.html
      // 这段代码的意思是，这个优化减少了if分支的数量。cuda没有分支预测，所以碰到if，就会暂停流水线。
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
      #ifdef ARRAY_DEBUG
        *(blkStatus.barrierCnt + 2 + 12 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      #endif
      #if __CUDA_ARCH__ < 700
        // Yeah, so all that above don't matter a lick on older hardware.
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        
        // OFCCL_LOG_WARP_HEAD(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, offset = %d, sliceSize = %d, nelem = %d, slice = %d, SlicePerChunk = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, offset, sliceSize, nelem, slice, SlicePerChunk);

        if (Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput))) {
          sharedCollCtx[currUsedSlotId].groups[group].srcs[0] = userBuff + srcIx + offset; // 传给srcIx形参其实也是个offset
          // OFCCL_LOG(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, userBuff = %p, srcIx = 0x%lx, offset = 0x%x, srcs[0]=%p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, userBuff, srcIx, offset, sharedCollCtx[currUsedSlotId].groups[group].srcs[0]);
        }
        if (Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput))) {
          sharedCollCtx[currUsedSlotId].groups[group].dsts[0] = userBuff + dstIx + offset;
          // OFCCL_LOG(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, userBuff = %p, dstIx = 0x%lx, offset = 0x%x, dsts[0]=%p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, userBuff, dstIx, offset, sharedCollCtx[currUsedSlotId].groups[group].dsts[0]);
        }
          
        // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, before call waitPeer", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid);
        
        // if ((flags & (Recv*RoleWaitRecv))) {
        //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, upper enter waitPeer", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
        // }
        // if ((flags & (Send*RoleWaitSend))) {
        //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, coll_id = %d, upper enter waitPeer", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
        // }
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 0 + 9 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(dstIx, remoteIx, offset, sliceSize);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 1 + 9 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
        // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, waitPeer return, before subBarrier, saveCtx7Quit = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, sharedCollCtx[currUsedSlotId].saveCtx7Quit);
        // if ((flags & (Recv*RoleWaitRecv))) {
        //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, waitPeer return, before subBarrier", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
        // }
        // if ((flags & (Send*RoleWaitSend))) {
        //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_P2P, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, coll_id = %d, waitPeer return, before subBarrier", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
        // }
        subBarrier();
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 2 + 9 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
        // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after subBarrier", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid);
        if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
          if (tid == 0) {
            sharedCollCtx[currUsedSlotId].dynamicCollCtx.slice4SimpleGenericOp = slice;
            sharedCollCtx[currUsedSlotId].dynamicCollCtx.offset4SimpleGenericOp = offset;
            // __threadfence_block();
            // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, offset = %d, slice = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, offset, slice);
          }

          // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d> before barrier 0, genericOp worker wait fail, slice = %d(SlicePerChunk=%d), offset = %d(nelem=%d, sliceSize=%d)", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, slice, SlicePerChunk, offset, nelem, sliceSize);
          // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d> before barrier 0, genericOp worker wait fail, slice = %d, offset = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, slice, offset);
          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 0 + 0 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif
          barrier(); // 我们在这里直接返回，跳过数据搬运，所以把相应的barrier调用了。
          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 1 + 0 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif
          
          goto generic_op_quit;
          // return;
        }
        if (DirectRecv && sharedCollCtx[currUsedSlotId].groups[group].srcs[0] == sharedCollCtx[currUsedSlotId].groups[group].dsts[0]) {

          // OFCCL_LOG_WARP_HEAD(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, case 1, DirectRecv = %d, srcs[0]=%p, dsts[0]=%p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, DirectRecv, sharedCollCtx[currUsedSlotId].groups[group].srcs[0], sharedCollCtx[currUsedSlotId].groups[group].dsts[0]);

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 0 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send) {

            // OFCCL_LOG_WARP_HEAD(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, case 1 inner, Send = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, Send);

            #ifdef ARRAY_DEBUG
              *(blkStatus.barrierCnt + 2 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
            #endif

            // (1-Send) is only there to avoid compilation errors in case MaxSend=0 (and Send=0).
            ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, (1-Send)+MaxSend, 0>
              (tid, nworkers, nullptr, false,
               1, (T const**)sharedCollCtx[currUsedSlotId].groups[group].srcs,
               fan.nsend(), (T**)sharedCollCtx[currUsedSlotId].groups[group].dsts+1,
               sliceSize);

            #ifdef ARRAY_DEBUG
              *(blkStatus.barrierCnt + 3 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
            #endif
          }

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 1 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

        } else if (DirectSend && !DirectRecv && SrcBuf != Input && sharedCollCtx[currUsedSlotId].groups[group].dsts[Dst] == nullptr) {

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 4 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

          // For broadcast in CollNet to do empty send
          ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1, 0>
            (tid, nworkers, sharedCollCtx[currUsedSlotId].redOpArgs, postOp,
             Recv, (T const**)sharedCollCtx[currUsedSlotId].groups[group].srcs,
             Dst, (T**)sharedCollCtx[currUsedSlotId].groups[group].dsts,
             sliceSize);

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 5 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

        } else {

          // OFCCL_LOG_WARP_HEAD(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, case 3, DirectRecv = %d, srcs[0]=%p, dsts[0]=%p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, DirectRecv, sharedCollCtx[currUsedSlotId].groups[group].srcs[0], sharedCollCtx[currUsedSlotId].groups[group].dsts[0]);

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 6 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

          constexpr int PreOpN = SrcBuf != Input ? 0 :
                                 DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          ReduceOrCopyMulti<Unroll, RedOp, T, Recv+Src, Recv*MaxRecv+Src, Send+Dst, Send*MaxSend+Dst, PreOpN>
            (tid, nworkers, sharedCollCtx[currUsedSlotId].redOpArgs, postOp,
             Recv*fan.nrecv()+Src, (T const**)sharedCollCtx[currUsedSlotId].groups[group].srcs,
             Send*fan.nsend()+Dst, (T**)sharedCollCtx[currUsedSlotId].groups[group].dsts,
             sliceSize);

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 7 + 13 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

        }
        // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d> before barrier 1, genericOp worker wait success, slice=%d(SlicePerChunk=%d), offset=%d(nelem=%d, sliceSize=%d)", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, slice, SlicePerChunk, offset, nelem, sliceSize);
        // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d> before barrier 1, genericOp worker wait success, slice=%d, offset=%d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, slice, offset);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 0 + 1 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
        barrier(); // This barrier has a counterpart in following loop
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 1 + 1 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
        if (Send && (flags & RolePostSend) && index == 0) __threadfence_system();
        __syncwarp();
        postPeer<Recv, Send>();
        offset += sliceSize;
        slice += 1;
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn(instruction).
    #pragma unroll 1
    // while (slice < SlicePerChunk && offset < nelem) { // 过滤不了。~~挺巧妙的，用这个条件把nworkers里的线程过滤掉了~~。// +  1023bug: 应该是从这里又让0号等线程钻进了waitPeer，然后设置了SaveCtx7Quit。
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        // if ((flags & (Recv*RoleWaitRecv))) {
        //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, lower enter waitPeer", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
        // }
        // if ((flags & (Send*RoleWaitSend))) {
        //   OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, coll_id = %d, lower enter waitPeer", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
        // }
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 3 + 9 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, 0);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 4 + 9 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
      }

      // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d> before barrier 2, after lower waitPeer, slice=%d, offset=%d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, slice, offset);
      #ifdef ARRAY_DEBUG
        *(blkStatus.barrierCnt + 0 + 2 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      #endif
      barrier(); // Has couterpart in preceding worker-only loop.
      #ifdef ARRAY_DEBUG
        *(blkStatus.barrierCnt + 1 + 2 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      #endif
      if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) { // 需要在barrier之后才访问shmem。
        //  OFCCL_LOG_WARP_HEAD(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> after barrierCnt 2, slice = %d, offset = %d genericOp non-worker return", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, slice, offset);
          // 通常情况下，nworkers以外的线程跑到这里，所以在上边的waitPeer里也不会做什么，各个线程的slice和offset看起来应该是通过barrier的同步，可以同步更新，所以之后恢复的时候，直接恢复0号线程的slice和offset应该没问题；他这里就不用保存了；加一个判断不让它跑到postPeer就好。
          
          goto generic_op_quit; 
          // return;
      }
      // 注意到这里的内存保护，先插入fence，而在fence之前有个barrier，worker线程中的barrier是在数据搬运完成之后调用的，所以这里就保证了数据搬运完成，插入fence的顺序；在插入fence之后，在通过postPeer使tail对接收端可见，使head对发送端可见。
      if (Send && (flags & RolePostSend) && sliceSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += sliceSize;
      slice += 1;
    }
    // waitPeer后边加了subBarrier来进行同步，但是postPeer后边没有任何同步方式，我们修改了nccl的行为方式，原来waitPeer那里可以无限等，现在加入了更加积极的主动跳过的方法，不加一个同步的话，postPeer里的工作本身并不轻松，可能导致block内线程的分化。
   
    generic_op_quit:
      #ifdef ARRAY_DEBUG
        *(blkStatus.barrierCnt + 0 + 3 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      #endif
      ofcclBarrier(14, nthreads);
      #ifdef ARRAY_DEBUG
        *(blkStatus.barrierCnt + 1 + 3 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      #endif
  }

  // TODO: 省略了 ScatterGatherOp

  // connIndex = group >> 16 = 0, e = nullptr，通过打log确认了，打log也确认了，经过loadRecvConn和loadSendConn，flags都没变。
  // 所以这两个函数的效果，就是设置了step，设置了 connStepPtr ，设置了 connStepCache，设置了 connEltsFifo，设置了ofcclShmem里的peer的conn
  // SlicePerChunk=2, StepPerSlice=2
  __device__ __forceinline__ void loadRecvConn(ncclPeer *peer, int connIndex, struct ncclWorkElem* e) {
    if (flags & (RoleWaitRecv|RolePostRecv)) {
      auto *conn = &peer->recv[connIndex].conn;
      step = conn->step; // uint64_t step;      // Keep where we are // 设置好就不会变了；每个线程有自己的值
      if (sharedCollCtx[currUsedSlotId].dynamicCollCtx.loadAgain == 0) {
        step = roundUp(step, SlicePerChunk*StepPerSlice); // return (x+y-1) - (x+y-1)%y;，就是向上取到y的倍数
      }
      
      // if ((flags & (RoleWaitRecv))) {
      //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, load step(head) = %llu from conns", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, step);
      // }
      // if ((flags & (RolePostRecv))) {
      //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RolePostRecv>, load step(head) = %llu from conns", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, step);
      // }
      // __syncwarp(); // ！！！！！！为了打印log加的！！！！

      if (flags & RolePostRecv) { // (ng - 2) * 8 + 0 号线程是 RolePostRecv
        connStepPtr = conn->head; // uint64_t *head;     // Local for send, remote for recv
        *connStepPtr = step; // Return credits in case we rounded up.
      }
      if (flags & RoleWaitRecv) { // 0 * 8 + 0 号线程是 RoleWaitRecv
        // conn 来自于&peer->recv[connIndex].conn，构造函数里connIndex是0，所以不论这里的index是几，都是同一个conn，所以目前不用太关心
        sharedCollCtx[currUsedSlotId].groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()

        // #ifdef ARRAY_DEBUG
        //   *(blkStatus.barrierCnt + 2 + 16 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = (unsigned long long)sharedCollCtx[currUsedSlotId].groups[group].recvConns[index];
        //   *(blkStatus.barrierCnt + 3 + 16 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = (unsigned long long)sharedCollCtx[currUsedSlotId].groups[group].recvConns[index]->ptrExchange;
        // #endif

        connStepPtr = conn->tail; // uint64_t *tail;     // Local for recv, remote for send
        connStepCache = *connStepPtr; // 这个应该就是simple协议的标记位，这个被设置了，就代表buffer里是新数据了。对于recv来说，就代表收到了新数据
        flags |= (conn->offsFifo != nullptr) ? OffsFifoEnabled : 0; // 这个和proxy相关，先不关注。int *offsFifo;      // Buffer fifo from proxy to GPU，所以对GPU是recv

        // OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, conn->offsFifo = %p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, conn->offsFifo);

        if (Direct) { // 模板参数 Direct 是 1，下边会区分两种direct的方式
          // User buffers have been registered
          // 现在log打印出来 conn->direct=0
          if ((conn->direct & (NCCL_IPC_READ|NCCL_IPC_WRITE)) && e != nullptr && e->regUsed) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= (e->direct & NCCL_DIRECT_WRITE) ? DirectWrite :
                       (e->direct & NCCL_DIRECT_READ)  ? DirectRead  : 0;
            }
          } else if (conn->direct & (NCCL_DIRECT_WRITE|NCCL_DIRECT_READ)) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              // direct read not allowed in non-register case
              // otherwise, in one-to-multi send, we could mix empty send and intermediate send
              flags |= (conn->direct & NCCL_DIRECT_WRITE) ? DirectWrite : 0;
            }
          }
        }
        if (flags & OffsFifoEnabled)
          connOffsFifoPtr = conn->offsFifo;
        connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclPeer *peer, int connIndex, struct ncclWorkElem* e) {
    if (flags & (RoleWaitSend|RolePostSend)) {
      auto *conn = &peer->send[connIndex].conn;
      step = conn->step;
      if (sharedCollCtx[currUsedSlotId].dynamicCollCtx.loadAgain == 0) {
        step = roundUp(step, SlicePerChunk*StepPerSlice);
      }

      // if ((flags & (RoleWaitSend))) {
      //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, load step(tail) = %llu from conns", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, step);
      // }
      // if ((flags & (RolePostSend))) {
      //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RolePostSend>, load step(tail) = %llu from conns", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, step);
      // }
      // __syncwarp(); // ！！！！！！为了打印log加的！！！！

      if (flags & RolePostSend) {
        connStepPtr = conn->tail;
      }
      if (flags & RoleWaitSend) {
        sharedCollCtx[currUsedSlotId].groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()

        // #ifdef ARRAY_DEBUG
        //   *(blkStatus.barrierCnt + 4 + 16 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = (unsigned long long)sharedCollCtx[currUsedSlotId].groups[group].sendConns[index];
        //   *(blkStatus.barrierCnt + 5 + 16 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = (unsigned long long)sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->ptrExchange;
        // #endif

        connStepPtr = conn->head;
        connStepCache = *connStepPtr;
        flags |= (conn->offsFifo != nullptr) ? OffsFifoEnabled : 0;

        // OFCCL_LOG_RANK_X_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, coll_id = %d, conn->offsFifo = %p, conn->sizesFifo = %p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, conn->offsFifo, conn->sizesFifo);
        
        if (flags & OffsFifoEnabled)
          connOffsFifoPtr = conn->offsFifo;
        connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];

        if (conn->sizesFifo != nullptr) { // int *sizesFifo;     // Sizes fifo from GPU to proxy，从GPU发到网络，是send，先不关注。
          flags |= SizesFifoEnabled;
          connSizesFifoPtr = conn->sizesFifo;
        } else if (Direct) { // 模板参数 Direct 是 1，下边会区分两种direct的方式
          // User buffers have been registered
          if ((conn->direct & (NCCL_IPC_READ|NCCL_IPC_WRITE)) && e != nullptr && e->regUsed) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= (e->direct & NCCL_DIRECT_WRITE) ? DirectWrite :
                       (e->direct & NCCL_DIRECT_READ)  ? DirectRead  : 0;
            }
          } else if (conn->direct & (NCCL_DIRECT_WRITE|NCCL_DIRECT_READ)) {
            if (connIndex == 1 && P2p == 0) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              // direct read not allowed in non-register case
              // otherwise, in one-to-multi send, we could mix empty send and intermediate send
              flags |= (conn->direct & NCCL_DIRECT_WRITE) ? DirectWrite : 0;
            }
          }
        }
      }
    }
  }

 public:

  // 模板参数：Ring AllReduce里边，MaxSend和MaxRecv都是 1，Direct = 1
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint32_t group=0, struct ncclWorkElem* e = nullptr
    ):
    tid(tid),
    genericOpExecCnt(0),
    currUsedSlotId(blkStatus.currLoadedCollId % NUM_SHMEM_SLOT),
    stepSize(sharedCollCtx[currUsedSlotId].buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)) {

    // For send operations, we need an extra warp to overlap the threadfence and the copy
    this->nthreads = nthreads;
    this->nworkers = nthreads - (MaxSend > 0 && nthreads-WARP_SIZE >= 64 ? WARP_SIZE : 0); // 这一行的意思是，nthreads足够大的话，就可以剔除出一个warp了，剩下的是worker。
    this->group = group & (uint16_t)0xFFFF; // 还是 0. ring的情况下，group一直是0。
    int connIndex = group >> 16; // 还是 0.

    int nrecv=0, nsend=0; // 后边两行代码之后，变成nrecv=1, nsend=1，这也是因为ring AllReduce，上下游只有一个peer吧。
    while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
    while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
    this->fan = Fan(nrecv, nsend); // 相当于调用 __device__ FanSymmetric<1>(1, 1)

    constexpr int ThreadPerSync = 8;
    static_assert(MaxSend < ThreadPerSync && MaxRecv < ThreadPerSync, "Not enough threads to cover all peers");

    int g = tid / ThreadPerSync; // 当前线程属于哪个 “g”
    int ng = nthreads / ThreadPerSync; // 一共有几个 “g”
    index = tid % ThreadPerSync; // 当前线程在 “g” 里的位置 // Peer index I'm responsible for
    flags = 0;
    if (g == 0) {
      if (index < nrecv) flags |= RoleWaitRecv; // 0 * 8 + 0 号线程是 RoleWaitRecv(0x04) 0
      if (index == nrecv) flags |= RoleInput; // 0 * 8 + 1 号线程是 RoleInput(0x08) 1
    } else if (g == 1) {
      if (index < nsend) flags |= RoleWaitSend; // 1 * 8 + 0 号线程是 RoleWaitSend 8 
      if (index == nsend) flags |= RoleOutput; // 1 * 8 + 1 号线程是 RoleOutput 9
    } else if (g == ng - 2) { // 从 ng - 3 到 ng 都是属于最后一个warp的。
      if (index < nrecv) flags |= RolePostRecv; // (ng - 2) * 8 + 0 号线程是 RolePostRecv(0x20) 272
    } else if (g == ng - 1) {
      if (index < nsend) flags |= RolePostSend; // (ng - 1) * 8 + 0 号线程是 RolePostSend(0x10) 280
    }
    // 所以总的来看，有flag的线程不多，事实上只有6个。
    // 没有任何线程被 **单独** 设置了flag DirectWrite 和 DirectRead

    int peer = 0;
    if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index]; // 有RoleWaitRecv或RolePostRecv标记的线程，设置其peer。虽然我们给线程分配了0-7的index，但是只给index=0的线程分配了recv相关的flag。这个函数同时也会根据conn.direct，为flags追加DirectRead、DirectWrite
    if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index]; // 同上
    
    // OFCCL_LOG_RANK_X_THRD_0_SHMEM(OFCCL_MPI, 0, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, coll_id = %d, peer = %d, connIndex = %d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, peer, connIndex);

    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 0 + 17 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif

    loadRecvConn(&sharedCollCtx[currUsedSlotId].staticCollCtx.devPeers[peer], connIndex, e); // 没有RoleWaitRecv或RolePostRecv标记的线程直接返回。

    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 1 + 17 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
    loadSendConn(&sharedCollCtx[currUsedSlotId].staticCollCtx.devPeers[peer], connIndex, e); // 没有RoleWaitSend或RolePostSend标记的线程直接返回。

    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 2 + 17 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif

    // inputBuf = sendbuff, outputBuf = recvbuff
    setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclWorkElemReg*)e); // 事实上也是需要被指定了flag的线程才会进行相应的工作

    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 3 + 17 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
  
    // ofcclBarrier(6); // TODO: 可删。
  }

  __device__ ~Primitives() {
    // Ensure sharedCollCtx[currUsedSlotId].groups[].send/recvConns are available
    if (!(flags & ThreadsSynced)) {
      #ifdef ARRAY_DEBUG
        *(blkStatus.barrierCnt + 0 + 19 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      #endif
      barrier();
      #ifdef ARRAY_DEBUG
        *(blkStatus.barrierCnt + 0 + 19 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
      #endif
    }
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) {
      auto *conns = (flags & RolePostSend) ? sharedCollCtx[currUsedSlotId].groups[group].sendConns : sharedCollCtx[currUsedSlotId].groups[group].recvConns;
      conns[index]->step = step;
    }
    
    // if ((flags & (RolePostRecv))) {
    //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RolePostRecv>, save step(head) = %llu", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, step);
    // }
    // if ((flags & (RolePostSend))) {
    //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d-RolePostSend>, save step(tail) = %llu", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, step);
    // }
    // __syncwarp(); // ！！！！！！为了打印log加的！！！！

    // Make sure all threads are done writing back conn->step and done using
    // sharedCollCtx[currUsedSlotId].groups[group]
    // OFCCL_LOG_WARP_HEAD(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> before barrierCnt 4", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid);
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 0 + 4 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
    barrier();
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 1 + 4 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
  }

  // inputBuf = sendbuff, outputBuf = recvbuff
  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, uint64_t redOpArg, struct ncclWorkElemReg* e) {
    if (flags & RoleInput) { // 0 * 8 + 1 号线程是 RoleInput(0x08) 1
      userBuff = (T*)inputBuf;
      sharedCollCtx[currUsedSlotId].redOpArgs[0] = redOpArg;  // scaler for local input
    }
    if (flags & RoleOutput) userBuff = (T*)outputBuf; // 1 * 8 + 1 号线程是 RoleOutput 9
    bool recvProvider = flags == (flags|RoleWaitRecv|DirectWrite);
    bool sendAcceptor = flags == (flags|RoleWaitSend|DirectWrite);
    bool sendProvider = flags == (flags|RoleWaitSend|DirectRead); // sender provides direct buffer (to be fetched)
    bool recvAcceptor = flags == (flags|RoleWaitRecv|DirectRead); // receiver accepts direct buffer
    int regUsed = e != nullptr ? e->elem.regUsed : 0;
    // 打log 显示上边5个变量全是0
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 0 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = recvProvider;
      *(blkStatus.barrierCnt + 1 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = sendAcceptor;
      *(blkStatus.barrierCnt + 2 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = sendProvider;
      *(blkStatus.barrierCnt + 3 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = recvAcceptor;
      *(blkStatus.barrierCnt + 4 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = regUsed;
    #endif

    // if (!(recvProvider == 0 && sendAcceptor == 0 && sendProvider == 0 && recvAcceptor == 0 && regUsed == 0)) {
    //   OFCCL_LOG(OFCCL, "Rank<%d>, Blk<%d>, Thrd<%d>, recvProvider=%d, sendAcceptor=%d, sendProvider=%d, recvAcceptor=%d, regUsed=%d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, threadIdx.x, recvProvider, sendAcceptor, sendProvider, recvAcceptor, regUsed);
    //   __syncwarp(); // ！！！！！！为了打印log加的！！！！
    // }

    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 5 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif

    // 模板参数 Direct 是 1
    // 如果是从上下文恢复，那就什么也不用做，否则按照原来的流程
    if (Direct && recvProvider) {

      // bugfix：不论是否是上下文切换恢复回来的，都需要初始化0号线程的Primitive对象的directBuff成员变量。
      directBuff = (T*)outputBuf;

      if (!sharedCollCtx[currUsedSlotId].dynamicCollCtx.loadAgain) {
        int spins = 0;
        void *volatile *slot = sharedCollCtx[currUsedSlotId].groups[group].recvConns[index]->ptrExchange;
        // Wait for consumer to consume previous value before trampling it.

        while (*slot != nullptr && !checkAbort(spins));
        // Encode pointer by XOR'ing against some address they definitely wouldn't send
        // since we want to allow them sending us nullptr while not colliding with
        // the empty slot value. // 这个编码方式没太懂。
        *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(directBuff) ^ reinterpret_cast<uintptr_t>(slot));
        
        // OFCCL_LOG(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, encode directBuff=%p to *(recvConns[index]->ptrExchange)=%p, in recvConns[index]->ptrExchange@%p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, directBuff, *slot, slot);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 7 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = (unsigned long long)directBuff;
        #endif
      }
    }
    if (Direct && sendAcceptor) {
      if (sharedCollCtx[currUsedSlotId].dynamicCollCtx.loadAgain) {
        void *ptr = sharedCollCtx[currUsedSlotId].dynamicCollCtx.sendConnPtrExchage;
        void **slot = sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->ptrExchange;
        directBuff = regUsed ? (T*)(e->dnOutputs[index]) :
                    reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));

        // OFCCL_LOG(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, recover sendConnPtrExchage=%p from context, directBuff=%p, sendConns[index]->ptrExchange@%p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, ptr, directBuff, sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->ptrExchange);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 7 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = (unsigned long long)directBuff;
        #endif
      } else {
        int spins = 0;
        void *volatile *slot = sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->ptrExchange;
        void *ptr;
        while (true) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(spins)) break;
        }
        directBuff = regUsed ? (T*)(e->dnOutputs[index]) :
                    reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));

        // bugfix：这个赋值不太好等到发现fail再做，因为下边就直接把*slot设为了nullptr，也就是 *(sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->ptrExchange)被设为了nullptr，所以fail的时候，这个值已经丢失了。
        sharedCollCtx[currUsedSlotId].dynamicCollCtx.sendConnPtrExchage = ptr;

        // bugfix：应该把这个重置放到coll完成的时候，来实现一种peer间的同步。 // *slot = nullptr;

        // OFCCL_LOG(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, init *(sendConns[index]->ptrExchange)=%p with peer, directBuff=%p, sendConns[index]->ptrExchange@%p", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, ptr, directBuff, sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->ptrExchange);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 7 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) = (unsigned long long)directBuff;
        #endif
      }
    }
    if (Direct && sendProvider) {
      int spins = 0;
      void *volatile *slot = sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->ptrExchange;
      volatile uint64_t* argSlot0 = sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->redOpArgExchange;
      volatile uint64_t* argSlot1 = sharedCollCtx[currUsedSlotId].groups[group].sendConns[index]->redOpArgExchange+1;
      // Wait for consumer to consume previous value before trampling it.
      while ((*slot != nullptr || *argSlot0 != 0 || *argSlot1 !=0) && !checkAbort(spins));
      // If there is no recv, then we are directly pulling from input buffer (e.g. directScatter)
      // Otherwise, we are pulling from output buffer (e.g. recvCopyDirectSend)
      directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
      // Exchange pre-scalers for use in direct pull
      *argSlot0 = (uint64_t(1)<<32) | (uint32_t)redOpArg;
      *argSlot1 = (uint64_t(1)<<32) | (uint32_t)(redOpArg>>32);
      // Encode pointer by XOR'ing against some address they definitely wouldn't send
      // since we want to allow them sending us nullptr while not colliding with
      // the empty slot value.
      *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(directBuff) ^ reinterpret_cast<uintptr_t>(slot));
    }
    if (Direct && recvAcceptor) {
      int spins = 0;
      void *volatile *slot = sharedCollCtx[currUsedSlotId].groups[group].recvConns[index]->ptrExchange;
      volatile uint64_t* argSlot0 = sharedCollCtx[currUsedSlotId].groups[group].recvConns[index]->redOpArgExchange;
      volatile uint64_t* argSlot1 = sharedCollCtx[currUsedSlotId].groups[group].recvConns[index]->redOpArgExchange+1;
      void *ptr;
      while (true) {
        ptr = *slot;
        if (ptr != nullptr || checkAbort(spins)) break;
      }
      directBuff = regUsed ? (T*)(MaxSend == 0 ? e->upOutputs[index] : e->dnInputs[index]) :
                   reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));
      if (MaxSend != 0) { // reduce group rather than gather group
        // Store scalers for remote inputs
        uint64_t arg0, arg1;
        while (true) {
          arg0 = *argSlot0;
          arg1 = *argSlot1;
          if ((arg0 != 0 && arg1 != 0) || checkAbort(spins)) break;
        }
        sharedCollCtx[currUsedSlotId].redOpArgs[1+index] = ((arg1 & 0xffffffff)<<32) | (arg0 & 0xffffffff);
      }
      *argSlot0 = 0; *argSlot1 = 0;
      *slot = nullptr;
    }
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 6 + 7 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (flags & (RoleInput|RoleOutput))
      userBuff += delta;
  }
  
  // static constexpr int Input=0, Output=1;
  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, -1, remoteOutIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, -1, remoteOutIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(-1, outIx, -1, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopySend(intptr_t outIx, intptr_t remoteOutIx, int eltN) {
    genericOp<1, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, false);
  }
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, remoteOutIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, intptr_t remoteInpIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, remoteInpIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopySend(intptr_t inpIx, intptr_t outIx, intptr_t remoteOutIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    // 在这个操作中契合了俊丞写的 directRecv是被动操作。
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, remoteOutIx, eltN, postOp);
  }

};