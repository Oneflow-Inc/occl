/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

//                                       NCCL_STEPS(8)/2=4     NCCL_STEPS/4=2         2
// ringAllreduce里边，Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
// 对应                       ProtoSimple<          SlicePerChunk,                    StepPerSlice, Unroll>
//                                                SlicePerChunk=2                     StepPerSlice=2
// Unroll 在primitives.h 里有默认值 COLL_UNROLL=4
// P2p=0
#include "common.h"
#include "debug.h"
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
    if (!(flags & Aborted) && spins == NCCL_SPINS_BEFORE_CHECK_ABORT) {
      flags |= *ncclShmem.comm.abortFlag ? Aborted : 0;
      spins = 0;
    }
    return flags & Aborted;
  }

  // 所有模板参数都和genericOp相同
  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t dstIx, intptr_t remoteIx, int offset, int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    const bool noRecvWait = DirectRecv && Src && (flags & DirectRead); // 目前应该一直是false       // no wait when directly reading from remote input
    const bool noSendWait = DirectSend && (flags & (DirectRead|DirectWrite)); // 目前应该一直是false // no wait in empty send (e.g. directScatter) or direct remote write
    if (((flags & (Recv*RoleWaitRecv)) && !noRecvWait) || // 0 * 8 + 0 号线程是 RoleWaitRecv(0x04) 0
        ((flags & (Send*RoleWaitSend)) && !noSendWait)) { // 1 * 8 + 0 号线程是 RoleWaitSend 8

      // if ((flags & (Recv*RoleWaitRecv))) {
      //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, enter waitPeer", ncclShmem.comm.rank, blockIdx.x, tid);
      // }
      // if ((flags & (Send*RoleWaitSend))) {
      //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, enter waitPeer", ncclShmem.comm.rank, blockIdx.x, tid);
      // }

      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = *connStepPtr;
        if (checkAbort(spins)) break; // nccl自己有个退出机制，不过没有保留上下文的功能
        //if (spins == 0) printf("r=%d b=%d t=%d SPUN OUT got=%d want=%d\n", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, int(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), int(step+StepPerSlice));
      }
    }

    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) { // 这里还是只有特殊线程在做
      if (isSendNotRecv && (flags & SizesFifoEnabled)) // proxy 相关，不用考虑
        connSizesFifoPtr[step%NCCL_STEPS] = nelts*sizeof(T);

      void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                                  : (ncclShmem.groups[group].srcs + Src);
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
          // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, update srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
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
    //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, waitPeer success, tail from Rank[%d] = %llu, new step %llu", ncclShmem.comm.rank, blockIdx.x, tid, (ncclShmem.comm.rank - 1 + ncclShmem.comm.nRanks) % ncclShmem.comm.nRanks, connStepCache, step);
    // }
    // if ((flags & (Send*RoleWaitSend))) {
    //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, waitPeer success, head from Rank[%d] = %llu, new step %llu", ncclShmem.comm.rank, blockIdx.x, tid, (ncclShmem.comm.rank - 1 + ncclShmem.comm.nRanks) % ncclShmem.comm.nRanks, connStepCache, step);
    // }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer() {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      *connStepPtr = step;
    }
    
    // if ((flags & (Recv*RolePostRecv))) {
    //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RolePostRecv>, postPeer update head: *connStepPtr = %llu, connStepPtr = %p, to Rank[%d]", ncclShmem.comm.rank, blockIdx.x, tid, *connStepPtr, connStepPtr, (ncclShmem.comm.rank - 1 + ncclShmem.comm.nRanks) % ncclShmem.comm.nRanks);
    // }
    // if ((flags & (Send*RolePostSend))) {
    //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RolePostSend>, postPeer update tail: *connStepPtr = %llu, connStepPtr = %p, to Rank[%d]", ncclShmem.comm.rank, blockIdx.x, tid, *connStepPtr, connStepPtr, (ncclShmem.comm.rank + 1) % ncclShmem.comm.nRanks);
    // }
  }

  // 后两个模板参数的常见取值：static constexpr int Input=0, Output=1;
  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, intptr_t remoteIx, int nelem, bool postOp
    ) {
    // 下边这两个由模板参数决定
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    // 只要传了那个静态变量，不是-1，下边的值就是true
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    // stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T))
    // SlicePerChunk = 2, StepPerSlice = 2
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

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
      #if __CUDA_ARCH__ < 700
        // Yeah, so all that above don't matter a lick on older hardware.
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;

        // if (tid == 1) {
        //   NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d> before assigning src/dst, srcs[0]=%p, srcs[0]@%p, Src=%d, SrcBuf=%d, Input=%d, Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput))=%d", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0], ncclShmem.groups[group].srcs, Src, SrcBuf, Input, Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput)));
        // } else if (tid == 9) {
        //   NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d> before assigning src/dst, srcs[0]=%p, srcs[0]@%p, Dst=%d, DstBuf=%d, Input=%d, Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput))=%d", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0], ncclShmem.groups[group].srcs, Dst, DstBuf, Input, Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput)));
        // }

        if (Src && (flags & (SrcBuf==Input ? RoleInput : RoleOutput))) {
          ncclShmem.groups[group].srcs[0] = userBuff + srcIx + offset; // 传给srcIx形参其实也是个offset

          // NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, userBuff = %p, srcIx = 0x%lx, offset = 0x%x, srcs[0]=%p, srcs[0]@%p", ncclShmem.comm.rank, blockIdx.x, tid, userBuff, srcIx, offset, ncclShmem.groups[group].srcs[0], ncclShmem.groups[group].srcs);
        }
        if (Dst && (flags & (DstBuf==Input ? RoleInput : RoleOutput))) {
          ncclShmem.groups[group].dsts[0] = userBuff + dstIx + offset;
          // NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, userBuff = %p, dstIx = 0x%lx, offset = 0x%x, dsts[0]=%p, dsts[0]@%p, srcs[0]=%p, srcs[0]@%p", ncclShmem.comm.rank, blockIdx.x, tid, userBuff, dstIx, offset, ncclShmem.groups[group].dsts[0], ncclShmem.groups[group].dsts, ncclShmem.groups[group].srcs[0], ncclShmem.groups[group].srcs);
        }
        
        // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, before waitPeer, srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
        // if ((flags & (Recv*RoleWaitRecv))) {
        //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, upper enter waitPeer", ncclShmem.comm.rank, blockIdx.x, tid);
        // }
        // if ((flags & (Send*RoleWaitSend))) {
        //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, upper enter waitPeer", ncclShmem.comm.rank, blockIdx.x, tid);
        // }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(dstIx, remoteIx, offset, sliceSize);

        // if (tid == 1 || tid == 9 || tid == 0) {
        //   NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after waitPeer before subBarrier, srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
        // }
        
        // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after waitPeer before subBarrier, srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
        subBarrier();
        if (tid == 1 || tid == 9) {
          // NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after subBarrier, srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
        }
        // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after subBarrier, srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
        if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]) {

          // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, case 1, DirectRecv = %d, srcs[0]=%p, dsts[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, DirectRecv, ncclShmem.groups[group].srcs[0], ncclShmem.groups[group].dsts[0]);
          
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send) {

            // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, case 1 inner, Send = %d", ncclShmem.comm.rank, blockIdx.x, tid, Send);

            // (1-Send) is only there to avoid compilation errors in case MaxSend=0 (and Send=0).
            ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, (1-Send)+MaxSend, 0>
              (tid, nworkers, nullptr, false,
               1, (T const**)ncclShmem.groups[group].srcs,
               fan.nsend(), (T**)ncclShmem.groups[group].dsts+1,
               sliceSize);

            // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after case 1, srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
          }
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // For broadcast in CollNet to do empty send
          ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1, 0>
            (tid, nworkers, ncclShmem.redOpArgs, postOp,
             Recv, (T const**)ncclShmem.groups[group].srcs,
             Dst, (T**)ncclShmem.groups[group].dsts,
             sliceSize);
        } else {

          // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, case 3, DirectRecv = %d, srcs[0]=%p, dsts[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, DirectRecv, ncclShmem.groups[group].srcs[0], ncclShmem.groups[group].dsts[0]);

          constexpr int PreOpN = SrcBuf != Input ? 0 :
                                 DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          ReduceOrCopyMulti<Unroll, RedOp, T, Recv+Src, Recv*MaxRecv+Src, Send+Dst, Send*MaxSend+Dst, PreOpN>
            (tid, nworkers, ncclShmem.redOpArgs, postOp,
             Recv*fan.nrecv()+Src, (T const**)ncclShmem.groups[group].srcs,
             Send*fan.nsend()+Dst, (T**)ncclShmem.groups[group].dsts,
             sliceSize);

          // NCCL_LOG_RANK_0_THRD_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, after case 3, srcs[0]=%p", ncclShmem.comm.rank, blockIdx.x, tid, ncclShmem.groups[group].srcs[0]);
        }
        barrier(); // This barrier has a counterpart in following loop
        if (Send && (flags & RolePostSend) && index == 0) __threadfence_system();
        __syncwarp();
        postPeer<Recv, Send>();
        offset += sliceSize;
        slice += 1;
        
        // NCCL_LOG_THRD_0(NCCL, "Rank<%d> Blk<%d> Thrd<%d>, offset = %d, sliceSize = %d, nelem = %d, slice = %d, SlicePerChunk = %d", ncclShmem.comm.rank, blockIdx.x, tid, offset, sliceSize, nelem, slice, SlicePerChunk);

      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn(instruction).
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        // if ((flags & (Recv*RoleWaitRecv))) {
        //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, lower enter waitPeer", ncclShmem.comm.rank, blockIdx.x, tid);
        // }
        // if ((flags & (Send*RoleWaitSend))) {
        //   NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, lower enter waitPeer", ncclShmem.comm.rank, blockIdx.x, tid);
        // }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, 0);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      if (Send && (flags & RolePostSend) && sliceSize > 0 && index == 0) __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += sliceSize;
      slice += 1;
    }
  }

  // Scatter/Gather generic op
  // skip: my own rank order in the buffer chunks
  // shift: peer offset to avoid all ranks sending to or receiving from same peer
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      int realSize = max(0, min(dataSize, peerElem-offset));
      if (tid < nworkers) {
        if (Send) {
          // Scatter pre-scales data of input buffer only in non-Direct case
          constexpr int PreOpN = DirectSend ? 0 : 1;
          if (flags & RoleInput) ncclShmem.groups[group].srcs[0] = userBuff + inpIx + offset;
          if (tid == 0) ncclShmem.groups[group].totalSendSize[slice] = 0; // Skip the threadfence
          // realSize is not accurate here; but intra-node does not rely on sizes FIFO
          waitPeer<0, DirectSend, 0, 1, 1, 0>(0, inpIx, offset, realSize);
          subBarrier();
          #pragma unroll
          // Loop over peers
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            int peerOffset = i*peerElem;
            // Skip the data I am responsible of reducing myself
            if (skip >= 0 && i >= skip) peerOffset += peerElem;
            const T* src0 = (T*)ncclShmem.groups[group].srcs[0] + peerOffset;
            int realPeerSize = min(realSize, totalElem-peerOffset);
            if (realPeerSize > 0 && ncclShmem.groups[group].dsts[i] != nullptr) {
              ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1, PreOpN>(tid, nworkers, ncclShmem.redOpArgs, false, 1, &src0, 1, (T**)ncclShmem.groups[group].dsts+i, realPeerSize);
              // Mark for threadfence at the end
              if (tid == 0) ncclShmem.groups[group].totalSendSize[slice] += realPeerSize;
            }
          }
        } else if (Recv) {
          if (flags & RoleOutput) ncclShmem.groups[group].dsts[0] = userBuff + outIx + offset;
          int peerOffset = index*peerElem;
          if (skip >= 0 && index >= skip) peerOffset += peerElem;
          // Adjust remote index with peer offset in case we are directly pulling from peer's output buffer
          waitPeer<DirectRecv, 0, 1, 0, 0, 1>(outIx, outIx+peerOffset, offset, realSize);
          subBarrier();
          if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]) {
            // Since waitPeer sets srcs[0] to output buffer + offset, we are doing a direct-write based recv
            // Do nothing
          } else {
            #pragma unroll
            for (int j=0; j<fan.nrecv(); j++) {
              int i = (j+shift)%fan.nrecv();
              peerOffset = i*peerElem;
              if (skip >= 0 && i >= skip) peerOffset += peerElem;
              T* dst0 = (T*)ncclShmem.groups[group].dsts[0] + peerOffset;
              int realPeerSize = min(realSize, totalElem-peerOffset);
              if (realPeerSize > 0) ReduceOrCopyMulti<Unroll, RedOp, T, 1, 1, 1, 1, 0>(tid, nworkers, ncclShmem.redOpArgs, postOp, 1, (const T**)ncclShmem.groups[group].srcs+i, 1, &dst0, realPeerSize);
            }
          }
        }
      }
      barrier();
      // If we indeed send something, threadfence
      if (Send && (flags & RolePostSend) && ncclShmem.groups[group].totalSendSize[slice] > 0 && index == 0)
        __threadfence_system();
      __syncwarp();
      postPeer<Recv, Send>();
      offset += realSize;
    }
  }

  // connIndex = group >> 16 = 0, e = nullptr，通过打log确认了，打log也确认了，经过loadRecvConn和loadSendConn，flags都没变。
  // 所以这两个函数的效果，就是设置了step，设置了connStepPtr，设置了connStepCache，设置了connEltsFifo，设置了ncclShmem里的peer的conn
  // SlicePerChunk=2, StepPerSlice=2
  __device__ __forceinline__ void loadRecvConn(ncclPeer *peer, int connIndex, struct ncclWorkElem* e) {
    if (flags & (RoleWaitRecv|RolePostRecv)) {
      auto *conn = &peer->recv[connIndex].conn;
      step = conn->step; // uint64_t step;      // Keep where we are // 设置好就不会变了；每个线程有自己的值
      step = roundUp(step, SlicePerChunk*StepPerSlice); // return (x+y-1) - (x+y-1)%y;，就是向上取到y的倍数
      if (flags & RolePostRecv) { // (ng - 2) * 8 + 0 号线程是 RolePostRecv
        connStepPtr = conn->head; // uint64_t *head;     // Local for send, remote for recv
        *connStepPtr = step; // Return credits in case we rounded up.
      }
      if (flags & RoleWaitRecv) { // 0 * 8 + 0 号线程是 RoleWaitRecv
        ncclShmem.groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
        connStepPtr = conn->tail; // uint64_t *tail;     // Local for recv, remote for send
        connStepCache = *connStepPtr; // 这个应该就是simple协议的标记位，这个被设置了，就代表buffer里是新数据了。对于recv来说，就代表收到了新数据
        flags |= (conn->offsFifo != nullptr) ? OffsFifoEnabled : 0; // 这个和proxy相关，先不关注。int *offsFifo;      // Buffer fifo from proxy to GPU，所以对GPU是recv

        //  NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitRecv>, conn->offsFifo = %p", ncclShmem.comm.rank, blockIdx.x, tid, conn->offsFifo);

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
        // OFCCL_LOG(OFCCL, "recv Rank<%d>, Blk<%d>, Thrd<%d>, flags=0x%X, conn->direct=%d, step=%llu, connStepCache=%llu", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, flags, conn->direct, step, connStepCache);
      }
      // int g = tid / 8; // 当前线程属于哪个 “g”
      // int ng = nthreads / 8; // 一共有几个 “g”
      // OFCCL_LOG(OFCCL, "Blk<%d>, Thrd<%d>, ng=%d, g=%d, connIndex=%d, e @ %p, flags=0x%X", blockIdx.x, threadIdx.x, ng, g, connIndex, e, flags);
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclPeer *peer, int connIndex, struct ncclWorkElem* e) {
    if (flags & (RoleWaitSend|RolePostSend)) {
      auto *conn = &peer->send[connIndex].conn;
      step = conn->step;
      step = roundUp(step, SlicePerChunk*StepPerSlice);
      if (flags & RolePostSend) {
        connStepPtr = conn->tail;
      }
      if (flags & RoleWaitSend) {
        ncclShmem.groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
        connStepPtr = conn->head;
        connStepCache = *connStepPtr;
        flags |= (conn->offsFifo != nullptr) ? OffsFifoEnabled : 0;

        // NCCL_LOG_RANK_0(OFCCL_MPI, "Rank<%d> Blk<%d> Thrd<%d-RoleWaitSend>, conn->offsFifo = %p, conn->sizesFifo = %p", ncclShmem.comm.rank, blockIdx.x, tid, conn->offsFifo, conn->sizesFifo);

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
        // OFCCL_LOG(OFCCL, "send Rank<%d>, Blk<%d>, Thrd<%d>, flags=0x%X, step=%llu, connStepCache=%llu", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, flags, step, connStepCache);
      }
      // int g = tid / 8; // 当前线程属于哪个 “g”
      // int ng = nthreads / 8; // 一共有几个 “g”
      // OFCCL_LOG(OFCCL, "Blk<%d>, Thrd<%d>, ng=%d, g=%d, connIndex=%d, e @ %p, flags=0x%X", blockIdx.x, threadIdx.x, ng, g, connIndex, e, flags);
    }
  }

 public:
  // 模板参数：Ring AllReduce里边，MaxSend和MaxRecv都是 1，Direct = 1
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint32_t group=0, struct ncclWorkElem* e = nullptr
    ):
    tid(tid),
    stepSize(ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T)) {

    // For send operations, we need an extra warp to overlap the threadfence and the copy
    this->nthreads = nthreads;
    this->nworkers = nthreads - (MaxSend > 0 && nthreads-WARP_SIZE >= 64 ? WARP_SIZE : 0); // 这一行的意思是，nthreads足够大的话，就可以剔除出一个warp了，剩下的是worker。
    this->group = group & (uint16_t)0xFFFF; // 还是 0. ring的情况下，group一直是0。
    int connIndex = group >> 16; // 还是 0.

    int nrecv=0, nsend=0; // 后边两行代码之后，变成nrecv=1, nsend=1
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
    
    // if (flags & RoleWaitRecv) {
    //   OFCCL_LOG(OFCCL, "Rank<%d>, Blk<%d>, Thrd<%d>, flags=0x%X, invoke loadRecvConn", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, flags);
    // }
    loadRecvConn(&ncclShmem.channel.devPeers[peer], connIndex, e); // 没有RoleWaitRecv或RolePostRecv标记的线程直接返回。
    loadSendConn(&ncclShmem.channel.devPeers[peer], connIndex, e); // 没有RoleWaitSend或RolePostSend标记的线程直接返回。

    // inputBuf = sendbuff, outputBuf = recvbuff
    setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclWorkElemReg*)e); // 事实上也是需要被指定了flag的线程才会进行相应的工作

    // 初步观察了下，genericOp里边所有thrd都有活干——上边的load、set函数都是在操作shmem，单个thrd操作，然后真正执行的时候，大家都利用shmem里的信息工作
  }

  __device__ ~Primitives() {
    // Ensure ncclShmem.groups[].send/recvConns are available
    if (!(flags & ThreadsSynced))
      barrier();
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) {
      auto *conns = (flags & RolePostSend) ? ncclShmem.groups[group].sendConns : ncclShmem.groups[group].recvConns;
      conns[index]->step = step;
    }
    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem.groups[group]
    barrier();
  }

  // inputBuf = sendbuff, outputBuf = recvbuff
  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, uint64_t redOpArg, struct ncclWorkElemReg* e) {
    if (flags & RoleInput) { // 0 * 8 + 1 号线程是 RoleInput(0x08) 1
      userBuff = (T*)inputBuf;
      ncclShmem.redOpArgs[0] = redOpArg;  // scaler for local input
    }
    if (flags & RoleOutput) userBuff = (T*)outputBuf; // 1 * 8 + 1 号线程是 RoleOutput 9
    bool recvProvider = flags == (flags|RoleWaitRecv|DirectWrite);
    bool sendAcceptor = flags == (flags|RoleWaitSend|DirectWrite);
    bool sendProvider = flags == (flags|RoleWaitSend|DirectRead); // sender provides direct buffer (to be fetched)
    bool recvAcceptor = flags == (flags|RoleWaitRecv|DirectRead); // receiver accepts direct buffer
    int regUsed = e != nullptr ? e->elem.regUsed : 0;
    // 打log 显示上边5个变量全是0

    // if (!(recvProvider == 0 && sendAcceptor == 0 && sendProvider == 0 && recvAcceptor == 0 && regUsed == 0)) {
    //   OFCCL_LOG(OFCCL, "Rank<%d>, Blk<%d>, Thrd<%d>, recvProvider=%d, sendAcceptor=%d, sendProvider=%d, recvAcceptor=%d, regUsed=%d", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, recvProvider, sendAcceptor, sendProvider, recvAcceptor, regUsed);
    // }

    // 模板参数 Direct 是 1
    if (Direct && recvProvider) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
      // Wait for consumer to consume previous value before trampling it.
      while (*slot != nullptr && !checkAbort(spins));
      directBuff = (T*)outputBuf;
      // Encode pointer by XOR'ing against some address they definitely wouldn't send
      // since we want to allow them sending us nullptr while not colliding with
      // the empty slot value.
      *slot = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(directBuff) ^ reinterpret_cast<uintptr_t>(slot));
      
      // NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, encode directBuff=%p to *(recvConns[index]->ptrExchange)=%p, in recvConns[index]->ptrExchange@%p", ncclShmem.comm.rank, blockIdx.x, tid, directBuff, *slot, slot);
    }
    if (Direct && sendAcceptor) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
      void *ptr;
      while (true) {
        ptr = *slot;
        if (ptr != nullptr || checkAbort(spins)) break;
      }
      directBuff = regUsed ? (T*)(e->dnOutputs[index]) :
                   reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr) ^ reinterpret_cast<uintptr_t>(slot));

      // NCCL_LOG_RANK_0(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, init *(sendConns[index]->ptrExchange)=%p with peer, directBuff=%p, sendConns[index]->ptrExchange@%p", ncclShmem.comm.rank, blockIdx.x, tid, ptr, directBuff, ncclShmem.groups[group].sendConns[index]->ptrExchange);

      *slot = nullptr;
    }
    if (Direct && sendProvider) {
      int spins = 0;
      void *volatile *slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
      volatile uint64_t* argSlot0 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange;
      volatile uint64_t* argSlot1 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange+1;
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
      void *volatile *slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
      volatile uint64_t* argSlot0 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange;
      volatile uint64_t* argSlot1 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange+1;
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
        ncclShmem.redOpArgs[1+index] = ((arg1 & 0xffffffff)<<32) | (arg0 & 0xffffffff);
      }
      *argSlot0 = 0; *argSlot1 = 0;
      *slot = nullptr;
    }
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

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, int totalElem, int peerElem, int skip, int shift) { // collnet里使用
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void
  directScatter(intptr_t inpIx, int totalElem, int peerElem, int skip, int shift) { // collnet里使用
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, int totalElem, int peerElem, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, skip, shift, postOp);
  }
  __device__ __forceinline__ void
  directGather(intptr_t outIx, int totalElem, int peerElem, int skip, int shift) {  // collnet里使用
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, skip, shift, /*postOp=*/false);
  }
};
