#include "devcomm.h"
#include "collectives_ofccl.h"
#include "ofccl_primitives.h"
#include "debug.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->header.nWarps*WARP_SIZE;
    const int bid = args->bid;
    const int nChannels = args->nChannels;
    const int currUsedSlotId = blkStatus.currLoadedCollId % NUM_SHMEM_SLOT;

    const int *ringRanks = sharedCollCtx[currUsedSlotId].staticCollCtx.ringRanks;

    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? BROADCAST_CHUNKSTEPS : 1));
    // const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T)));
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->count;
    const int rank = ringRanks[0];
    const int nextRank = ringRanks[1];
    const int root = args->root;

    T *inputBuf = (T*)args->sendbuff;
    T *outputBuf = (T*)args->recvbuff;
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &sharedCollCtx[currUsedSlotId].staticCollCtx.ringPrev, &sharedCollCtx[currUsedSlotId].staticCollCtx.ringNext, inputBuf, outputBuf, args->redOpArg);

    int currentStep = 0;
    ssize_t gridOffset = 0;

    for (gridOffset = sharedCollCtx[currUsedSlotId].dynamicCollCtx.gridOffset4RunRing; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      // else if (Proto::Id == NCCL_PROTO_LL)
      //   realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
      // else if (Proto::Id == NCCL_PROTO_LL128)
      //   realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128);
      realChunkSize = int(realChunkSize);

      ssize_t offset = gridOffset + int(bid*realChunkSize);
      int nelem = min(realChunkSize, size-offset);

      // 这里不能直接赋值，因为这是在循环里，在恢复的上下文中的gridOffset对应的循环中，需要恢复，否则直接用0初始化。
      if (gridOffset == sharedCollCtx[currUsedSlotId].dynamicCollCtx.gridOffset4RunRing) {
        currentStep = sharedCollCtx[currUsedSlotId].dynamicCollCtx.currentStep4RunRing;
      } else {
        currentStep = 0;
      }

      if (rank == root) {
        if (inputBuf == outputBuf) {
          prims.send(offset, nelem);
        } else {
          prims.copySend(offset, offset, nelem);
        }
      } else if (nextRank == root) { // 叶子
        prims.recv(offset, nelem);
      } else {
        prims.recvCopySend(offset, nelem);
      }
      if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
        goto run_ring_end;
      }
      currentStep++;
    }
  run_ring_end:
    if (tid == 0) {
      if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
        blkStatus.collStatusAlign.collStatus[blkStatus.currLoadedCollId] = -2;
        // 说明是跑到一半要退出了，保存上下文
        sharedCollCtx[currUsedSlotId].dynamicCollCtx.currentStep4RunRing = currentStep;
        sharedCollCtx[currUsedSlotId].dynamicCollCtx.gridOffset4RunRing = gridOffset;

        if (sharedCollCtx[currUsedSlotId].progressed == 1) { // 不需要在下边完成的情况下判断是否progress。
          blkStatus.collStatusAlign.collStatus[blkStatus.currLoadedCollId] = -1;
        }
      } else {
        blkStatus.collStatusAlign.collStatus[blkStatus.currLoadedCollId] = 2;
      }
    }
  }
}


template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<BROADCAST_CHUNKSTEPS/BROADCAST_SLICESTEPS, BROADCAST_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncBroadcast, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL128>(args);
  }
};