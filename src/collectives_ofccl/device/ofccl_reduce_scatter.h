#include "devcomm.h"
#include "collectives_ofccl.h"
#include "ofccl_primitives.h"
#include "debug.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->header.nWarps*WARP_SIZE;
    const int bid = blockIdx.x;
    const int nChannels = args->nChannels;
    const int currUsedSlotId = blkStatus.currLoadedCollId % NUM_SHMEM_SLOT;

    const int *ringRanks = sharedCollCtx[currUsedSlotId].staticCollCtx.ringRanks;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCESCATTER_CHUNKSTEPS : 1));
    // // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    // const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2);
    const int nranks = sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks;
    const ssize_t loopSize = nChannels*chunkSize;
    const ssize_t size = args->count; // recv count

    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &sharedCollCtx[currUsedSlotId].staticCollCtx.ringPrev, &sharedCollCtx[currUsedSlotId].staticCollCtx.ringNext, args->sendbuff, args->recvbuff, args->redOpArg);
    ssize_t offset = 0;
    int nelem = 0;
    int currentStep = 0;
    ssize_t gridOffset = 0;
    int rankDest;

    for (gridOffset = sharedCollCtx[currUsedSlotId].dynamicCollCtx.gridOffset4RunRing; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      // else if (Proto::Id == NCCL_PROTO_LL)
      //   realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
      // else if (Proto::Id == NCCL_PROTO_LL128)
      //   realChunkSize = min(divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128, chunkSize);
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + bid*int(realChunkSize);

      // 这里不能直接赋值，因为这是在循环里，在恢复的上下文中的gridOffset对应的循环中，需要恢复，否则直接用0初始化。
      if (gridOffset == sharedCollCtx[currUsedSlotId].dynamicCollCtx.gridOffset4RunRing) {
        currentStep = sharedCollCtx[currUsedSlotId].dynamicCollCtx.currentStep4RunRing;
      } else {
        currentStep = 0;
      }

      /////////////// begin ReduceScatter steps 共nranks步///////////////
      nelem = min(realChunkSize, size-chunkOffset);

      // step 0: push data to next GPU
      if (currentStep < 1) {
        rankDest = ringRanks[nranks-1];
        offset = chunkOffset + rankDest * size;
        prims.send(offset, nelem);
        if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
          goto run_ring_end;
        }
        currentStep++;
      }

      // k-2 steps: reduce and copy to next GPU
      if (currentStep < nranks - 1) { // 2卡不执行这里。
        for (int j=currentStep + 1; j<nranks; ++j) { // j需要根据currentStep进行相应调整。原来j初值是2. 第一次进入时，currentStep=1, j=2
          rankDest = ringRanks[nranks-j];
          offset = chunkOffset + rankDest * size;
          prims.recvReduceSend(offset, nelem);

          if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
            goto run_ring_end;
          }
          currentStep++;
        }
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      if (currentStep < nranks){
        rankDest = ringRanks[0];
        offset = chunkOffset + rankDest * size;
        prims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
        if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
          goto run_ring_end;
        }
        currentStep++;
      }
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
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL128>(args);
  }
};