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
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLGATHER_CHUNKSTEPS : 1));
    // // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
    // const ssize_t minChunkSizeLL128 = int(nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2);
    const int nranks = sharedCollCtx[currUsedSlotId].staticCollCtx.nRanks;
    const ssize_t loopSize = nChannels*int(chunkSize);
    const ssize_t size = args->count; // send count
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 0 + 14 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif

    T *inputBuf = (T*)args->sendbuff;
    T *outputBuf = (T*)args->recvbuff;
    
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 0 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &sharedCollCtx[currUsedSlotId].staticCollCtx.ringPrev, &sharedCollCtx[currUsedSlotId].staticCollCtx.ringNext, inputBuf, outputBuf, args->redOpArg);
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 1 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif

    ssize_t offset = 0;
    int nelem = 0;
    int currentStep = 0;
    ssize_t gridOffset = 0;
    int rankDest;

    for (gridOffset = sharedCollCtx[currUsedSlotId].dynamicCollCtx.gridOffset4RunRing; gridOffset < size; gridOffset += loopSize) {
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset,nChannels));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T));
      }
      // else if (Proto::Id == NCCL_PROTO_LL)
      //   realChunkSize = size-gridOffset < loopSize ? args->lastChunkSize : chunkSize;
      // else if (Proto::Id == NCCL_PROTO_LL128)
      //   realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*minChunkSizeLL128)*minChunkSizeLL128);
      realChunkSize = int(realChunkSize);

      ssize_t chunkOffset = gridOffset + int(bid*realChunkSize);

      // 这里不能直接赋值，因为这是在循环里，在恢复的上下文中的gridOffset对应的循环中，需要恢复，否则直接用0初始化。
      if (gridOffset == sharedCollCtx[currUsedSlotId].dynamicCollCtx.gridOffset4RunRing) {
        currentStep = sharedCollCtx[currUsedSlotId].dynamicCollCtx.currentStep4RunRing;
      } else {
        currentStep = 0;
      }

      /////////////// begin AllGather steps, 共nranks步 ///////////////
      nelem = min(realChunkSize, size-chunkOffset);

      // step 0: push data to next GPU
      if (currentStep < 1) {
        rankDest = ringRanks[0];
        offset = chunkOffset + rankDest * size;

        if (inputBuf + chunkOffset == outputBuf + offset) { // In place
          // OFCCL_LOG_WARP_HEAD(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directSend. gridOffset=%ld, currentStep=%d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep);
          prims.directSend(chunkOffset, offset, nelem);
        } else {
          // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directCopySend. gridOffset=%ld, currentStep=%d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep);
          // OFCCL_LOG_WARP_HEAD(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directCopySend", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 2 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

          prims.directCopySend(chunkOffset, offset, offset, nelem);

          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 3 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif
        }
        if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
          goto run_ring_end;
        }
        currentStep++;
      }

      // k-2 steps: copy to next GPU
      if (currentStep < nranks - 1) { // 2卡不执行这里。
        for (int j=currentStep; j<nranks-1; ++j) { // j需要根据currentStep进行相应调整。原来j初值是1. 第一次进入时，currentStep=1, j=1
          rankDest = ringRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directRecvCopySend. gridOffset=%ld, currentStep=%d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep);
          // OFCCL_LOG_WARP_HEAD(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directRecvCopySend", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 4 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif
          prims.directRecvCopySend(offset, offset, nelem);
          #ifdef ARRAY_DEBUG
            *(blkStatus.barrierCnt + 5 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
          #endif

          if (sharedCollCtx[currUsedSlotId].saveCtx7Quit == 1) {
            goto run_ring_end;
          }
          currentStep++;
        }
      }

      // Make final copy from buffer to dest.
      if (currentStep < nranks){
        rankDest = ringRanks[1];
        offset = chunkOffset + rankDest * size;

        // Final wait/copy.
        // OFCCL_LOG_RANK_0_WARP_HEAD_SHMEM(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directRecv. gridOffset=%ld, currentStep=%d", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep);
        // OFCCL_LOG_WARP_HEAD(OFCCL_P2P, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directRecv.", sharedCollCtx[currUsedSlotId].staticCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 6 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
        prims.directRecv(offset, nelem);
        #ifdef ARRAY_DEBUG
          *(blkStatus.barrierCnt + 7 + 18 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
        #endif
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
    
    #ifdef ARRAY_DEBUG
      *(blkStatus.barrierCnt + 1 + 14 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
    #endif
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllGather, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL128>(args);
  }
};
