#include "common_ofccl.h"
#include "devcomm.h"
#include "collectives_ofccl.h"
#include "ofccl_primitives.h"
#include "debug.h"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(ncclWorkElem *args) {
    const int tid = threadIdx.x;
    const int nthreads = args->header.nWarps*WARP_SIZE;
    const int bid = blockIdx.x; // TODO: 可以修复一下args->bid;
    const int nChannels = args->nChannels;
    // ncclRing *ring = &sharedCollCtx.channel.ring;
    int ringIx = sharedCollCtx.ringIndex;
    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = sharedCollCtx.nRanks;
    const ssize_t loopSize = nChannels*nranks*chunkSize; // 没有办法按照应用的buff的大小来切分chunk，而是需要从硬件的角度去指定chunkSize。所以可能要运行多次逻辑上的ringAllReduce操作。
    const ssize_t size = args->count;

    *(blkStatus.barrierCnt + 0 + 14 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

    // TODO: minChunkSize 是LL和LL128用的，先省略

    // TODO: ofccl/src/include/devcomm.h中ncclRing的定义里，int prev和int next两个成员变量的定位是Shortcuts for userRanks[1] and userRanks[n-1]。不过我们直接把int取出来存着应该没问题。
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &sharedCollCtx.ringPrev, &sharedCollCtx.ringNext, args->sendbuff, args->recvbuff, args->redOpArg);
    
    // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> create prims, gridOffset=%ld, size = %ld, currentStep = %d, maxStep = %d", sharedCollCtx.rank, blockIdx.x, tid, sharedCollCtx.gridOffset4RingAllReduce, size, sharedCollCtx.currentStep4RingAllReduce, 2 * nranks - 1);
    // __syncwarp(); // ！！！！！！为了打印log加的！！！！

    ssize_t offset = 0;
    int nelem = 0;
    int chunk = 0;
    int currentStep = 0;
    ssize_t gridOffset = 0;

    for (gridOffset = sharedCollCtx.gridOffset4RingAllReduce; gridOffset < size; gridOffset += loopSize) {
      
      // 初步恢复执行后，这段计算realChunkSize的逻辑还是保留吧
      ssize_t realChunkSize;
      if (Proto::Id == NCCL_PROTO_SIMPLE) {
        realChunkSize = min(chunkSize, divUp(size-gridOffset, nChannels*nranks));
        realChunkSize = roundUp(realChunkSize, (nthreads-WARP_SIZE)*sizeof(uint64_t)/sizeof(T)); // 常数。
      }
      // TODO: 先忽略else的其他proto

      realChunkSize = int(realChunkSize);

      auto calcOffset = [&]__device__(int chunk)->ssize_t {
        if (Proto::Id == NCCL_PROTO_SIMPLE)
          return gridOffset + bid*nranks*realChunkSize + chunk*realChunkSize;
        else
          return gridOffset + (chunk*nChannels + bid)*realChunkSize;
      };
      auto modRanks = [&]__device__(int r)->int {
        return r - (r >= nranks ? nranks : 0);
      };

      // 这里不能直接赋值，因为这是在循环里，在恢复的上下文中的gridOffset对应的循环中，需要恢复，否则直接用0初始化。
      if (gridOffset == sharedCollCtx.gridOffset4RingAllReduce) {
        currentStep = sharedCollCtx.currentStep4RingAllReduce;
      } else {
        currentStep = 0;
      }

      if (currentStep < 1) {
        ofcclBarrier(11, nthreads);;
        chunk = modRanks(ringIx + nranks-1);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);
        
        // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.send. [< 1] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep, offset, nelem, chunk);
        // __syncwarp(); // ！！！！！！为了打印log加的！！！！

        prims.send(offset, nelem); // **send** 将 sendbuff 中的数据通过 sendConn 发送给 peer
        // threadfence已经在genericOp里边做过了，这里不需要了，可以直接读
        if (sharedCollCtx.saveCtx7Quit == 1) {
          // if (tid == 0) {
          //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prims.send quit. [< 1] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, gridOffset, currentStep, offset, nelem, chunk);
          // }
          // __syncwarp(); // ！！！！！！为了打印log加的！！！！
          
          goto run_ring_end;
        }
        currentStep++;
      }

      // k-2 steps: reduce and copy to next GPU
      if (currentStep < nranks - 1) { // 2卡不执行这里。
        ofcclBarrier(11, nthreads);;
        for (int j=currentStep + 1; j<nranks; ++j) { // j需要根据currentStep进行相应调整。原来j初值是2.
          chunk = modRanks(ringIx + nranks-j);
          offset = calcOffset(chunk);
          nelem = min(realChunkSize, size-offset);

          // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.recvReduceSend. [< nranks - 1] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep, offset, nelem, chunk);
          // __syncwarp(); // ！！！！！！为了打印log加的！！！！

          prims.recvReduceSend(offset, nelem); // **recvReduceSend** 通过 recvConn 接收 peer 发送的数据，和 sendbuff 的数据进行 reduce 后通过  sendConn 发送给 peer 
          if (sharedCollCtx.saveCtx7Quit == 1) {
            // if (tid == 0) {
            //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prims.recvReduceSend quit. [< nranks - 1] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, gridOffset, currentStep, offset, nelem, chunk);
            // }
            // __syncwarp(); // ！！！！！！为了打印log加的！！！！

            goto run_ring_end;
          }
          currentStep++;
        }
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      if (currentStep < nranks){
        ofcclBarrier(11, nthreads);;
        chunk = ringIx + 0;
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);

        // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directRecvReduceCopySend. [< nranks] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep, offset, nelem, chunk);
        // __syncwarp(); // ！！！！！！为了打印log加的！！！！

        prims.directRecvReduceCopySend(offset, offset, offset, nelem, /*postOp=*/true); // **directRecvReduceCopySend** 通过 recvConn 接收 peer 发送的数据，和 sendbuff 的数据进行 reduce 后 copy 到 recvbuff，并通过 P2P write 写入到 peer 的 recvbuff，direct主要是修饰send，意思要直接写入peer的 recvbuff
        if (sharedCollCtx.saveCtx7Quit == 1) {
          // if (tid == 0) {
          //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prims.directRecvReduceCopySend quit. [< nranks] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, gridOffset, currentStep, offset, nelem, chunk);
          // }
          // __syncwarp(); // ！！！！！！为了打印log加的！！！！

          goto run_ring_end;          
        }
        currentStep++;
      }

      // k-2 steps: copy to next GPU
      if (currentStep < 2 * nranks - 2) { // 2卡不执行这里
        ofcclBarrier(11, nthreads);;
        for (int j=currentStep-nranks+1; j<nranks-1; ++j) { // j需要根据currentStep进行相应调整。原来j初值是1. 第一次进入时，currentStep=nranks, j=1
          chunk = modRanks(ringIx + nranks-j);
          offset = calcOffset(chunk);
          nelem = min(realChunkSize, size-offset);

          // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directRecvCopySend. [< 2 * nranks - 2] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep, offset, nelem, chunk);
          // __syncwarp(); // ！！！！！！为了打印log加的！！！！

          prims.directRecvCopySend(offset, offset, nelem); // **directRecvCopySend** 被动操作，数据已经被 peer 直接写入到 ==recvbuff==，copy 也无需发生，并将数据通过 P2P write 写入到 peer 的 recvbuff
          if (sharedCollCtx.saveCtx7Quit == 1) {
            // if (tid == 0) {
            //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prims.directRecvCopySend quit. [< 2 * nranks - 2] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, gridOffset, currentStep, offset, nelem, chunk);
            // }
            // __syncwarp(); // ！！！！！！为了打印log加的！！！！

            goto run_ring_end;
          }
          currentStep++;
        }
      }

      // Make final copy from buffer to dest.
      if (currentStep < 2 * nranks - 1) {
        ofcclBarrier(11, nthreads);;
        chunk = modRanks(ringIx + 1);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size-offset);

        // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, before prims.directRecv. [< 2 * nranks - 1] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, currentStep, offset, nelem, chunk);
        // __syncwarp(); // ！！！！！！为了打印log加的！！！！

        prims.directRecv(offset, nelem); // **directRecv** 被动操作，数据已经被 peer 直接写入到 recvbuff
        if (sharedCollCtx.saveCtx7Quit == 1) {
          // if (tid == 0) {
          //   OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, prims.directRecv quit. [< 2 * nranks - 1] gridOffset = %ld, currentStep = %d, offset = %ld, nelem = %d, chunk = %d", sharedCollCtx.rank, blockIdx.x, tid, gridOffset, currentStep, offset, nelem, chunk);
          // }
          // __syncwarp(); // ！！！！！！为了打印log加的！！！！

          goto run_ring_end;
        }
        currentStep++;
      }
      // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, gridOffset = %ld, size = %ld, realChunkSize = %ld, chunk = %d, offset = %ld, nelem = %d", sharedCollCtx.rank, blockIdx.x, tid, gridOffset, size, realChunkSize, chunk, offset, nelem);
      // __syncwarp(); // ！！！！！！为了打印log加的！！！！
    }
  run_ring_end:
    ofcclBarrier(11, nthreads);;
    if (tid == 0) {
      if (sharedCollCtx.saveCtx7Quit == 1) {
        blkStatus.collStatus[blkStatus.currLoadedCollId] = -1;
        // 说明是跑到一半要退出了，保存上下文
        sharedCollCtx.currentStep4RingAllReduce = currentStep;
        sharedCollCtx.gridOffset4RingAllReduce = gridOffset;

        // OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, runRing saveCtx&Quit, gridOffset = %lu, currentStep = %d", sharedCollCtx.rank, blockIdx.x, tid, gridOffset, currentStep);
        // __syncwarp(); // ！！！！！！为了打印log加的！！！！
      } else {
        blkStatus.collStatus[blkStatus.currLoadedCollId] = 2;
      //   OFCCL_LOG_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, coll_id = %d, runRing success, gridOffset = %lu, size = %lu, currentStep = %d, loopSize = %ld", sharedCollCtx.rank, blockIdx.x, tid, blkStatus.currLoadedCollId, gridOffset, size, currentStep, loopSize);
      //   __syncwarp(); // ！！！！！！为了打印log加的！！！！
      }
      sharedCollCtx.saveCtx7Quit = 0; // 重置。
    }
    
    *(blkStatus.barrierCnt + 1 + 14 * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

    OFCCL_LOG_WARP_HEAD(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> leave runRing", sharedCollCtx.rank, blockIdx.x, tid);
  }


  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeUpDown(ncclWorkElem *args) {
  }
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runTreeSplit(ncclWorkElem *args) {
  }
}

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    runRing<T, RedOp, Proto>(args);
    // OFCCL_LOG_BLK_0_THRD_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d>, runRing return", sharedCollCtx.rank, blockIdx.x, threadIdx.x);

  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // #if CUDART_VERSION >= 11020 && CUDART_VERSION < 11040 && __CUDA_ARCH__ >= 800
    //   runTreeUpDown<T, RedOp, ProtoSimple<1, 1>>(args);
    // #else
    //   runTreeSplit<T, RedOp, ProtoSimple<1, 1>>(args);
    // #endif
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_COLLNET, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runTreeSplit<T, RedOp, ProtoLL>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // runRing<T, RedOp, ProtoLL128>(args);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_TREE, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    // int print = 0;
    // if (!print) {
    //   OFCCL_LOG1(OFCCL, "RunWorkElement AllReduce, Tree, LL128");
    //   print = 1;
    // }
    // runTreeSplit<T, RedOp, ProtoLL128>(args);
  }
};
