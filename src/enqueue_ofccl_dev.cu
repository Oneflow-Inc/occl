#include "enqueue_ofccl.h"
#include "collectives_ofccl/device/op128_ofccl.h"

static __shared__ int quit;

// Copy src to dst and fill extra size with zeroes
template<typename Tdst, typename Tsrc>
__device__ void copyToShmem(Tdst *dst, Tsrc const *src) { // nccl的这个的函数签名里有个nthreads参数，但是并没有用，应该是为了和下边那个作区分，现在我们可以区分开了，反而带上nthreads是区分不开的。、
  int tid = threadIdx.x;
  static_assert(sizeof(Tdst)%(2*sizeof(uint64_t)) == 0 && sizeof(Tsrc)%(2*sizeof(uint64_t)) == 0,
      "copyToShmem needs sizes which are multiple of 16B");
  static_assert(sizeof(Tdst) >= sizeof(Tsrc), "Tdst size is too small");
  static_assert(sizeof(Tdst) <= WARP_SIZE*2*sizeof(uint64_t), "copyToShmem limited to 512B to make sure it can always be done in one cycle");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  uint64_t *shmemPtr = shmemCvtPtr_ofccl(d);
  int offset = 2*tid;
  uint64_t v0, v1;
  if (offset >= sizeof(Tsrc)/sizeof(uint64_t)) {
    v0 = v1 = 0ULL;
  } else {
    v0 = s[offset] ; v1 = s[offset+1];
  }
  if (offset < sizeof(Tdst)/sizeof(uint64_t)) storeShmem128_ofccl(shmemPtr+offset, v0, v1);
}

template<typename T>
__device__ int copyToShmem(T *dst, T const *src, int tid, int nthreads, int turn=0) {
  static_assert(sizeof(uint64_t) <= alignof(T), "Uhoh");
  uint64_t *d = reinterpret_cast<uint64_t*>(dst);
  uint64_t const *s = reinterpret_cast<uint64_t const*>(src);
  int t = tid - turn;
  if (t < 0) t += nthreads;
  int n = sizeof(T)/sizeof(uint64_t);

  int delta = (n + WARP_SIZE-1) & -WARP_SIZE; // round up to warp lane 0
  if (delta < nthreads) {
    turn += delta;
    if (turn >= nthreads) turn -= nthreads;
  }
  else
    turn = 0;

  n -= t;
  d += t;
  s += t;
  #pragma unroll // 指示要循环展开。
  for (int i=0; i < divUp(sizeof(T), WARP_SIZE*sizeof(uint64_t)); i++) {
    if (n > 0) {
      *d = *s;
      d += nthreads;
      s += nthreads;
      n -= nthreads;
    }
  }
  return turn;
}

static __device__ void ofcclRedopPtrDeref(struct ncclWorkElem* we) {
  if (we->header.type != ncclWorkTypeUnused && we->redOpArgIsPtr) {
    /* redOpArg is a pointer to the scalar value, so we'll dereference it
     * here so that redOpArg holds the bits of the scalar going forward.
     * The tricky thing is we don't know its type T since that's encoded in
     * the funcIndex. Because it would be difficult to get sizeof(T) from
     * funcIndex, we'll cheat and just dereference the largest possible size
     * given the alignment of the pointer. We might be reading in more bytes
     * than we need but that's harmless.
     */
    if (we->redOpArg%2 != 0)
      we->redOpArg = *reinterpret_cast<uint8_t*>(we->redOpArg);
    else if (we->redOpArg%4 != 0)
      we->redOpArg = *reinterpret_cast<uint16_t*>(we->redOpArg);
    else if (we->redOpArg%8 != 0)
      we->redOpArg = *reinterpret_cast<uint32_t*>(we->redOpArg);
    else
      we->redOpArg = *reinterpret_cast<uint64_t*>(we->redOpArg);
  }
}

__device__ int sqRead(SQ *sq, unsigned long long int sqReadFrontier, SQE *target, int *globalBlkCount4Coll, int thrdCudaDev) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int sqeCollId;
  // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> enter sqRead, sqHead=%llu, sqTail=%llu, empty=%d, RingBuffer_get(sq, sqReadFrontier)->counter=%d, RingBuffer_get(sq, sqReadFrontier)->collId=%d, RingBuffer_get(sq, sqReadFrontier)->quit=%d, RingBuffer_get(sq, sqReadFrontier)->logicHead=%d, GetLogicFrontier(sq, sqReadFrontier)=%llu", thrdCudaDev, bid, tid, RingBuffer_logic_head(sq), RingBuffer_logic_tail(sq), RingBuffer_empty(sq), RingBuffer_get(sq, sqReadFrontier)->counter, RingBuffer_get(sq, sqReadFrontier)->collId, RingBuffer_get(sq, sqReadFrontier)->quit, RingBuffer_get(sq, sqReadFrontier)->logicHead, GetLogicFrontier(sq, sqReadFrontier));
  if (RingBuffer_empty(sq)) {
    return -1;
  }
  // 先读过来，然后再判断，最后更新状态：sqe->counter; 以及在恰当的时候commit read
  *target = *RingBuffer_get(sq, sqReadFrontier);
  if (target->quit) {
    // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Get quit", thrdCudaDev, bid, tid);
    return 0;
  }

  // 先判断一下相应的collId是不是该自己的bid处理，不该自己处理直接返回-1
  sqeCollId = target->collId;
  // OFCCL_LOG(OFCCL, "Blk<%d>, Thrd<%d> globalBlkCount4Coll[%d]=%d", thrdCudaDev, bid, tid, sqeCollId, globalBlkCount4Coll[sqeCollId]);
  if (bid >= globalBlkCount4Coll[sqeCollId]) {
    return -1;
  } else {
    // 自己读到之后，更新相应的counter；至于读到的sqe对应的collId是不是该自己处理，是caller的事。
    // 如果发现自己读完之后，所有block都读了，那么commit read
    // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> PREPARE to increase counter(curr=%d) for sqe of collId %d", thrdCudaDev, bid, tid, RingBuffer_get(sq, sqReadFrontier)->counter, sqeCollId);
    int old_counter = atomicAdd(&(RingBuffer_get(sq, sqReadFrontier)->counter), 1);
    // RingBuffer_get(sq, sqReadFrontier)->counter += 1;
    __threadfence_system();
    // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> increase counter to %d for sqe of collId %d", thrdCudaDev, bid, tid, old_counter + 1, sqeCollId);
    
    if (old_counter + 1 == globalBlkCount4Coll[sqeCollId]) {
      
      unsigned long long int old_head = atomicAdd(&sq->head, 1);

      __threadfence_system();
      // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> sqe of collId %d commit read, new sqHead is %llu", thrdCudaDev, bid, tid, sqeCollId, old_head + 1);
    }
  }
  
  return 0;
}

__device__ int cqWrite(CQ *cq, CQE *cqe, int thrdCudaDev) {
  // the first thread of a block do this.
  // int bid = blockIdx.x;
  // int tid = threadIdx.x;
  // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> enter cqRead, RingBuffer_full(cq)=%d, cqHead=%llu, cqTail=%llu", thrdCudaDev, bid, tid, RingBuffer_full(cq), RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));
  if (RingBuffer_full(cq)) {
    // not an error; caller keeps trying.
    return -1;
  }

  *RingBuffer_get_tail(cq) = *cqe;

  __threadfence_system();

  atomicAdd(&cq->tail, 1); // uint64, 一往无前++
  // RingBuffer_commit_write(cq, 1);

  return 0;
}

// TODO: share mem用超了。
static __shared__ ofcclShmemData ofcclShmem;
static __shared__ int sharedCollIds[MAX_LENGTH];
static __shared__ int sharedBlkCount4Coll[MAX_LENGTH];
static __shared__ int sharedThrdCount4Coll[MAX_LENGTH];

__global__ void daemonKernel(SQ *sq, CQ *cq, int thrdCudaDev, int collCount, CQE *globalCqes, int *globalBlkCount4Coll, int *globalThrdCount4Coll, int *globalCollIds, DevComm7WorkElem *globalDevComm7WorkElems) {
  uint64_t round = 0;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  unsigned long long int sqReadFrontier = 0;
  // int tempRound = 0;

  // TODO: 构建任务列表
  for (int i = 0; i < collCount; i++) {
    int collId = sharedCollIds[i] = globalCollIds[i];
    // 这两个变量会限制很多行为。
    int blkLimit = sharedBlkCount4Coll[collId] = globalBlkCount4Coll[collId];
    int thrdLimit = sharedThrdCount4Coll[collId] = globalThrdCount4Coll[collId];

    int nthreads = thrdLimit; // 需要按不同的集合通信分别指定。

  //   // ***** 移植ncclKernel的逻辑 *****
  //   if (bid < blkLimit && tid < thrdLimit) {
  //     ncclDevComm *comm = globalDevComm7WorkElems[collId].comm;
  //     int turn = copyToShmem(&(ofcclShmem[collId].comm), comm, tid, nthreads);
  //     // 不明白为啥能这样强转：get address of channel without incurring indirect load from ncclDevComm::channels
  //     // 这里通过bid选择了合适的channel，很多集合通信真正执行时用到的硬件信息就存在channel里边。
  //     ncclChannel *channel = &((ncclDevCommAndChannels*)comm)->channels[bid];
  //     turn = copyToShmem(&ofcclShmem[collId].channel, channel, tid, nthreads, turn);
  //     // 按目前的理解，copyToShmem的行为应该是每个thread往shmem复制了一点。
  //     __syncthreads();

  //     // nccl中限制只在bid=0里进行这样的拷贝，对于ofccl而言，ofcclShmem就是任务列表，所以对于所有的线程，我们都把同样的work存进去；
  //     copyToShmem(&(ofcclShmem[collId].work), &(globalDevComm7WorkElems[collId].first));
  //     // nccl中接下来要处理channel.workFifoDev，然而对于目前的ofccl，只处理first就好，channel.workFifoDev不会有其他任务了。
  //     __syncthreads();

  //     if (ofcclShmem[collId].work.header.type == ncclWorkTypeColl) {
  //       // #define NCCL_MAX_WORK_ELEMENTS (NCCL_WORK_SIZE / sizeof(struct ncclWorkElem))=512/64=8
  //       // 原来这个写法，应该是想修改we->redOpArg，不过修改we->redOpArg一个线程就够了，所以让理论上最多的线程来工作，咱们保留就好。
  //       if (tid < NCCL_MAX_WORK_ELEMENTS) ofcclRedopPtrDeref(&ofcclShmem[collId].work.elems[tid]);
  //     } // 目前不用考虑其他type
  //   }
  //   __syncthreads();

  }

  while (true) {
    if (tid == 0) {
      if (round++ == 0) {
        // init shmem
        quit = 0;
        __threadfence_block();
      }

      SQE target;
      // TODO: really need system?? 之后可以看看__threadfence()会不会提高性能。
      __threadfence_system(); // make sure read new head.
      if (sqReadFrontier < sq->head) {
        // 如果当前bid比较大，一些SQE不需要这个block处理，就会跳过。导致当前block的frontier小于head。
        // 不给sqRead增加返回值种类；否则会增加无谓的sqRead调用、增加访存次数。
        sqReadFrontier = sq->head;
      }
      if (RingBuffer_logic_tail(sq) == GetLogicFrontier(sq, sqReadFrontier) || sqRead(sq, sqReadFrontier, &target, sharedBlkCount4Coll, thrdCudaDev) == -1) {
        goto thrd_common;
      } else {
        sqReadFrontier++;
        // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> get collId %d, sqReadFrontier updates to %llu", thrdCudaDev, bid, tid, target.collId, GetLogicFrontier(sq, sqReadFrontier));
        if (target.quit) {
          quit = 1;
          // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Main Thrd of Blk quit", thrdCudaDev, bid, tid);
          goto thrd_common;
        }

        // real work
        // TODO: 在真正的ofccl代码里，需要在这里通过shmem把sendbuf，recvbuf分享出去，然后这个blk的全部线程开始执行
        // 分配的时候注意限制tid，以及等待相应的tid报告完成










        // 等待所有线程报告工作完成，0线程可以报告当前blk工作完成。
        // 然后协调所有blk，发现所有blk都完成，最后一个blk发送CQE
        int old_counter = atomicAdd(&(globalCqes[target.collId].counter), 1);
        __threadfence(); // cqes在global memory里边，全部thread关心。

        if (old_counter + 1 == sharedBlkCount4Coll[target.collId]) {
          atomicExch(&globalCqes[target.collId].counter, 0);
          while (cqWrite(cq, globalCqes + target.collId, thrdCudaDev) == -1) {
            // tempRound++;
            // if(tempRound % tempPrintRound == 0) OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> cqWrite fail, RingBuffer_full(cq)=%d, cqHead=%llu, cqTail=%llu", thrdCudaDev, bid, tid, RingBuffer_full(cq), RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));

          }
          // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> insert CQE for collId %d, cqHead=%llu, cqTail=%llu", thrdCudaDev, bid, tid, target.collId, RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));
          // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> insert CQE for collId %d, cqHead=%llu, cqTail=%llu", thrdCudaDev, bid, tid, target.collId, RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));
          __threadfence();
        }
      }
    }
    
thrd_common:
    __syncthreads();
    if (quit == 1) {
      // OFCCL_LOG_RANK_0(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> quit", thrdCudaDev, bid, tid);
      return;
    }
  }
}