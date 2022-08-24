#include "enqueue_ofccl.h"

static __shared__ int quit;

extern thread_local CQE *cqes;
extern thread_local int *BlkCount4Coll;
extern thread_local SQ *sq;
extern thread_local CQ *cq;

// TODO: still use 同步的Malloc吗？感觉是可以的，因为相当于是每个rank的init部分，而且prepareDone里还调用了cudaDeviceSynchronize
__host__ SQ *sqCreate(int length) {
  SQ *sq = nullptr;
  checkRuntime(cudaMallocHost((void **)&sq, sizeof(SQ)));
  sq->length = length + 1;
  sq->head = 0;
  sq->tail = 0;
  checkRuntime(cudaMallocHost((void **)&(sq->buffer), sq->length));
  pthread_mutex_init(&sq->mutex, nullptr);
  
  // OFCCL_LOG(OFCCL, "<%lu> rank=%d, sqCreate create sq @ %p", pthread_self(), thrdCudaDev, sq);

  return sq;
}

__host__ void sqDestroy(SQ *sq) {
  if (sq) {
    checkRuntime(cudaFreeHost(sq->buffer));
    checkRuntime(cudaFreeHost(sq));
  }
}

__host__ int sqWrite(SQ *sq, SQE *sqe, int thrdCudaDev) {
  OFCCL_LOG(OFCCL, "<%lu> rank=%d, Enter sqWrite, sq @ %p", pthread_self(), thrdCudaDev, sq);
  pthread_mutex_lock(&sq->mutex);

  if (RingBuffer_full(sq)) {
    // not an error; caller keeps trying.
    pthread_mutex_unlock(&sq->mutex);
    return -1;
  }
  sqe->logicHead = (int)RingBuffer_logic_tail(sq);
  *RingBuffer_get_tail(sq) = *sqe;
  OFCCL_LOG(OFCCL, "<%lu> write in sqe of collId %d counter=%d, quit=%d", pthread_self(), sqe->collId, sqe->counter, sqe->quit);

  __sync_synchronize();

  sq->tail += 1;
  OFCCL_LOG(OFCCL, "<%lu> commit write, sqHead=%llu, new sqTail is %llu", pthread_self(), RingBuffer_logic_head(sq), RingBuffer_logic_tail(sq));

  pthread_mutex_unlock(&sq->mutex);

  return 0;
}

__device__ int sqRead(SQ *sq, unsigned long long int readFrontier, SQE *target, int *BlkCount4Coll, int thrdCudaDev) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int sqeCollId;
  OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> enter sqRead, sqHead=%llu, sqTail=%llu, empty=%d, RingBuffer_get(sq, readFrontier)->counter=%d, RingBuffer_get(sq, readFrontier)->collId=%d, RingBuffer_get(sq, readFrontier)->quit=%d, RingBuffer_get(sq, readFrontier)->logicHead=%d, GetLogicFrontier(sq, readFrontier)=%llu", thrdCudaDev, bid, tid, RingBuffer_logic_head(sq), RingBuffer_logic_tail(sq), RingBuffer_empty(sq), RingBuffer_get(sq, readFrontier)->counter, RingBuffer_get(sq, readFrontier)->collId, RingBuffer_get(sq, readFrontier)->quit, RingBuffer_get(sq, readFrontier)->logicHead, GetLogicFrontier(sq, readFrontier));
  if (RingBuffer_empty(sq)) {
    return -1;
  }
  // 先读过来，然后再判断，最后更新状态：sqe->counter; 以及在恰当的时候commit read
  *target = *RingBuffer_get(sq, readFrontier);
  if (target->quit) {
    OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Get quit", thrdCudaDev, bid, tid);
    return 0;
  }

  // 先判断一下相应的collId是不是该自己的bid处理，不该自己处理直接返回-1
  sqeCollId = target->collId;
  // OFCCL_LOG(OFCCL, "Blk<%d>, Thrd<%d> BlkCount4Coll[%d]=%d", thrdCudaDev, bid, tid, sqeCollId, BlkCount4Coll[sqeCollId]);
  if (bid >= BlkCount4Coll[sqeCollId]) {
    return -1;
  } else {
    // 自己读到之后，更新相应的counter；至于读到的sqe对应的collId是不是该自己处理，是caller的事。
    // 如果发现自己读完之后，所有block都读了，那么commit read
    // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> PREPARE to increase counter(curr=%d) for sqe of collId %d", thrdCudaDev, bid, tid, RingBuffer_get(sq, readFrontier)->counter, sqeCollId);
    int old_counter = atomicAdd(&(RingBuffer_get(sq, readFrontier)->counter), 1);
    // RingBuffer_get(sq, readFrontier)->counter += 1;
    __threadfence_system();
    OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> increase counter to %d for sqe of collId %d", thrdCudaDev, bid, tid, old_counter + 1, sqeCollId);
    // if (RingBuffer_get(sq, readFrontier)->counter == BlkCount4Coll[sqeCollId]) {
    if (old_counter + 1 == BlkCount4Coll[sqeCollId]) {
      // RingBuffer_commit_read(sq, 1);
      unsigned long long int old_head = atomicAdd(&sq->head, 1);

      __threadfence_system();
      OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> sqe of collId %d commit read, new sqHead is %llu", thrdCudaDev, bid, tid, sqeCollId, old_head + 1);
    }
    
    // 修改了GPU和CPU都关心的sq.counter
    __threadfence_system();
  }
  
  return 0;
}

__host__ CQ *cqCreate(int length) {
  CQ *cq = nullptr;
  checkRuntime(cudaMallocHost((void **)&cq, sizeof(CQ)));
  cq->length = length + 1;
  cq->head = 0;
  cq->tail = 0;
  checkRuntime(cudaMallocHost((void **)&(cq->buffer), cq->length));
  pthread_mutex_init(&cq->mutex, nullptr);

  return cq;
}

__host__ void cqDestroy(CQ *cq) {
  if (cq) {
    checkRuntime(cudaFreeHost(cq->buffer));
    checkRuntime(cudaFreeHost(cq));
  }
}

__host__ int cqRead(CQ *cq, CQE *target, int collId, int thrdCudaDev) {
  pthread_mutex_lock(&cq->mutex);
  // OFCCL_LOG(OFCCL, "<%lu> enter cqRead, RingBuffer_empty(cq)=%d, cqHead=%llu, cqTail=%llu, headCollId=%d, wait for %d", pthread_self(), RingBuffer_empty(cq), RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq), RingBuffer_get_head(cq)->collId, collId);

  if (RingBuffer_empty(cq)) {
    pthread_mutex_unlock(&cq->mutex);
    return -1;
  }
  if (RingBuffer_get_head(cq)->collId != collId) {
    pthread_mutex_unlock(&cq->mutex);
    return -1;
  }
  // checkRuntime(cudaMemcpy(target, RingBuffer_get_head(cq), sizeof(CQE), cudaMemcpyHostToHost));
  *target = *RingBuffer_get_head(cq);

  __sync_synchronize();

  cq->head += 1;

  pthread_mutex_unlock(&cq->mutex);

  return 0;
}

__device__ int cqWrite(CQ *cq, CQE *cqe, int thrdCudaDev) {
  // the first thread of a block do this.
  // int tid = threadIdx.x;
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


__global__ void deamonKernel(SQ *sq, CQ *cq, CQE *cqes, int *BlkCount4Coll, int thrdCudaDev) {
  uint64_t round = 0;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  unsigned long long int readFrontier = 0;
  while (true) {
    if (tid == 0) {
      if (round++ == 0) {
        // init shmem
        quit = 0;
        __threadfence_block();
      }

      SQE target;
      // TODO: really need system??
      __threadfence_system(); // make sure read new head.
      if (readFrontier < sq->head) {
        // 当前bid比较大，一些SQE不需要这个block处理，就会跳过。导致当前block的frontier小于head。
        // 不给sqRead增加返回值种类；否则会增加无谓的sqRead调用、增加访存次数。
        readFrontier = sq->head;
      }
      if (RingBuffer_logic_tail(sq) == GetLogicFrontier(sq, readFrontier) || sqRead(sq, readFrontier, &target, BlkCount4Coll, thrdCudaDev) == -1) {
        goto thrd_common;
      } else {
        readFrontier++;
        // OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> get collId %d, readFrontier updates to %llu", thrdCudaDev, bid, tid, target.collId, GetLogicFrontier(sq, readFrontier));
        if (target.quit) {
          quit = 1;
          OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> Main Thrd of Blk quit", thrdCudaDev, bid, tid);
          goto thrd_common;
        }

        // real work
        // TODO: 在真正的ofccl代码里，需要在这里通过shmem把sendbuf，recvbuf分享出去，然后这个blk的全部线程开始执行
        // 分配的时候注意限制tid，以及等待相应的tid报告完成










        // 等待所有线程报告工作完成，0线程可以报告当前blk工作完成。
        // 然后协调所有blk，发现所有blk都完成，最后一个blk发送CQE
        int old_counter = atomicAdd(&(cqes[target.collId].counter), 1);
        __threadfence(); // cqes在global memory里边，全部thread关心。

        if (old_counter + 1 == BlkCount4Coll[target.collId]) {
          atomicExch(&cqes[target.collId].counter, 0);
          while (cqWrite(cq, cqes + target.collId, thrdCudaDev) == -1) {

          }
          OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> insert CQE for collId %d, cqHead=%llu, cqTail=%llu", thrdCudaDev, bid, tid, target.collId, RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));
          __threadfence();
        }
      }
    }
    
thrd_common:
    __syncthreads();
    if (quit == 1) {
      OFCCL_LOG(OFCCL, "Rank<%d> Blk<%d> Thrd<%d> quit", thrdCudaDev, bid, tid);
      return;
    }
  }
}