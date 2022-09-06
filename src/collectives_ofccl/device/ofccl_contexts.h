#ifndef OFCCL_DEVICE_CONTEXTS_H_
#define OFCCL_DEVICE_CONTEXTS_H_

#include "collectives_ofccl.h"
#include "devcomm.h"

typedef struct {
  ncclConnInfo *recvConns[NCCL_MAX_DIRECT_ARITY];
  ncclConnInfo *sendConns[NCCL_MAX_DIRECT_ARITY];
  void* srcs[NCCL_MAX_DIRECT_ARITY+1];
  void* dsts[NCCL_MAX_DIRECT_ARITY+1];
  int totalSendSize[NCCL_MAX_SLICE_PER_CHUNK];
} ofcclShmemGroup;

// sizeof(ofcclShmemData)=42104, sizeof(ofcclShmemGroup)=248, sizeof(ncclDevComm)=40, sizeof(ncclChannel)=512, sizeof(ncclWork)=512
typedef struct {
  union {
    uint64_t ll128warp[NCCL_LL128_MAX_NTHREADS/WARP_SIZE][NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE]; // 这个占得大，占了40960
    ofcclShmemGroup groups[NCCL_MAX_GROUPS]; // 这个只占了3968
  };
  uint64_t redOpArgs[NCCL_MAX_DIRECT_ARITY+1];
  struct ncclDevComm comm;
  struct ncclChannel channel;
  uint64_t pad;
  struct ncclWork work;
  // 代表当前的表项对应的集合通信被调用，还没有执行完成。初始化置0；发现了相应的sqe之后置1；执行完成后置0。
  int executing;
} ofcclShmemData;
static_assert(offsetof(ofcclShmemData, work)%16 == 0, "shmem.work needs to be 16B aligned");

// TODO: 这里需要改造一下，切分成初始化和真正执行两部分。

#endif