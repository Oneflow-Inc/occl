/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue_ofccl.h"
#include "devcomm.h"
#include "enqueue.h" // struct ncclQueueInfo
#include "argcheck.h"
#include "bootstrap.h"
#include "channel.h"
#include "coll_net.h"
#include "debug.h"
#include "gdrwrap.h"
#include "group.h"
#include "transport.h"

#include <cstdlib>
#include <cstring> // std::memcpy
#include <pthread.h> // pthread_self()
#include <math.h> // floor()
#include <unordered_set>

static void *const ofcclKerns[1] = {
    (void *)try_make_kern,
};

namespace {
void try_make() {
  dim3 gridDim, blockDim;
  gridDim.x = 8;
  blockDim.x = 4;
  int a = 1;
  int *b = &a;
  cudaLaunchKernel(ofcclKerns[0], gridDim, blockDim, (void **)&b, 0, NULL);
}

static inline ncclResult_t ofcclGetCollNetSupport(struct ncclInfo* info, int* collNetTypeSupport) {
  if (info->comm->collNetSupport > 0) {
    // Translate ncclAvg and PreMulSum
    ncclRedOp_t netOp = info->op == ncclAvg || info->op >= ncclNumOps ? ncclSum : info->op;
    NCCLCHECK(collNetReduceSupport(info->datatype, netOp, collNetTypeSupport));
  } else {
    *collNetTypeSupport = 0;
  }
  return ncclSuccess;
}

// numPipeOps: number of pipelined ops. Can be greater than 1 in aggregation mode. Used to adjust latency.
static ncclResult_t ofcclGetAlgoInfo(struct ncclInfo* info, int collNetTypeSupport, int numPipeOps) {
  struct ncclComm* comm = info->comm;
  if (comm->nRanks == 1) {
    info->algorithm = NCCL_ALGO_RING;
    info->protocol = NCCL_PROTO_SIMPLE;
  }
  else {
    float minTime = 3600000000.0; // Hopefully no operation will take an hour to complete.
    // Find algorithm / protocol.
    info->algorithm = -1;
    info->protocol = -1;
    int nAlgos = NCCL_NUM_ALGORITHMS;
    for (int a=0; a<nAlgos; a++) {
      if (a == NCCL_ALGO_COLLNET && collNetTypeSupport != 1) continue;
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        float time;
        NCCLCHECK(ncclTopoGetAlgoTime(info, a, p, numPipeOps, &time));
        if (time >= 0 && time < minTime) {
          info->algorithm = a;
          info->protocol = p;
          minTime = time;
        }
      }
    }
    if (info->algorithm == -1 || info->protocol == -1) {
      OFCCL_LOG1(OFCCL_WARN, "Error : no algorithm/protocol available");
      return ncclInternalError;
    }
    //if (comm->rank == 0) INFO(NCCL_TUNING, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);
    TRACE(NCCL_COLL, "%ld Bytes -> Algo %d proto %d time %f", info->nBytes, info->algorithm, info->protocol, minTime);
  }

  int nc = (info->nChannels > 0) ? info->nChannels : comm->nChannels;
  int nt = comm->maxThreads[info->algorithm][info->protocol];
  int threadThreshold = comm->threadThresholds[info->algorithm][info->protocol];
  if (info->algorithm == NCCL_ALGO_COLLNET) {
    // CollNet channel tuning
    int ncSwitch = 16;
    bool flag = true;
    while (ncSwitch >= 1 && flag) {
      while ((flag = info->nBytes < nc*nt*info->comm->channels[0].collTree.nHeads*threadThreshold) && nc > ncSwitch) {
        if (nc == ncSwitch+ncSwitch/2) threadThreshold /= 2;
        nc--;
      }
      ncSwitch /= 2;
    }
  } else {
    // Ring/Tree channel tuning
    while (info->nBytes < nc*nt*threadThreshold) {
      if (nc >= 2) nc--;
      else if ((nt % 128) == 0) nt/=2;
      else break;
    }
  }
  if (info->protocol == NCCL_PROTO_SIMPLE) {
    nt += WARP_SIZE; // Extra warp for sync
    // More threads or sync warps needed due to split thread model
    if (info->algorithm == NCCL_ALGO_TREE) nt += 3*WARP_SIZE;
    if (info->algorithm == NCCL_ALGO_COLLNET) nt += 3*WARP_SIZE;
  }
  info->nChannels = nc;
  info->nThreads = nt;
  // OFCCL_LOG(OFCCL, "info->algorithm=%d, info->protocol=%d", info->algorithm, info->protocol);
  return ncclSuccess;
}

static ncclResult_t ofcclGetPatternInfo(struct ncclInfo* info) {
  switch (info->coll) {
    case ncclFuncBroadcast:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeDown : ncclPatternPipelineFrom; break;
    case ncclFuncReduce:
      info->pattern = info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUp : ncclPatternPipelineTo; break;
    case ncclFuncReduceScatter:
    case ncclFuncAllGather:
      info->pattern = ncclPatternRing; break;
    case ncclFuncAllReduce:
      info->pattern = info->algorithm == NCCL_ALGO_COLLNET ? ncclPatternCollTreeUpDown : info->algorithm == NCCL_ALGO_TREE ? ncclPatternTreeUpDown : ncclPatternRingTwice; break;
    default:
      OFCCL_LOG(OFCCL_WARN, "Unknown pattern for collective %d algorithm %d", info->coll, info->algorithm);
      return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t ofcclGetLoopInfo(struct ncclInfo* info) {
  switch (info->pattern) {
    case ncclPatternTreeUp:
    case ncclPatternTreeDown:
    case ncclPatternTreeUpDown:
    case ncclPatternPipelineFrom:
    case ncclPatternPipelineTo:
      info->nstepsPerLoop = info-> nchunksPerLoop = 1; break;
    case ncclPatternCollTreeUpDown:
      info->nstepsPerLoop = 1; info->nchunksPerLoop = info->comm->channels[0].collTree.nHeads; break;
    case ncclPatternRing:
      info->nstepsPerLoop = info->comm->nRanks-1; info->nchunksPerLoop = info->comm->nRanks; break;
    case ncclPatternRingTwice:
      info->nstepsPerLoop = 2*(info->comm->nRanks-1); info->nchunksPerLoop = info->comm->nRanks; break;
    default:
      OFCCL_LOG(OFCCL_WARN, "Unknown pattern %d", info->pattern);
      return ncclInternalError;
  }
  return ncclSuccess;
}

// info is input, work and proxyOp are outputs
static ncclResult_t ofcclComputeColl(struct ncclInfo* info /* input */, struct ncclWorkElem* work, struct ncclProxyOp* proxyOp /* output */) {
  int collNetTypeSupport = 0;

  // ***** omit checking aggregation case *****

  NCCLCHECK(ofcclGetCollNetSupport(info, &collNetTypeSupport));
  NCCLCHECK(ofcclGetAlgoInfo(info, collNetTypeSupport, 1));

  // Set nstepsPerLoop and nchunksPerLoop
  NCCLCHECK(ofcclGetPatternInfo(info));
  NCCLCHECK(ofcclGetLoopInfo(info));

  work->header.type = ncclWorkTypeColl;
  // ***** omit setting work->sendbuff and work->recvbuff *****
  work->root = info->root;
  work->count = info->count;
  work->nChannels = info->nChannels;
  work->header.nWarps = info->nThreads / WARP_SIZE;
  work->redOpArg = info->opFull.scalarArg;
  work->redOpArgIsPtr = info->opFull.scalarArgIsPtr;

  // ***** skip special case comm->nRanks == 1 *****

  work->header.funcIndex = FUNC_INDEX(info->coll, info->opFull.op, info->datatype, info->algorithm, info->protocol);

  int stepSize   = info->comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize  = stepSize*chunkSteps;

  // Compute lastChunkSize
  if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_SIMPLE) {
    if (info->pattern == ncclPatternTreeUpDown) {
      // Optimize chunkSize / nSteps
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*8 && chunkSize > 131072) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth*4 && chunkSize > 65536) chunkSize /= 2;
      while (info->nBytes / (info->nChannels*chunkSize) < info->comm->channels[0].tree.depth && chunkSize > 32768) chunkSize /= 2;
    }
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_COLLNET && info->protocol == NCCL_PROTO_SIMPLE) {
    // Optimize chunkSize / nSteps
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collTree.nHeads*chunkSize) < info->comm->channels[0].collTree.depth*64 && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collTree.nHeads*chunkSize) < info->comm->channels[0].collTree.depth*8 && chunkSize > 65536) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*info->comm->channels[0].collTree.nHeads*chunkSize) < info->comm->channels[0].collTree.depth*8 && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize / ncclTypeSize(info->datatype);
    // Set direct direction for broadcast-gather (read or write)
    work->direct = (info->nBytes / info->nChannels <= 1024*1024) ? NCCL_DIRECT_WRITE : NCCL_DIRECT_READ;
  } else if (info->protocol == NCCL_PROTO_LL) {
    const ssize_t sliceSize = stepSize*sizeof(uint64_t)/sizeof(union ncclLLFifoLine);
    const ssize_t loopSize = info->nChannels*info->nchunksPerLoop*(ssize_t)sliceSize;
    work->lastChunkSize = DIVUP((info->nBytes-(info->nBytes/loopSize)*loopSize), info->nChannels*info->nchunksPerLoop);
    ALIGN_SIZE(work->lastChunkSize, info->nThreads*sizeof(uint64_t));
    work->lastChunkSize /= ncclTypeSize(info->datatype);
  } else if (info->algorithm == NCCL_ALGO_TREE && info->protocol == NCCL_PROTO_LL128) {
    int nNodes = info->comm->nNodes;
    float ppn = info->comm->nRanks / (float)nNodes;
    float nstepsLL128 = 1+log2i(nNodes) + 0.1*ppn;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*64/ppn && chunkSize > 131072) chunkSize /= 2;
    while (info->nBytes / (info->nChannels*chunkSize) < nstepsLL128*16/ppn && chunkSize > 32768) chunkSize /= 2;
    // Use lastChunkSize as chunkSize
    work->lastChunkSize = chunkSize*NCCL_LL128_DATAELEMS/(NCCL_LL128_LINEELEMS*ncclTypeSize(info->datatype));
  }

  // Compute nSteps for proxies
  int chunkEffectiveSize = chunkSize;
  if (info->protocol == NCCL_PROTO_LL) chunkEffectiveSize /= 2;
  if (info->protocol == NCCL_PROTO_LL128) chunkEffectiveSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  //if (info->comm->rank == 0) printf("Coll %d, size %ld -> %dx%d, chunkSize %d (algo %d proto%d)\n", info->coll, info->nBytes, info->nChannels, info->nThreads, chunkSize, info->algorithm, info->protocol);
  int nLoops = (int)(DIVUP(info->nBytes, (((size_t)(info->nChannels))*info->nchunksPerLoop*chunkEffectiveSize)));
  proxyOp->nsteps = info->nstepsPerLoop * nLoops * chunkSteps;
  proxyOp->sliceSteps = sliceSteps;
  proxyOp->chunkSteps = chunkSteps;
  proxyOp->chunkSize = chunkSize;
  proxyOp->protocol = info->protocol;
  proxyOp->dtype = info->datatype;
  proxyOp->redOp = info->algorithm != NCCL_ALGO_COLLNET ? ncclNumOps : // Only set redOp when using CollNet
                     info->opFull.op==ncclDevPreMulSum || info->opFull.op==ncclDevSumPostDiv ? ncclSum : // Network sees avg as sum
                     info->op;
  proxyOp->pattern = info->pattern;
  proxyOp->root = info->root;
  // This is used by P2P to reduce the receive buffer size. We don't use it in collectives
  // because some protocols need to transmit more than the total size, plus they sometimes
  // round up
  proxyOp->nbytes = stepSize*proxyOp->sliceSteps;

  // OFCCL_LOG(OFCCL,"opCount %lx slicesteps %d spl %d cpl %d nbytes %zi -> protocol %d nchannels %d nthreads %d, nloops %d nsteps %d chunksize %d comm %p",
  //     proxyOp->opCount, sliceSteps, info->nstepsPerLoop, info->nchunksPerLoop, info->nBytes, info->protocol, info->nChannels, info->nThreads,
  //     nLoops, proxyOp->nsteps, chunkSize, info->comm);
  return ncclSuccess;
}


// Compute enqueue element, save it in list
// Compute CUDA launch parameters
static ncclResult_t ofcclSetupCollKernel(struct ncclInfo* info) {
  ncclComm_t comm = info->comm;

  // ***** skip special case comm->nRanks == 1 *****

  // Compute cuda kernel arg and proxy arg templates
  struct ncclQueueElem* eqElem;
  NCCLCHECK(comm->enqueueInfo->elemList->getNewElem(&eqElem));
  struct ncclWork* work = &eqElem->work;
  NCCLCHECK(ofcclComputeColl(info, work->elems, &eqElem->proxyOp));

  // Determine grid size
  struct cudaLaunchParams* params = comm->myParams;
  params->gridDim.x += info->nChannels;
  params->gridDim.x = std::min<unsigned>(params->gridDim.x, comm->nChannels);
  params->blockDim.x = std::max<unsigned>(params->blockDim.x, info->nThreads);
  comm->enqueueInfo->maxChannels = params->gridDim.x;  // params may be varied by a second graph hence we need to capture it here

  // ***** omit Inline the first kernel。这个设计是匹配了以comm为粒度启动kernel的，我们现在是要启动一个囊括全部comm的kernel *****
  // ***** omit params->func = ncclKerns[work->header.funcIndex]; ****

  return ncclSuccess;
}

// Op info has been previously saved in comm->asyncOps
static ncclResult_t ofcclSetupAsyncKernels(ncclComm_t comm) {
  // TODO: No aggregation for now
  // 我们的大kernel，或许可以参考通过channelSize反推channel数目的方法
  // 后续我们即便要实现“aggregation”的效果，和目前nccl的方式也会很不一样，nccl现在是直接在共享一个comm的多个op上计算；而我们是需要所有的op各自有独立的comm，所以到时候的调整会直接针对最终的gridDim等参数，而不涉及comm等相关数据结构的操作。
  struct ncclInfo* info = comm->asyncOps;
  info->nChannels = 0;
  NCCLCHECK(ofcclSetupCollKernel(info));

  // Reset counters
  comm->asyncOpCount = 0;
  comm->asyncTotalSize = 0;
  return ncclSuccess;
}

// Get next channel based on shortest-queue mode or round-robin mode
static inline int ofGetNextChannel(ncclComm_t comm) {
  int nextChannel = 0;
  nextChannel = comm->lastChannel % comm->nChannels;
  comm->lastChannel++;
  return nextChannel;
}

static ncclResult_t ofGetNextOp(struct ncclChannel* channel, struct ncclWork** work, struct ncclWorkElem* base) {
  if (channel->workCount == NCCL_MAX_OPS) {
    OFCCL_LOG(OFCCL_WARN, "Too many aggregated operations on channel %d (%d max)", channel->id, NCCL_MAX_OPS);
    return ncclInvalidUsage;
  }
  int opIndex = channel->workFifoTail%NCCL_MAX_OPS;
  struct ncclWork* w = channel->workFifo+opIndex;
  volatile uint8_t* typePtr = (volatile uint8_t*)&w->header.type;
  while (typePtr[0] != ncclWorkTypeUnused) sched_yield();
  memset(w, 0, sizeof(struct ncclWork));
  // Initialize with work elem if provided
  if (base) memcpy(w->elems, base, sizeof(struct ncclWorkElem));
  channel->workFifoTail++;
  channel->workCount++;
  if (work) *work = w;
  return ncclSuccess;
}

// Equeue work elements into segment of ncclWork
// Supporting both collectives (aggregated or not) and P2P
static ncclResult_t ofEnqueueSegOp(enum ncclWorkElemType type, struct ncclWork* elem /* input */, struct ncclWork* work, int s,
    struct ncclBuffRegInfo* regInfo, struct ncclChannel* channel, struct ncclComm* comm) {

  // ***** omit type == ncclWorkTypeP2p *****

  // ***** 目前最重要的是这里 *****
  memcpy(work->elems+s, elem, sizeof(struct ncclWorkElem));

  if (regInfo->nBuffs == 0) return ncclSuccess;
  // ***** 不考虑CollNet的话，接下来的内容不是很重要。 *****

  // Copy registered buffer addresses into ncclWork
  struct ncclWorkElemReg* regElem = (struct ncclWorkElemReg*)(work->elems+s);
  // For CollNet
  for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) {
    int peer = channel->collTree.down[i];
    if (peer == -1) break;
    // Get intra-node slot
    int j = comm->rankToLocalRank[peer];
    if (j < 0) {
      OFCCL_LOG(OFCCL_WARN, "Invalid intra-node rank %d for peer %d", j, peer);
      return ncclInternalError;
    }
    // Input buffer of leaf peer
    regElem->dnInputs[i] = regInfo->sendbuffs[j];
    // Output buffer of leaf peer
    regElem->dnOutputs[i] = regInfo->recvbuffs[j];
  }
  for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) {
    int peer = channel->collTree.up[i];
    if (peer == -1) break;
    int j = comm->rankToLocalRank[peer];
    if (j < 0) {
      OFCCL_LOG(OFCCL_WARN, "Invalid intra-node rank %d for peer %d", j, peer);
      return ncclInternalError;
    }
    // Output buffer of root peer
    regElem->upOutputs[i] = regInfo->recvbuffs[j];
  }
  work->elems[s].regUsed = 1;
  return ncclSuccess;
}

// Dynamic enqueue function for collective kernels
static ncclResult_t ofcclEnqueueCollKernel(struct ncclComm* comm, struct ncclQueueElem* eqElem) {
  
  struct ncclWork* work = &eqElem->work;
  struct ncclWorkElem* elem = work->elems;
  struct ncclProxyOp* proxyOp = &eqElem->proxyOp;

  int nChannels = elem->nChannels;
  size_t channelSize = elem->count*ncclTypeSize(proxyOp->dtype)/elem->nChannels;
  enum ncclWorkElemType workElemType = proxyOp->redOp == ncclNumOps ? ncclWorkTypeColl : ncclWorkTypeRegColl;  // redOp is only set when using CollNet
  
  for (int bid=0; bid<nChannels; bid++) {
    int channelId = ofGetNextChannel(comm);
    struct ncclChannel* channel = comm->channels+channelId;

    // Proxy
    proxyOp->channelId = channelId;
    proxyOp->opCount = comm->collOpCount;
    if (proxyOp->nsteps) NCCLCHECK(ncclProxySaveColl(comm, proxyOp, comm->nRanks));

    elem->bid = bid % nChannels;
    struct ncclWork* w = NULL;
    int segment = -1;
    if (segment == -1) {
      NCCLCHECK(ofGetNextOp(channel, &w, NULL));
      segment = 0;
    }

    // store work element into FIFO
    NCCLCHECK(ofEnqueueSegOp(workElemType, work, w, segment, &eqElem->buffRegInfo, channel, comm));
    channel->totalSize += channelSize;
  }
  comm->collOpCount++;

  return ncclSuccess;
}

// Finalize channel work FIFO states before launch
// Called during dynamic enqueue
static ncclResult_t ofcclSetupLaunch(struct ncclQueueInfo* eqInfo) {
  ncclComm_t comm = eqInfo->comm;
  // Do not use comm->myParams in this function unless in non-graph mode
  // In graph mode, enqueue is async to capture, myParams can have been changed
  struct cudaLaunchParams* params = comm->myParams;

  // Only launch blocks where we have work to do.
  // This is not supported when we are in cudaGraph mode.
  // Because in cudaGraph mode the launch param needs to be determined
  // at capture time instead of launch time.
  // ***** 对于我们来说其实也类似，最终启动的参数，是要综合全部的comm的情况的 *****
  // if (!usingCudaGraph) {
  int nChannels = std::max(comm->nChannels, comm->p2pnChannels);
  for (int c=0; c<nChannels; c++) {
    if (comm->channels[c].workCount) params->gridDim.x = c+1;
  }
  eqInfo->maxChannels = params->gridDim.x;
  // }

  // Set isLast = 1 for the last operation and add a no-op on empty channels (p2p case).
  for (int c=0; c<eqInfo->maxChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    // ***** no p2p now *****
    // if (channel->workCount == 0) {
    //   struct ncclWork* w;
    //   NCCLCHECK(ofGetNextOp(channel, &w, NULL));
    //   w->header.funcIndex = FUNC_INDEX_P2P;
    //   w->header.type = ncclWorkTypeP2p;
    //   w->header.nWarps = 0;
    // }
    channel->workFifo[(channel->workFifoTail-1)%NCCL_MAX_OPS].header.isLast = 1;

    // ***** 把第一个channel里的“正常操作”清零了，我们应该不需要这样。 *****
    // if (c == 0) {
    //   // As we inline the first coll directly, we can free it immediately.
    //   // Except P2P or aggregation or registration cases
    //   struct ncclWork* work = channel->workFifo+((channel->workFifoTail-channel->workCount)%NCCL_MAX_OPS);
    //   if (work->header.type == ncclWorkTypeColl && eqInfo->elemList->count() == 1)
    //     work->header.type = ncclWorkTypeUnused;
    // }

    if (channel->gdrMemDesc) {
      // GDRCOPY support
      uint64_t first = (channel->workFifoTail-channel->workCount)%NCCL_MAX_OPS;
      uint64_t nelems = channel->workCount;
      // OFCCL_LOG(OFCCL, "GDRCOPY : copy workFifo %p to %p first %ld nelems %zi", channel->workFifo, channel->workFifoGdr, first, nelems);

      for (int i = 0; i < nelems; i++) {
        int elem = (first+i) % NCCL_MAX_OPS;
        // Copy Host workFifo to CUDA workFifo via the GDRCOPY mapping
        // ***** 早在initChannel的时候，就决定了channel->workFifoDev是和channel->workFifo还是和channel->workFifoGdr绑定 *****
        NCCLCHECK(ncclGdrCudaCopy(channel->gdrMemDesc, channel->workFifoGdr+elem, channel->workFifo+elem, 1));
      }
    }
  }

  return ncclSuccess;
}

// Launch network proxy
static ncclResult_t ofcclLaunchProxy(struct ncclQueueInfo* eqInfo) {
  // Start the network proxies as soon as the kernel has been launched. We can't
  // perform any CUDA call between the two or having a cudaFree between the CUDA
  // launch and the ncclProxyStart call could cause a deadlock.
  // Also, starting the proxies after the CUDA launch seems to be better for
  // performance (latency).
  ncclComm_t comm = eqInfo->comm;
  if (eqInfo->maxChannels == 0) return ncclSuccess;

  for (int r=0; r<eqInfo->maxChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    channel->workCount = 0;
    channel->totalSize = 0;
  }
  comm->lastChannel = 0;
  NCCLCHECK(ncclProxyStart(comm));
  return ncclSuccess;
}

// Performs the enqueue job
static void CUDART_CB ofcclEnqueueHostSetup(void* arg) {
  NVTX3_FUNC_RANGE_IN(ofccl_domain);
  ncclResult_t ret = ncclSuccess;
  // All work for current launch has been captured in Queue Info
  struct ncclQueueInfo* eqInfo = (struct ncclQueueInfo*)arg;
  ncclComm_t comm = eqInfo->comm;
  // OFCCL_LOG(OFCCL, "ncclQueueInfo eqInfo->elemList->count()=%d", eqInfo->elemList->count());
  struct ncclQueueElem* eqElem = eqInfo->elemList->begin();

  if (eqInfo->elemList->count() > 1) {
    OFCCL_LOG(OFCCL_WARN, "eqInfo->elemList->count()=%d, but should not > 1", eqInfo->elemList->count());
    goto cb_end;
  }

  // ***** omit ncclEnqueueP2pKernel for now *****

  // ***** ncclEnqueueCollKernel *****
  NCCLCHECKGOTO(ofcclEnqueueCollKernel(comm, eqElem), ret, cb_end);

  NCCLCHECKGOTO(ofcclSetupLaunch(eqInfo), ret, cb_end);
  NCCLCHECKGOTO(ofcclLaunchProxy(eqInfo), ret, cb_end);
  
cb_end:
  if (ret != ncclSuccess) {
    OFCCL_LOG(OFCCL_WARN, "Failure in host setup : %s", ncclGetErrorString(ret));
  }
  eqInfo->ret = ret;
}

} // namespace

#define MAX_ASYNC_PANELS 32
#define MAX_ASYNC_OPS 128

struct ofcclCommArgs {
  ncclResult_t ret;
  ncclComm_t comm;
};

// TODO: 之后考虑搞一个Context整理起来。
static thread_local ofcclCommArgs ofcclCommList[MAX_ASYNC_PANELS][MAX_ASYNC_OPS];
static thread_local std::unordered_set<ncclComm_t> seenComms;
static thread_local pthread_t ofcclPrepareThreads[MAX_ASYNC_PANELS][MAX_ASYNC_OPS];
static thread_local int ofcclCommListPanel = 0;
static thread_local int ofcclCommListFront = 0;

static thread_local pthread_t kernelThrd;
static thread_local CQE *tempCqes = nullptr;
static thread_local int *tempBlkCount4Coll = nullptr;
static thread_local void *argsptrs[5];
static thread_local cudaStream_t kernelStream;
static thread_local ThrdArgs thrdArgs;
static thread_local int thrdCudaDev;

thread_local CQE *cqes;
thread_local int *BlkCount4Coll;
thread_local SQ *sq;
thread_local CQ *cq;

// Configs
static thread_local dim3 daemonKernelGridDim;
static thread_local dim3 daemonKernelBlockDim;
static thread_local int queueLength = -1;
static thread_local int collCount = -1;

void *ofcclAsyncThreadPreconnect(void* args_) {
  struct ofcclCommArgs* args = (struct ofcclCommArgs*)args_;
  CUDACHECKTHREAD(cudaSetDevice(args->comm->cudaDev));
  if (CPU_COUNT(&args->comm->cpuAffinity))
    sched_setaffinity(0, sizeof(cpu_set_t), &args->comm->cpuAffinity);
  NCCLCHECKTHREAD(ncclTransportP2pSetup(args->comm, NULL, 1));
  return args;
}

// NCCL_API(ncclResult_t, ofcclPrepareCollComm, struct ncclInfo *info, int collId);
ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId) {
  if (ofcclCommListPanel == 0 && ofcclCommListFront == 0) {
    memset(ofcclCommList, 0, sizeof(ncclComm_t) * MAX_ASYNC_PANELS * MAX_ASYNC_OPS); // legacy in ncclGroupStart
  }
  
  ncclResult_t ret = ncclSuccess;  
  // Check arguments
  info->comm->checkPointers = false; // we do not assign buff pointers yet.
  NCCLCHECK(PtrCheck(info->comm, info->opName, "comm"));
  NCCLCHECKGOTO(ArgsCheck(info), ret, end);

  // Copy reduction op state from op handle into info struct here since the
  // op handle may be destroyed before ncclGroupEnd().
  // ***** Ignore for now. *****
  // NCCLCHECKGOTO(hostToDevRedOp(&info->opFull, info->op, info->datatype,
  // info->comm), ret, end);

  // ***** ncclAsyncColl(info->comm) *****

  if (seenComms.find(info->comm) != seenComms.end()) {
    OFCCL_LOG1(OFCCL_WARN, "Reuse ncclComm is not allowed");
    ret = ncclInvalidUsage;
    goto end;
  }
  seenComms.insert(info->comm);

  ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm = info->comm;
  // OFCCL_LOG(OFCCL, "i = %d, j = %d, tempcomm=%p(at %p), info->comm=%p(at %p), tempcomm->nRanks = %d, tempcomm->connect=%d", ofcclCommListPanel, ofcclCommListFront, ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm, &(ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm), info->comm, &(info->comm), ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm->nRanks, ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm->connect);
  // OFCCL_LOG(OFCCL, "pthread_id=%lu, i = %d, j = %d, tempcomm=%p(at %p), info->comm=%p(at %p), tempcomm->nRanks = %d, tempcomm->connect=%d", pthread_self(), ofcclCommListPanel, ofcclCommListFront, ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm, &(ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm), info->comm, &(info->comm), ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm->nRanks, ofcclCommList[ofcclCommListPanel][ofcclCommListFront].comm->connect);

  ofcclCommListFront++;
  if (ofcclCommListFront >= MAX_ASYNC_OPS) {
    ofcclCommListFront = 0;
    ofcclCommListPanel++;
    if (ofcclCommListPanel >= MAX_ASYNC_PANELS) {
      OFCCL_LOG(OFCCL_WARN, "Too many async operations in progress, max is %d",
           MAX_ASYNC_PANELS * MAX_ASYNC_OPS);
      ret = ncclInvalidUsage;
      goto end;
    }
  }

  // ***** Ignore *****
  // NCCLCHECKGOTO(checkSetStream(info), ret, end);

  // ***** ncclSaveAsyncColl(info) *****

  if (info->comm->asyncOpCount > 1) {
    OFCCL_LOG1(OFCCL_WARN, "comm->asyncOpCount shouldn't be larger than 1");
    ret = ncclInvalidUsage;
  }

  memcpy(info->comm->asyncOps + info->comm->asyncOpCount, info,
         sizeof(struct ncclInfo));
  info->comm->asyncOpCount++;
  info->comm->asyncTotalSize += info->nBytes;

end:
  
  return ret;
}

void *startKernel(void *args) {

  SQ *sq = ((ThrdArgs *)args)->sq;
  CQ *cq = ((ThrdArgs *)args)->cq;
  CQE *cqes = ((ThrdArgs *)args)->cqes;
  int *BlkCount4Coll = ((ThrdArgs *)args)->BlkCount4Coll;
  cudaStream_t stream = ((ThrdArgs *)args)->stream;

  // Setup thread_local variables using values from parent thread.
  // OFCCL_LOG(OFCCL, "<%lu> thrdCudaDev in new thread is %d", pthread_self(), thrdCudaDev);
  thrdCudaDev = ((ThrdArgs *)args)->cudaDev;
  daemonKernelGridDim = ((ThrdArgs *)args)->gridDim;
  daemonKernelBlockDim = ((ThrdArgs *)args)->blockDim;

  checkRuntime(cudaSetDevice(thrdCudaDev));

  // TODO: 之后考虑按需启停kernel
  
  OFCCL_LOG_RANK_0(OFCCL, "<%lu> rank=%d after KernelThrd set daemonKernelGridDim, gridDimx=%d, blockDimx=%d", pthread_self(), thrdCudaDev, daemonKernelGridDim.x, daemonKernelBlockDim.x);

  argsptrs[0] = &sq;
  argsptrs[1] = &cq;
  argsptrs[2] = &cqes;
  argsptrs[3] = &BlkCount4Coll;
  argsptrs[4] = &thrdCudaDev;

  struct cudaLaunchParams daemonKernelParam;
  daemonKernelParam.func = (void *)daemonKernel;
  daemonKernelParam.gridDim = daemonKernelGridDim;
  daemonKernelParam.blockDim = daemonKernelBlockDim;
  daemonKernelParam.sharedMem = 0;
  daemonKernelParam.stream = stream;
  daemonKernelParam.args = argsptrs;

  // OFCCL_LOG(OFCCL, "<%lu> rank=%d, sq @ %p, cq @ %p, cqes @ %p, BlkCount4Coll @ %p, func @ %p, stream @ %p, args @ %p", pthread_self(), thrdCudaDev, sq, cq, cqes, BlkCount4Coll, daemonKernelParam.func, daemonKernelParam.stream, daemonKernelParam.args);

  checkRuntime(cudaLaunchKernel(daemonKernelParam.func, daemonKernelParam.gridDim, daemonKernelParam.blockDim, daemonKernelParam.args, daemonKernelParam.sharedMem, daemonKernelParam.stream));

  // daemonKernel<<<gridDimx, blockDimx, 0, stream>>>(sq, cq, cqes, BlkCount4Coll);
  
  cudaStreamSynchronize(stream);

  return NULL;
}

NCCL_API(ncclResult_t, ofcclPrepareDone);
ncclResult_t ofcclPrepareDone() {
  // ***** ncclGroupEnd() *****

  CUDACHECK(cudaGetDevice(&thrdCudaDev));
  // OFCCL_LOG(OFCCL, "<%lu> thrdCudaDev is %d", pthread_self(), thrdCudaDev);
  ncclResult_t ret = ncclSuccess;

  int front_of_panel = -1;

  // ***** ncclAsyncThreadPreconnect threads *****
  for (int i = 0; i <= ofcclCommListPanel; i++) {
    front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_OPS : ofcclCommListFront;
    for (int j = 0; j < front_of_panel; j++) {
      ofcclCommArgs args = ofcclCommList[i][j];
      // OFCCL_LOG(OFCCL, "i=%d, j=%d, ofcclCommListPanel=%d, ofcclCommListFront=%d, tempcomm=%p(at %p), args.comm->connect=%d", i, j, ofcclCommListPanel, ofcclCommListFront, ofcclCommList[i][j].comm, &(ofcclCommList[i][j].comm), args.comm->connect);
      // OFCCL_LOG(OFCCL, "i=%d, j=%d, ofcclCommListPanel=%d, ofcclCommListFront=%d, tempcomm=%p(at %p)", i, j, ofcclCommListPanel, ofcclCommListFront, ofcclCommList[i][j].comm, &(ofcclCommList[i][j].comm));
      // ***** 目前应该是不会执行 *****
      if (args.comm->connect) {
        pthread_create(&ofcclPrepareThreads[i][j], NULL, ofcclAsyncThreadPreconnect, &args);
      }
    }
  }
  for (int i = 0; i <= ofcclCommListPanel; i++) {
    front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_OPS : ofcclCommListFront;
    for (int j = 0; j < front_of_panel; j++) {
      ofcclCommArgs args = ofcclCommList[i][j];
      if (args.comm->connect) {
        int err = pthread_join(ofcclPrepareThreads[i][j], NULL);
        if (err != 0) {
          OFCCL_LOG(OFCCL_WARN, "Error waiting for pthread_join : %s", strerror(errno));
          return ncclSystemError;
        }
        NCCLCHECKGOTO(args.ret, ret, end);
        args.comm->connect = 0;
        }
    }
  }

  // ***** Skip p2p nChannel deciding for now *****
  // ***** Skip related methods like scheduleRecv(), ncclSetupP2pKernel(), etc *****

  // ***** first for loop *****
  for (int i = 0; i <= ofcclCommListPanel; i++) {
    front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_OPS : ofcclCommListFront;
    for (int j = 0; j < front_of_panel; j++) {
      ofcclCommArgs args = ofcclCommList[i][j];
      ncclComm_t comm = args.comm;
      NCCLCHECKGOTO(ofcclSetupAsyncKernels(comm), ret, end);
    }
  }
  
  // ***** second for loop *****
  for (int i = 0; i <= ofcclCommListPanel; i++) {
    front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_OPS : ofcclCommListFront;
    for (int j = 0; j < front_of_panel; j++) {
      ofcclCommArgs args = ofcclCommList[i][j];
      ncclComm_t comm = args.comm;

      // ***** omit cudaSetDevice related to stream *****
      // ***** omit ncclCudaGraphHostSetup *****

      // ***** ncclEnqueueHostSetup<0> *****
      ofcclEnqueueHostSetup(comm->enqueueInfo);

      // ***** omit ncclLaunchBarrier *****
    }
  }

  // ***** check & 构建任务列表 *****
  for (int i = 0; i <= ofcclCommListPanel; i++) {
    front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_OPS : ofcclCommListFront;
    for (int j = 0; j < front_of_panel; j++) {
      ofcclCommArgs args = ofcclCommList[i][j];
      ncclComm_t comm = args.comm;
      struct ncclQueueInfo* eqInfo = comm->enqueueInfo;
      if (eqInfo->elemList->count() > 1) {
        ret = ncclInvalidUsage;
        OFCCL_LOG1(OFCCL_WARN, "eqInfo->elemList->count() shouldn't be larger than 1");
        goto end;
      }
      for (int k = 0; k < eqInfo->maxChannels; k++) {
        struct ncclChannel channel = comm->channels[k];
        // %NCCL_MAX_OPS
        // OFCCL_LOG(OFCCL, "pthread_id=%lu, %dth comm(comm->nChannels=%d), %dth channel, channel.workCount=%d, channel.workFifoTail=%lu, channel.index=%u", pthread_self(), j, comm->nChannels, k, channel.workCount, channel.workFifoTail, channel.index);
        for (int l = 0; l < channel.workFifoTail; l++) {
          if (l > 1) {
            ret = ncclInvalidUsage;
            OFCCL_LOG1(OFCCL_WARN, "channel.workFifoTail shouldn't be larger than 1");
            goto end;
          }
          ncclWork *work = channel.workFifo + l;
          
          for(int e=0; e < NCCL_MAX_WORK_ELEMENTS && work->elems[e].header.type != ncclWorkTypeUnused; e += 1) {
            if (e > 1) {
              ret = ncclInvalidUsage;
              OFCCL_LOG1(OFCCL_WARN, "channel.workFifo[0].elems's count shouldn't be larger than 1");
              goto end;
            }
            // OFCCL_LOG(OFCCL, "elem.count=%lu, elem.nChannels=%d, elem.header.nWarps=%u, elem.header.funcIndex=%u, elem.lastChunkSize=%lu, elem.direct=%u", work->elems[e].count, work->elems[e].nChannels, work->elems[e].header.nWarps, work->elems[e].header.funcIndex, work->elems[e].lastChunkSize, work->elems[e].direct);
            // TODO: 构建任务列表
          }
        }
      }
    }
  }

  // ***** 在独立的线程中启动守护者kernel *****
  // 当前线程管理单独一个设备，所以用同步的malloc、memcpy应该是可以的。

  // TODO: 指定参数
  queueLength = 4;
  collCount = ofcclCommListFront + ofcclCommListPanel * MAX_ASYNC_OPS;

  // OFCCL_LOG(OFCCL, "<%lu> device %d participate in %d colls", pthread_self(), thrdCudaDev, collCount);
  
  sq = sqCreate(queueLength);
  cq = cqCreate(queueLength);

  checkRuntime(cudaMalloc(&cqes, collCount * sizeof(CQE)));
  tempCqes = (CQE *)calloc(collCount, sizeof(CQE));
  // TODO: ！！！！！在复杂场景中，每个rank看到的collId未必是连续的！！！！
  for (int i = 0; i < collCount; i++) {
    // TODO: 需要和传进来的comm的id联动。
    tempCqes[i].collId = i;
  }
  checkRuntime(cudaMemcpy(cqes, tempCqes, collCount * sizeof(CQE), cudaMemcpyHostToDevice));
  
  daemonKernelGridDim.x = 4;
  daemonKernelBlockDim.x = 8;

  checkRuntime(cudaMalloc(&BlkCount4Coll, collCount * sizeof(int)));
  tempBlkCount4Coll = (int *)malloc(collCount * sizeof(int));
  for (int i = 0; i < collCount; i++) {
    // TODO: 需要和实际的集合通信的解析结果联动
    tempBlkCount4Coll[i] = testBlkCnt4Coll(i);
    // OFCCL_LOG(OFCCL, "<%lu> rank=%d tempBlkCount4Coll[%d] = %d", pthread_self(), thrdCudaDev, i, tempBlkCount4Coll[i]);
  }
  checkRuntime(cudaMemcpy(BlkCount4Coll, tempBlkCount4Coll, collCount * sizeof(int), cudaMemcpyHostToDevice));

  // make sure Memcpy to BlkCount4Coll finish
  checkRuntime(cudaStreamCreate(&kernelStream));

  checkRuntime(cudaDeviceSynchronize());
  
  thrdArgs = { sq, cq, cqes, BlkCount4Coll, kernelStream, thrdCudaDev, daemonKernelGridDim, daemonKernelBlockDim };
  pthread_create(&kernelThrd, NULL, startKernel, &thrdArgs);
  // OFCCL_LOG(OFCCL, "<%lu> rank=%d create <%lu>, thrdArgs.cudaDev = %d", pthread_self(), thrdCudaDev, kernelThrd, thrdArgs.cudaDev);

end:
  CUDACHECK(cudaSetDevice(thrdCudaDev)); // do other clean-ups first before calling
  return ret;
}

NCCL_API(ncclResult_t, ofcclDestroy);
ncclResult_t ofcclDestroy() {
  // OFCCL_LOG1(OFCCL, "Enter ofcclDestroy");
  ncclResult_t ret = ncclSuccess;

  // TODO: 目前选择在client手动调用ofcclDestroy的时候，发送最终的quit
  SQE sqe = { -1, 0, (int)RingBuffer_logic_tail(sq), nullptr, nullptr, true };
  sqWrite(sq, &sqe, thrdCudaDev);

  pthread_join(kernelThrd, nullptr);
  checkRuntime(cudaFree(cqes));
  free(tempCqes);
  checkRuntime(cudaFree(BlkCount4Coll));
  free(tempBlkCount4Coll);
  sqDestroy(sq);
  cqDestroy(cq);

  // ***** seems do not need to transverse ofcclCommList *****
  
  ofcclCommListPanel = 0;
  ofcclCommListFront = 0;
  return ret;
}








 














ncclResult_t ofcclEnqueueCheck(struct ncclInfo *info) {
  try_make();
  return ncclSuccess;
}

//   for (int i = 0; i <= ofcclCommListPanel; i++) {
//     front_of_panel = i < ofcclCommListPanel ? MAX_ASYNC_PANELS : ofcclCommListFront;
//     for (int j = 0; j < front_of_panel; j++) {
//       ofcclCommArgs args = ofcclCommList[i][j];
//       ncclComm_t comm = args.comm;
//       int node = comm->node;
//       int nNodes = comm->nNodes;
//       int localRank = comm->localRank;

//       // Compute how much to split operations
//       // Natural step size matching buffer steps.
//       ssize_t stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
//       // Try to use all channels
//       int nChannelsMax = comm->p2pnChannelsPerPeer;
//       int nChannelsMin = nChannelsMax;
//       // Try to use all channels, but one channel per operation.
//       while (nChannelsMin*comm->nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;
//       // Avoid overloading channels with 8+ operations as we loose the sync warp, hence a bit of bandwidth.
//       while (nChannelsMax*comm->nRanks > comm->p2pnChannels*4 && nChannelsMax > 1) nChannelsMax /= 2;

//       while (comm->p2pSendCount > 0 || comm->p2pRecvCount > 0) {
//         // schedule delta 0, +1, -1, +2, -2, ...
//         // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
//         for (int d=0; d<=nNodes/4; d++) {
//           int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
//           int index = 0;
//           int delta = deltas[index];
// sched_delta:
//           uint32_t recvNode = (node+nNodes-delta)%nNodes;
//           uint32_t sendNode = (node+delta)%nNodes;
//           int steps = comm->maxLocalRanks;
//           for (int s=0; s<steps; s++) {
//             int recvIndex = (localRank-s+steps)%steps;
//             int recvPeer = recvIndex<comm->nodeRanks[recvNode].localRanks ? comm->nodeRanks[recvNode].localRankToRank[recvIndex] : -1;
//             int sendIndex = (localRank+s)%steps;
//             int sendPeer = sendIndex<comm->nodeRanks[sendNode].localRanks ? comm->nodeRanks[sendNode].localRankToRank[sendIndex] : -1;
//             struct ncclP2Pinfo* recv = recvPeer != -1 && comm->p2pRecvs[recvPeer] ? comm->p2pRecvs[recvPeer]->getNext() : NULL;
//             struct ncclP2Pinfo* send = sendPeer != -1 && comm->p2pSends[sendPeer] ? comm->p2pSends[sendPeer]->getNext() : NULL;
//             if (recv != NULL || send != NULL) {
//               ssize_t totRecvBytes = -1, totSendBytes = -1;
//               if (recv != NULL) totRecvBytes = recv->nbytes;
//               if (send != NULL) totSendBytes = send->nbytes;
//               if (recv) comm->p2pRecvCount--;
//               if (send) comm->p2pSendCount--;
//               if (recvPeer == comm->rank) { // Check self send/recv
//                 if (sendPeer != comm->rank) { OFCCL_LOG1(OFCCL_WARN, "Sendrecv schedule not aligned for self"); ret = ncclInternalError; goto group_cleanup; }
//                 if (send && recv == NULL) { OFCCL_LOG1(OFCCL_WARN, "Trying to send to self without a matching recv"); ret = ncclInvalidUsage; goto group_cleanup; }
//                 if (send == NULL && recv) { OFCCL_LOG1(OFCCL_WARN, "Trying to recv to self without a matching send"); ret = ncclInvalidUsage; goto group_cleanup; }
//               }
//               void* recvBuff = recv ? recv->buff : NULL;
//               void* sendBuff = send ? send->buff : NULL;
//               // After we recycle p2pSend/Recv, we're no longer allowed to dereference send or recv, only use them as boolean NULL/not NULL.
//               if (recv && comm->p2pRecvs[recvPeer]->peakNext() == NULL) comm->p2pRecvs[recvPeer]->recycle();
//               if (send && comm->p2pSends[sendPeer]->peakNext() == NULL) comm->p2pSends[sendPeer]->recycle();

//               ssize_t recvChunkSize = getP2pChunkSize(totRecvBytes, nChannelsMin, nChannelsMax, stepSize, SENDRECV_SLICEFACTOR*stepSize);
//               ssize_t sendChunkSize = getP2pChunkSize(totSendBytes, nChannelsMin, nChannelsMax, stepSize, SENDRECV_SLICEFACTOR*stepSize);

//               ssize_t sendOffset = 0;
//               ssize_t recvOffset = 0;
//               int sendRemaining = 1, recvRemaining = 1;
//               int chunk = 0;
//               do {
//                 // Shuffle channels with s intra-node, and delta inter-node. Inter-node, make sure
//                 // to use multiple channels to guarantee progress on all ranks from the same node.
//                 ssize_t recvbytes = totRecvBytes-recvOffset;
//                 ssize_t sendbytes = totSendBytes-sendOffset;
//                 if (recvbytes > recvChunkSize) { recvbytes = recvChunkSize; } else { recvRemaining = 0; }
//                 if (sendbytes > sendChunkSize) { sendbytes = sendChunkSize; } else { sendRemaining = 0; }
//                 // 0-bytes send/recv are considered as syncs. Make sure we only add syncs when requested
//                 // (total size == 0), otherwise set size to -1.
//                 if (sendbytes < 0 || (sendbytes == 0 && totSendBytes != 0)) send = NULL;
//                 if (recvbytes < 0 || (recvbytes == 0 && totRecvBytes != 0)) recv = NULL;
//                 if (recv) {
//                   NCCLCHECKGOTO(scheduleRecv(comm, recvPeer, chunk, recvbytes, ((char*)recvBuff)+recvOffset), ret, group_cleanup);
//                 }
//                 if (send) {
//                   NCCLCHECKGOTO(scheduleSend(comm, sendPeer, chunk, sendbytes, ((char*)sendBuff)+sendOffset), ret, group_cleanup);
//                 }
//                 recvOffset += recvChunkSize;
//                 sendOffset += sendChunkSize;
//                 chunk++;
//               } while (sendRemaining || recvRemaining);
//             }
//           }
//           index++;
//           if (index == 1 && deltas[1] == deltas[0]) index++;
//           if (index == 2 && deltas[2] == deltas[0]) index++;
//           if (index == 3 && deltas[3] == deltas[2]) index++;
//           if (index == 3 && deltas[3] == deltas[1]) index++;
//           if (index < 4) {
//             delta = deltas[index];
//             goto sched_delta;
//           }
//         }
//       }
//     }
//   }