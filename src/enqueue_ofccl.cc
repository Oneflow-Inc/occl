/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue_ofccl.h"
#include "collectives_ofccl.h"
#include "debug.h"
#include "enqueue.h" // struct ncclQueueInfo
#include "argcheck.h"
#include "bootstrap.h"
#include "channel.h"
#include "coll_net.h"
#include "gdrwrap.h"
#include "group.h"
// #include "nccl.h"
#include "transport.h"

#include <cstddef>
#include <cstdlib>
#include <cstring> // std::memcpy
#include <pthread.h> // pthread_self()
#include <math.h> // floor()
#include <sched.h>
#include <algorithm> // max
#include <unordered_set>

namespace {

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

  // ***** Inline the first kernel *****
  // ***** but omit params->func = ncclKerns[work->header.funcIndex]; ****
  // ***** 而且我们应该也不需要 Only inline for channel 0 的限制 *****
  if (work->header.type == ncclWorkTypeColl) {
    memcpy(&comm->args, work->elems, sizeof(struct ncclWorkElem));
  }

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
    // 原来的nccl代码中，这里会尝试设置一个更大的segment：Try to pack more segments into a single operation
    // 我们的comm目前只被一个coll使用，所以没有这种需求。
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

    // ***** 把最后一个channel里的“正常操作”清零了，我们应该不需要这样。 *****
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
    ret = ncclInvalidUsage;
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

// 5 * 3 * 3
// TODO: algo = tree的时候，根据cuda版本不同可能有差别。allreduce里会根据cuda版本调用treeUpDown或者treeSplit
static int collExecContextCount[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS] = {
  {
    {
      0, //Broadcast, Tree, LL
      0, //Broadcast, Tree, LL128
      0  //Broadcast, Tree, Simple
    }, {
      0, //Broadcast, Ring, LL
      0, //Broadcast, Ring, LL128
      0  //Broadcast, Ring, Simple
    }, {
      0, //Broadcast, CollNet, LL
      0, //Broadcast, CollNet, LL128
      0  //Broadcast, CollNet, Simple
    }
  }, {
    {
      0, //Reduce, Tree, LL
      0, //Reduce, Tree, LL128
      0  //Reduce, Tree, Simple
    }, {
      0, //Reduce, Ring, LL
      0, //Reduce, Ring, LL128
      0  //Reduce, Ring, Simple
    }, {
      0, //Reduce, CollNet, LL
      0, //Reduce, CollNet, LL128
      0  //Reduce, CollNet, Simple
    }
  }, {
    {
      0, //AllGather, Tree, LL
      0, //AllGather, Tree, LL128
      0  //AllGather, Tree, Simple
    }, {
      0, //AllGather, Ring, LL
      0, //AllGather, Ring, LL128
      0  //AllGather, Ring, Simple
    }, {
      0, //AllGather, CollNet, LL
      0, //AllGather, CollNet, LL128
      0  //AllGather, CollNet, Simple
    }
  }, {
    {
      0, //ReduceScatter, Tree, LL
      0, //ReduceScatter, Tree, LL128
      0  //ReduceScatter, Tree, Simple
    }, {
      0, //ReduceScatter, Ring, LL
      0, //ReduceScatter, Ring, LL128
      0  //ReduceScatter, Ring, Simple
    }, {
      0, //ReduceScatter, CollNet, LL
      0, //ReduceScatter, CollNet, LL128
      0  //ReduceScatter, CollNet, Simple
    }
  }, {
    {
      0, //AllReduce, Tree, LL
      0, //AllReduce, Tree, LL128
      0  //AllReduce, Tree, Simple
    }, {
      0, //AllReduce, Ring, LL
      0, //AllReduce, Ring, LL128
      4  //AllReduce, Ring, Simple
    }, {
      0, //AllReduce, CollNet, LL
      0, //AllReduce, CollNet, LL128
      0  //AllReduce, CollNet, Simple
    }
  }
};

// TODO: 之后考虑搞一个Context整理起来。
static thread_local ofcclCommArgs ofcclCommList[MAX_LENGTH];
static thread_local pthread_t ofcclPrepareThreads[MAX_LENGTH];
static thread_local int collCount = 0;
static thread_local std::unordered_set<ncclComm_t> seenComms;

static thread_local dim3 daemonKernelGridDim;
static thread_local dim3 daemonKernelBlockDim;
static thread_local int queueLength = QLen;
static thread_local dim3 gridDim4Coll[MAX_LENGTH];
static thread_local dim3 blockDim4Coll[MAX_LENGTH];

static thread_local pthread_t kernelThrd;
static thread_local void *argsptrs[10];
static thread_local cudaStream_t kernelStream;
static thread_local KernelThrdArgs kernelThrdArgs;
static thread_local int thrdCudaDev;

static thread_local CQE hostCqes[MAX_LENGTH];
static thread_local CQE *globalCqes;
static thread_local int hostBlkCount4Coll[MAX_LENGTH];
static thread_local int *globalBlkCount4Coll;
static thread_local int hostThrdCount4Coll[MAX_LENGTH];
static thread_local int *globalThrdCount4Coll;
static thread_local int hostCollIds[MAX_LENGTH];
static thread_local int *globalCollIds;
static thread_local DevComm7WorkElem hostDevComm7WorkElems[MAX_LENGTH];
static thread_local DevComm7WorkElem *globalDevComm7WorkElems;
static thread_local CollCtx *globalBlk2CollId2CollCtx;

thread_local SQ *sq;
thread_local CQ *cq;

// for poller thread
static thread_local pthread_t poller;
static thread_local PollerArgs pollerArgs;
static thread_local int *poll_start;
static thread_local int *poll_stop;
// 之后主线程里sqWrite对这个的修改，希望poller线程可以看到
// static thread_local std::unordered_map<int, CallbackFunc> *collId2callback;
// 不能使用静态分配的数组，否则无法赋值，需要动态分配，跨线程传递。
static thread_local void **callbackArgList;
thread_local CallbackFunc *callbacks;


// still use 同步的Malloc吗？感觉是可以的，因为相当于是每个rank的init部分，而且prepareDone里还调用了cudaDeviceSynchronize
static SQ *sqCreate(int length) {
  SQ *sq = nullptr;
  checkRuntime(cudaMallocHost((void **)&sq, sizeof(SQ)));
  sq->length = length + 1;
  sq->head = 0;
  sq->tail = 0;
  checkRuntime(cudaMallocHost((void **)&(sq->buffer), sq->length * sizeof(SQE)));
  pthread_mutex_init(&sq->mutex, nullptr);

  return sq;
}

static void sqDestroy(SQ *sq) {
  if (sq) {
    checkRuntime(cudaFreeHost(sq->buffer));
    checkRuntime(cudaFreeHost(sq));
  }
}

int sqWrite(SQ *sq, SQE *sqe, int thrdCudaDev, CallbackFunc callback, void *callbackArgs) {
  // OFCCL_LOG_RANK_0(OFCCL, "<%lu> rank=%d, Enter sqWrite, sq @ %p", pthread_self(), thrdCudaDev, sq);
  pthread_mutex_lock(&sq->mutex);

  if (RingBuffer_full(sq)) {
    // not an error; caller keeps trying.
    pthread_mutex_unlock(&sq->mutex);
    return -1;
  }
  sqe->logicHead = (int)RingBuffer_logic_tail(sq);
  *RingBuffer_get_tail(sq) = *sqe;
  // OFCCL_LOG_RANK_0(OFCCL, "<%lu> write in sqe of collId %d counter=%d, quit=%d", pthread_self(), sqe->collId, sqe->counter, sqe->quit);

  __sync_synchronize();

  sq->tail += 1;
  // OFCCL_LOG_RANK_0(OFCCL, "<%lu> commit write, sqHead=%llu, new sqTail is %llu", pthread_self(), RingBuffer_logic_head(sq), RingBuffer_logic_tail(sq));

  pthread_mutex_unlock(&sq->mutex);

  if (sqe->collId != -1) {
    callbacks[sqe->collId] = callback;
    callbackArgList[sqe->collId] = callbackArgs;
  }
  // 即便我们一个正常sqe都不插，直接插quit，poller线程也能正常工作。
  if (*poll_start == 0) {
    *poll_start = 1;
  }
  
  return 0;
}

static CQ *cqCreate(int length) {
  CQ *cq = nullptr;
  checkRuntime(cudaMallocHost((void **)&cq, sizeof(CQ)));
  cq->length = length + 1;
  cq->head = 0;
  cq->tail = 0;
  checkRuntime(cudaMallocHost((void **)&(cq->buffer), cq->length * sizeof(CQE)));
  pthread_mutex_init(&cq->mutex, nullptr);

  return cq;
}

static void cqDestroy(CQ *cq) {
  if (cq) {
    checkRuntime(cudaFreeHost(cq->buffer));
    checkRuntime(cudaFreeHost(cq));
  }
}
// thread_local static int tempRound = 0;
static int cqRead(CQ *cq, CQE *target, int thrdCudaDev) {
  pthread_mutex_lock(&cq->mutex);
  // tempRound++;
  // if(tempRound % tempPrintRound == 0) {
  //   OFCCL_LOG(OFCCL, "<%lu> rank=%d enter cqRead, RingBuffer_empty(cq)=%d, cqHead=%llu, cqTail=%llu", pthread_self(), thrdCudaDev, RingBuffer_empty(cq), RingBuffer_logic_head(cq), RingBuffer_logic_tail(cq));
  // }

  if (RingBuffer_empty(cq)) {
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

void *ofcclAsyncThreadPreconnect(void* args_) {
  struct ofcclCommArgs* args = (struct ofcclCommArgs*)args_;
  CUDACHECKTHREAD(cudaSetDevice(args->comm->cudaDev));
  if (CPU_COUNT(&args->comm->cpuAffinity))
    sched_setaffinity(0, sizeof(cpu_set_t), &args->comm->cpuAffinity);
  NCCLCHECKTHREAD(ncclTransportP2pSetup(args->comm, NULL, 1));
  return args;
}

ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId) {
  if (collCount == 0) {
    memset(ofcclCommList, 0, sizeof(ncclComm_t) * MAX_LENGTH); // legacy in ncclGroupStart
  }
  
  ncclResult_t ret = ncclSuccess;
  
  if (collCount >= MAX_LENGTH || collId >= MAX_LENGTH) {
    OFCCL_LOG(OFCCL_WARN, "Too many async operations in progress, max is %d",
          MAX_LENGTH);
    ret = ncclInvalidUsage;
    goto end;
  }

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

  // 调整为插到相应的collId那里。
  ofcclCommList[collId].comm = info->comm;
  hostCollIds[collCount++] = collId;

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

  // Setup thread_local variables using values from parent thread.
  // OFCCL_LOG(OFCCL, "<%lu> thrdCudaDev in new thread is %d", pthread_self(), thrdCudaDev);
  sq = ((KernelThrdArgs *)args)->sq;
  cq = ((KernelThrdArgs *)args)->cq;
  kernelStream = ((KernelThrdArgs *)args)->stream;
  thrdCudaDev = ((KernelThrdArgs *)args)->cudaDev;
  daemonKernelGridDim = ((KernelThrdArgs *)args)->gridDim;
  daemonKernelBlockDim = ((KernelThrdArgs *)args)->blockDim;
  collCount = ((KernelThrdArgs *)args)->collCount;
  globalCqes = ((KernelThrdArgs *)args)->globalCqes;
  globalBlkCount4Coll = ((KernelThrdArgs *)args)->globalBlkCount4Coll;
  globalThrdCount4Coll = ((KernelThrdArgs *)args)->globalThrdCount4Coll;
  globalCollIds = ((KernelThrdArgs *)args)->globalCollIds;
  globalDevComm7WorkElems = ((KernelThrdArgs *)args)->globalDevComm7WorkElems;
  globalBlk2CollId2CollCtx = ((KernelThrdArgs *)args)->globalBlk2CollId2CollCtx;
  
  checkRuntime(cudaSetDevice(thrdCudaDev));

  // TODO: 之后考虑按需启停kernel
  
  // OFCCL_LOG(OFCCL, "<%lu> rank=%d after KernelThrd set daemonKernelGridDim, gridDimx=%d, blockDimx=%d", pthread_self(), thrdCudaDev, daemonKernelGridDim.x, daemonKernelBlockDim.x);

  argsptrs[0]  = &sq;
  argsptrs[1]  = &cq;
  argsptrs[2]  = &thrdCudaDev;
  argsptrs[3]  = &collCount;
  argsptrs[4]  = &globalCqes;
  argsptrs[5]  = &globalBlkCount4Coll;
  argsptrs[6]  = &globalThrdCount4Coll;
  argsptrs[7]  = &globalCollIds;
  argsptrs[8]  = &globalDevComm7WorkElems;
  argsptrs[9]  = &globalBlk2CollId2CollCtx;

  struct cudaLaunchParams daemonKernelParam;
  daemonKernelParam.func = (void *)daemonKernel;
  daemonKernelParam.gridDim = daemonKernelGridDim;
  daemonKernelParam.blockDim = daemonKernelBlockDim;
  daemonKernelParam.sharedMem = 0;
  // daemonKernelParam.sharedMem = 60 * 1024 * MAX_LENGTH;
  daemonKernelParam.stream = kernelStream;
  daemonKernelParam.args = argsptrs;

  // OFCCL_LOG(OFCCL, "<%lu> rank=%d, sq @ %p, cq @ %p, globalCqes @ %p, globalBlkCount4Coll @ %p, func @ %p, stream @ %p, args @ %p, collCount=%d", pthread_self(), thrdCudaDev, sq, cq, globalCqes, globalBlkCount4Coll, daemonKernelParam.func, daemonKernelParam.stream, daemonKernelParam.args, collCount);

  checkRuntime(cudaLaunchKernel(daemonKernelParam.func, daemonKernelParam.gridDim, daemonKernelParam.blockDim, daemonKernelParam.args, daemonKernelParam.sharedMem, daemonKernelParam.stream));

  // daemonKernel<<<gridDimx, blockDimx, 0, stream>>>(sq, cq, globalCqes, globalBlkCount4Coll);
  
  cudaStreamSynchronize(kernelStream);

  return NULL;
}

void *startPoller(void *args) {
  poll_start = ((PollerArgs *)args)->poll_start;
  poll_stop = ((PollerArgs *)args)->poll_stop;
  callbacks = ((PollerArgs *)args)->callbacks;
  thrdCudaDev = ((PollerArgs *)args)->cudaDev;
  checkRuntime(cudaSetDevice(thrdCudaDev));
  cq = ((PollerArgs *)args)->cq;
  callbackArgList = ((PollerArgs *)args)->callbackArgList;
  while (*poll_start == 0) {
    sched_yield();
  }

  while (*poll_stop == 0) {
    CQE target;
    if (cqRead(cq, &target, thrdCudaDev) == -1) {
      sched_yield();
    } else {
      int collId = target.collId;
      // OFCCL_LOG_RANK_0(OFCCL, "<%lu> rank=%d get cqe for collId %d, will invoke callback", pthread_self(), thrdCudaDev, collId);
      // OFCCL_LOG(OFCCL, "<%lu> rank=%d get cqe for collId %d, will invoke callback", pthread_self(), thrdCudaDev, collId);
      callbacks[collId](collId, callbackArgList[collId]);
    }
  }

  return nullptr;
}

NCCL_API(ncclResult_t, ofcclPrepareDone);
ncclResult_t ofcclPrepareDone() {
  // ***** ncclGroupEnd() *****

  CUDACHECK(cudaGetDevice(&thrdCudaDev));
  // OFCCL_LOG(OFCCL, "<%lu> thrdCudaDev is %d", pthread_self(), thrdCudaDev);
  ncclResult_t ret = ncclSuccess;

  // int front_of_panel = -1;

  // ***** ncclAsyncThreadPreconnect threads *****
  for (int i = 0; i < collCount; i++) {
    ofcclCommArgs *args = ofcclCommList + hostCollIds[i];
    // ***** 目前应该是不会执行 *****
    if (args->comm->connect) {
      pthread_create(&ofcclPrepareThreads[hostCollIds[i]], NULL, ofcclAsyncThreadPreconnect, &args);
    }
  }
  for (int i = 0; i < collCount; i++) {
    ofcclCommArgs *args = ofcclCommList + hostCollIds[i];
    if (args->comm->connect) {
      int err = pthread_join(ofcclPrepareThreads[hostCollIds[i]], NULL);
      if (err != 0) {
        OFCCL_LOG(OFCCL_WARN, "Error waiting for pthread_join : %s", strerror(errno));
        return ncclSystemError;
      }
      NCCLCHECKGOTO(args->ret, ret, end);
      args->comm->connect = 0;
    }
  }

  // ***** Skip p2p nChannel deciding for now *****
  // ***** Skip related methods like scheduleRecv(), ncclSetupP2pKernel(), etc *****

  // ***** first for loop *****
  for (int i = 0; i < collCount; i++) {
    int collId = hostCollIds[i];
    ofcclCommArgs *args = ofcclCommList + collId;
    ncclComm_t comm = args->comm;
    NCCLCHECKGOTO(ofcclSetupAsyncKernels(comm), ret, end);
  }
  
  // ***** second for loop *****
  for (int i = 0; i < collCount; i++) {
    ofcclCommArgs *args = ofcclCommList + hostCollIds[i];
    ncclComm_t comm = args->comm;

    // ***** omit cudaSetDevice related to stream *****
    // ***** omit ncclCudaGraphHostSetup *****

    // ***** ncclEnqueueHostSetup<0> *****
    ofcclEnqueueHostSetup(comm->enqueueInfo);

    // ***** omit ncclLaunchBarrier *****
  }

  daemonKernelGridDim.x = 0;
  daemonKernelBlockDim.x = 0;

  for (int i = 0; i < collCount; i++) {
    int collId = hostCollIds[i];
    ofcclCommArgs *args = ofcclCommList + collId;
    ncclComm_t comm = args->comm;
    struct ncclQueueInfo* eqInfo = comm->enqueueInfo;
    if (eqInfo->elemList->count() > 1) {
      ret = ncclInvalidUsage;
      OFCCL_LOG1(OFCCL_WARN, "eqInfo->elemList->count() shouldn't be larger than 1");
      goto end;
    }

    // TODO: 如果要支持其他work，这里要调整。
    if (comm->args.header.type == ncclWorkTypeUnused) {
      ret = ncclInvalidUsage;
      OFCCL_LOG1(OFCCL_WARN, "comm->args.header.type should be ncclWorkTypeColl(1)");
      goto end;
    }
    hostDevComm7WorkElems[collId].comm = comm->devComm;
    hostDevComm7WorkElems[collId].first = comm->args;
    
    struct cudaLaunchParams *params = comm->myParams;
    daemonKernelGridDim.x = std::max(daemonKernelGridDim.x, params->gridDim.x);
    daemonKernelBlockDim.x = std::max(daemonKernelBlockDim.x, params->blockDim.x);
    gridDim4Coll[collId] = params->gridDim;
    blockDim4Coll[collId] = params->blockDim;
    
    // OFCCL_LOG(OFCCL, "<%lu> rank=%d, comm of collId(%d) (comm->nChannels=%d), params->gridDim.x=%d, params->blockDim.x=%d", pthread_self(), thrdCudaDev, collId, comm->nChannels, params->gridDim.x, params->blockDim.x);

    hostCqes[collId].collId = collId;
    hostBlkCount4Coll[collId] = gridDim4Coll[collId].x;
    hostThrdCount4Coll[collId] = blockDim4Coll[collId].x;

    // check 确实一个comm对应了一个coll
    for (int k = 0; k < eqInfo->maxChannels; k++) {
      struct ncclChannel channel = comm->channels[k];
    
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
        }
      }
    }
  }

  // ***** 在独立的线程中启动守护者kernel *****
  // 当前线程管理单独一个设备，所以用同步的malloc、memcpy应该是可以的。

  // OFCCL_LOG(OFCCL, "<%lu> device %d participate in %d colls, daemonKernelGridDim.x=%d, daemonKernelBlockDim.x=%d, sizeof(CollCtx)=%lu, sizeof(CollCtxGroup)=%lu, offsetof(struct CollCtx, redOpArgs)=%lu, sizeof(ncclDevComm)=%lu, sizeof(ncclChannel)=%lu, sizeof(ncclWork)=%lu, offsetof(struct CollCtx, work)=%lu, sizeof(struct ncclWorkElem)=%lu, alignof(ncclDevComm)=%lu, alignof(ncclChannel)=%lu, alignof(CollCtx)=%lu", pthread_self(), thrdCudaDev, collCount, daemonKernelGridDim.x, daemonKernelBlockDim.x, sizeof(CollCtx), sizeof(CollCtxGroup), offsetof(CollCtx, redOpArgs), sizeof(ncclDevComm), sizeof(ncclChannel), sizeof(ncclWork), offsetof(CollCtx, work), sizeof(struct ncclWorkElem), alignof(ncclDevComm), alignof(ncclChannel), alignof(CollCtx));
  
  sq = sqCreate(queueLength);
  cq = cqCreate(queueLength);

  // TODO: 之后考虑换成ofccl/src/include/alloc.h里的宏。
  checkRuntime(cudaMalloc(&globalCqes, MAX_LENGTH * sizeof(CQE)));
  checkRuntime(cudaMemcpy(globalCqes, hostCqes, MAX_LENGTH * sizeof(CQE), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&globalBlkCount4Coll, MAX_LENGTH * sizeof(int)));
  checkRuntime(cudaMemcpy(globalBlkCount4Coll, hostBlkCount4Coll, MAX_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&globalThrdCount4Coll, MAX_LENGTH * sizeof(int)));
  checkRuntime(cudaMemcpy(globalThrdCount4Coll, hostThrdCount4Coll, MAX_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&globalCollIds, MAX_LENGTH * sizeof(int)));
  checkRuntime(cudaMemcpy(globalCollIds, hostCollIds, MAX_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&globalDevComm7WorkElems, MAX_LENGTH * sizeof(DevComm7WorkElem)));
  checkRuntime(cudaMemcpy(globalDevComm7WorkElems, hostDevComm7WorkElems, MAX_LENGTH * sizeof(DevComm7WorkElem), cudaMemcpyHostToDevice));

  checkRuntime(cudaStreamCreate(&kernelStream));

  checkRuntime(cudaMalloc(&globalBlk2CollId2CollCtx, daemonKernelGridDim.x * MAX_LENGTH * sizeof(CollCtx)));

  // make sure Memcpy to globalBlkCount4Coll finish
  checkRuntime(cudaDeviceSynchronize());
  
  kernelThrdArgs = { sq, cq, kernelStream, thrdCudaDev, daemonKernelGridDim, daemonKernelBlockDim, collCount, globalCqes, globalBlkCount4Coll, globalThrdCount4Coll, globalCollIds, globalDevComm7WorkElems, globalBlk2CollId2CollCtx };
  pthread_create(&kernelThrd, NULL, startKernel, &kernelThrdArgs);
  // OFCCL_LOG(OFCCL, "<%lu> rank=%d create <%lu>, kernelThrdArgs.cudaDev = %d", pthread_self(), thrdCudaDev, kernelThrd, kernelThrdArgs.cudaDev);

  poll_start = (int *)calloc(1, sizeof(int));
  poll_stop = (int *)calloc(1, sizeof(int));
  callbacks = (CallbackFunc *)calloc(MAX_LENGTH, sizeof(CallbackFunc));
  callbackArgList = (void **)calloc(MAX_LENGTH, sizeof(void *));
  pollerArgs = { poll_start, poll_stop, callbacks, thrdCudaDev, cq, callbackArgList };
  pthread_create(&poller, nullptr, startPoller, &pollerArgs);

end:
  CUDACHECK(cudaSetDevice(thrdCudaDev)); // do other clean-ups first before calling
  return ret;
}

NCCL_API(ncclResult_t, ofcclDestroy);
ncclResult_t ofcclDestroy() {
  // OFCCL_LOG1(OFCCL, "Enter ofcclDestroy");
  ncclResult_t ret = ncclSuccess;

  // 目前选择在client手动调用ofcclDestroy的时候，发送最终的quit
  SQE sqe = { -1, 0, (int)RingBuffer_logic_tail(sq), nullptr, nullptr, true };
  sqWrite(sq, &sqe, thrdCudaDev, nullptr, nullptr);

  pthread_join(kernelThrd, nullptr);
  *poll_stop = 1;
  pthread_join(poller, nullptr);

  checkRuntime(cudaFree(globalCqes));
  checkRuntime(cudaFree(globalBlkCount4Coll));
  sqDestroy(sq);
  cqDestroy(cq);

  free(poll_start);
  free(poll_stop);
  free(callbacks);
  free(callbackArgList);

  // ***** seems do not need to transverse ofcclCommList *****
  collCount = 0;
  return ret;
}








 














// 下边这部分主要是和单点的send、recv相关，所以目前没有支持。
//   for (int i = 0; i < collCount; i++) {
//     ofcclCommArgs *args = ofcclCommList + hostCollIds[i];
//     ncclComm_t comm = args->comm;
//     int node = comm->node;
//     int nNodes = comm->nNodes;
//     int localRank = comm->localRank;

//     // Compute how much to split operations
//     // Natural step size matching buffer steps.
//     ssize_t stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
//     // Try to use all channels
//     int nChannelsMax = comm->p2pnChannelsPerPeer;
//     int nChannelsMin = nChannelsMax;
//     // Try to use all channels, but one channel per operation.
//     while (nChannelsMin*comm->nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;
//     // Avoid overloading channels with 8+ operations as we loose the sync warp, hence a bit of bandwidth.
//     while (nChannelsMax*comm->nRanks > comm->p2pnChannels*4 && nChannelsMax > 1) nChannelsMax /= 2;

//     while (comm->p2pSendCount > 0 || comm->p2pRecvCount > 0) {
//       // schedule delta 0, +1, -1, +2, -2, ...
//       // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
//       for (int d=0; d<=nNodes/4; d++) {
//         int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
//         int index = 0;
//         int delta = deltas[index];
// sched_delta:
//         uint32_t recvNode = (node+nNodes-delta)%nNodes;
//         uint32_t sendNode = (node+delta)%nNodes;
//         int steps = comm->maxLocalRanks;
//         for (int s=0; s<steps; s++) {
//           int recvIndex = (localRank-s+steps)%steps;
//           int recvPeer = recvIndex<comm->nodeRanks[recvNode].localRanks ? comm->nodeRanks[recvNode].localRankToRank[recvIndex] : -1;
//           int sendIndex = (localRank+s)%steps;
//           int sendPeer = sendIndex<comm->nodeRanks[sendNode].localRanks ? comm->nodeRanks[sendNode].localRankToRank[sendIndex] : -1;
//           struct ncclP2Pinfo* recv = recvPeer != -1 && comm->p2pRecvs[recvPeer] ? comm->p2pRecvs[recvPeer]->getNext() : NULL;
//           struct ncclP2Pinfo* send = sendPeer != -1 && comm->p2pSends[sendPeer] ? comm->p2pSends[sendPeer]->getNext() : NULL;
//           if (recv != NULL || send != NULL) {
//             ssize_t totRecvBytes = -1, totSendBytes = -1;
//             if (recv != NULL) totRecvBytes = recv->nbytes;
//             if (send != NULL) totSendBytes = send->nbytes;
//             if (recv) comm->p2pRecvCount--;
//             if (send) comm->p2pSendCount--;
//             if (recvPeer == comm->rank) { // Check self send/recv
//               if (sendPeer != comm->rank) { OFCCL_LOG1(OFCCL_WARN, "Sendrecv schedule not aligned for self"); ret = ncclInternalError; goto group_cleanup; }
//               if (send && recv == NULL) { OFCCL_LOG1(OFCCL_WARN, "Trying to send to self without a matching recv"); ret = ncclInvalidUsage; goto group_cleanup; }
//               if (send == NULL && recv) { OFCCL_LOG1(OFCCL_WARN, "Trying to recv to self without a matching send"); ret = ncclInvalidUsage; goto group_cleanup; }
//             }
//             void* recvBuff = recv ? recv->buff : NULL;
//             void* sendBuff = send ? send->buff : NULL;
//             // After we recycle p2pSend/Recv, we're no longer allowed to dereference send or recv, only use them as boolean NULL/not NULL.
//             if (recv && comm->p2pRecvs[recvPeer]->peakNext() == NULL) comm->p2pRecvs[recvPeer]->recycle();
//             if (send && comm->p2pSends[sendPeer]->peakNext() == NULL) comm->p2pSends[sendPeer]->recycle();

//             ssize_t recvChunkSize = getP2pChunkSize(totRecvBytes, nChannelsMin, nChannelsMax, stepSize, SENDRECV_SLICEFACTOR*stepSize);
//             ssize_t sendChunkSize = getP2pChunkSize(totSendBytes, nChannelsMin, nChannelsMax, stepSize, SENDRECV_SLICEFACTOR*stepSize);

//             ssize_t sendOffset = 0;
//             ssize_t recvOffset = 0;
//             int sendRemaining = 1, recvRemaining = 1;
//             int chunk = 0;
//             do {
//               // Shuffle channels with s intra-node, and delta inter-node. Inter-node, make sure
//               // to use multiple channels to guarantee progress on all ranks from the same node.
//               ssize_t recvbytes = totRecvBytes-recvOffset;
//               ssize_t sendbytes = totSendBytes-sendOffset;
//               if (recvbytes > recvChunkSize) { recvbytes = recvChunkSize; } else { recvRemaining = 0; }
//               if (sendbytes > sendChunkSize) { sendbytes = sendChunkSize; } else { sendRemaining = 0; }
//               // 0-bytes send/recv are considered as syncs. Make sure we only add syncs when requested
//               // (total size == 0), otherwise set size to -1.
//               if (sendbytes < 0 || (sendbytes == 0 && totSendBytes != 0)) send = NULL;
//               if (recvbytes < 0 || (recvbytes == 0 && totRecvBytes != 0)) recv = NULL;
//               if (recv) {
//                 NCCLCHECKGOTO(scheduleRecv(comm, recvPeer, chunk, recvbytes, ((char*)recvBuff)+recvOffset), ret, group_cleanup);
//               }
//               if (send) {
//                 NCCLCHECKGOTO(scheduleSend(comm, sendPeer, chunk, sendbytes, ((char*)sendBuff)+sendOffset), ret, group_cleanup);
//               }
//               recvOffset += recvChunkSize;
//               sendOffset += sendChunkSize;
//               chunk++;
//             } while (sendRemaining || recvRemaining);
//           }
//         }
//         index++;
//         if (index == 1 && deltas[1] == deltas[0]) index++;
//         if (index == 2 && deltas[2] == deltas[0]) index++;
//         if (index == 3 && deltas[3] == deltas[2]) index++;
//         if (index == 3 && deltas[3] == deltas[1]) index++;
//         if (index < 4) {
//           delta = deltas[index];
//           goto sched_delta;
//         }
//       }
//     }
//   }