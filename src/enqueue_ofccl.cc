/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue_ofccl.h"
#include "checks.h"
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
#include "nccl.h"
#include "transport.h"

#include <cstddef>
#include <cstdlib>
#include <cstring> // std::memcpy
#include <fstream>
#include <ios>
#include <pthread.h> // pthread_self()
#include <math.h> // floor()
#include <sched.h>
#include <algorithm> // max
#include <semaphore.h>

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

  int stepSize = info->comm->buffSizes[info->protocol]/NCCL_STEPS;
  int chunkSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->chunkSteps : 1;
  int sliceSteps = (info->protocol == NCCL_PROTO_SIMPLE && info->algorithm == NCCL_ALGO_RING) ? info->sliceSteps : 1;
  int chunkSize = stepSize*chunkSteps;

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

  int nChannels = elem->nChannels; // ofcclComputeColl里边直接复制getAlgoInfo里存在ncclInfo里的计算结果
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


// still use 同步的Malloc吗？感觉是可以的，因为相当于是每个rank的init部分，而且prepareDone里还调用了cudaDeviceSynchronize
static SQ *sqCreate(unsigned long long int length) {
  SQ *sq = nullptr;
  checkRuntime(cudaMallocHost(&sq, sizeof(SQ)));
  sq->length = length + 1;
  sq->head = 0;
  sq->tail = 0;
  checkRuntime(cudaMallocHost(&(sq->buffer), sq->length * sizeof(SQE)));
  pthread_mutex_init(&sq->mutex, nullptr);

  return sq;
}

static void sqDestroy(SQ *sq) {
  if (sq) {
    checkRuntime(cudaFreeHost(sq->buffer));
    checkRuntime(cudaFreeHost(sq));
  }
}

int sqWrite(SQ *sq, SQE *sqe, int rank, CallbackFunc callback, void *callbackArgs, ofcclRankCtx_t rankCtx) {
  pthread_mutex_lock(&sq->mutex);

  if (CpuSqFull(sq)) {
    // not an error; caller keeps trying.
    pthread_mutex_unlock(&sq->mutex);
    return -1;
  }
  *RingBufferGetTail(sq) = *sqe;
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> write in sqe of coll_id = %d counter=%d @ %llu", pthread_self(), rank, sqe->collId, sqe->counter, RingBufferLogicTail(sq));

  __sync_synchronize();

  sq->tail += 1;
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> commit write for coll_id = %d, sqHead=%llu, new sqTail is %llu", pthread_self(), rank, sqe->collId, CpuLogicSqHead(sq), RingBufferLogicTail(sq));

  pthread_mutex_unlock(&sq->mutex);

  if (sqe->collId != -1) {
    // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> set callback for coll_id = %d", pthread_self(), rankCtx->rank, sqe->collId);
    rankCtx->callbacks[sqe->collId] = callback;
    rankCtx->callbackArgList[sqe->collId] = callbackArgs;
  }


  // 每次收到sqe，都唤醒一下。
  sem_post(&rankCtx->getNewSqeSema);
  
  // 即便我们一个正常sqe都不插，直接插quit，poller线程也能正常工作。
  if (rankCtx->poll_start == 0) {
    pthread_mutex_lock(&rankCtx->poller_mutex);
    rankCtx->poll_start = 1;
    pthread_mutex_unlock(&rankCtx->poller_mutex);
  }
  
  return 0;
}

static CQ *cqCreate(unsigned long long int length) {
  CQ *cq = nullptr;
  checkRuntime(cudaMallocHost(&cq, sizeof(CQ)));
  cq->length = length + 1;
  cq->head = 0;
  cq->tail = 0;
  cq->frontier = 0;
  checkRuntime(cudaMallocHost(&(cq->buffer), cq->length * sizeof(CQE)));
  // pthread_mutex_init(&cq->mutex, nullptr);

  return cq;
}

static void cqDestroy(CQ *cq) {
  if (cq) {
    checkRuntime(cudaFreeHost(cq->buffer));
    checkRuntime(cudaFreeHost(cq));
  }
}
// thread_local static int tempRound = 0;
static int cqRead(CQ *cq, CQE *target, int rank) {
  // pthread_mutex_lock(&cq->mutex);

  if (CpuCqEmpty(cq)) {
    // pthread_mutex_unlock(&cq->mutex);
    return -1;
  }
  
  *target = *RingBufferGetHead(cq);
  
  __sync_synchronize();

  cq->head += 1;

  // pthread_mutex_unlock(&cq->mutex);

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

NCCL_API(ncclResult_t, ofcclInitRankCtx, ofcclRankCtx_t* rankCtx, int rank);
ncclResult_t ofcclInitRankCtx(ofcclRankCtx_t* rankCtx, int rank) {
  ncclResult_t ret = ncclSuccess;

  ofcclRankCtx *newOfcclRankCtx = (ofcclRankCtx *)calloc(1, sizeof(ofcclRankCtx));
  *rankCtx = newOfcclRankCtx;

  newOfcclRankCtx->rank = rank;
  newOfcclRankCtx->queueLength = QLen;
  newOfcclRankCtx->seenComms = std::unordered_set<ncclComm_t>();

  // OfcclRankCtx里边的各种指针的内存分配、cudaMemcpy、值的初始化还是放到ofcclPrepareDone里边做

  return ret;
}

ncclResult_t ofcclPrepareCollComm(struct ncclInfo *info, int collId, ofcclRankCtx_t rankCtx) {  
  ncclResult_t ret = ncclSuccess;
  
  if (rankCtx->collCount >= MAX_LENGTH || collId >= MAX_LENGTH) {
    OFCCL_LOG(OFCCL_WARN, "Too many async operations in progress, max is %llu, rankCtx->collCount = %d, collId = %d",
          MAX_LENGTH, rankCtx->collCount, collId);
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

  if (rankCtx->seenComms.find(info->comm) != rankCtx->seenComms.end()) {
    OFCCL_LOG1(OFCCL_WARN, "Reuse ncclComm is not allowed");
    ret = ncclInvalidUsage;
    goto end;
  }
  rankCtx->seenComms.insert(info->comm);

  // 调整为插到相应的collId那里。
  rankCtx->ofcclCommList[collId].comm = info->comm;
  rankCtx->hostCollIds[rankCtx->collCount++] = collId;

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

// volunteer quit 调整：这个函数调整成一个专门的cudaLaunchKernel的包装函数
void startKernel(ofcclRankCtx *rankCtx) {
  checkRuntime(cudaSetDevice(rankCtx->rank));
  
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, gridDim=(%d, %d, %d), blockDim=(%d, %d, %d)", pthread_self(), rankCtx->rank, rankCtx->daemonKernelGridDim.x, rankCtx->daemonKernelGridDim.y, rankCtx->daemonKernelGridDim.z, rankCtx->daemonKernelBlockDim.x, rankCtx->daemonKernelBlockDim.y, rankCtx->daemonKernelBlockDim.z);

  rankCtx->argsptrs[0] = &rankCtx->sq;
  rankCtx->argsptrs[1] = &rankCtx->cq;
  rankCtx->argsptrs[2] = &rankCtx->rank;
  rankCtx->argsptrs[3] = &rankCtx->collCount;
  rankCtx->argsptrs[4] = &rankCtx->globalCqes;
  rankCtx->argsptrs[5] = &rankCtx->globalBlkCount4Coll;
  rankCtx->argsptrs[6] = &rankCtx->globalThrdCount4Coll;
  rankCtx->argsptrs[7] = &rankCtx->globalCollIds;
  rankCtx->argsptrs[8] = &rankCtx->globalDevComm7WorkElems;
  rankCtx->argsptrs[9] = &rankCtx->globalBlk2CollId2CollCtx;
  rankCtx->argsptrs[10] = &rankCtx->globalVolunteerQuitCounter;
  rankCtx->argsptrs[11] = &rankCtx->finallyQuit;
  rankCtx->argsptrs[12] = &rankCtx->globalBlkStatus;
  rankCtx->argsptrs[13] = &rankCtx->barrierCnt;
  rankCtx->argsptrs[14] = &rankCtx->collCounters;

  struct cudaLaunchParams daemonKernelParam;
  daemonKernelParam.func = (void *)daemonKernel;
  daemonKernelParam.gridDim = rankCtx->daemonKernelGridDim;
  daemonKernelParam.blockDim = rankCtx->daemonKernelBlockDim;
  daemonKernelParam.sharedMem = 0;
  // daemonKernelParam.sharedMem = 60 * 1024 * MAX_LENGTH;
  daemonKernelParam.stream = rankCtx->kernelStream;
  daemonKernelParam.args = rankCtx->argsptrs;

  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, sq @ %p, cq @ %p, globalCqes @ %p, globalBlkCount4Coll @ %p, func @ %p, stream @ %p, args @ %p, collCount=%d", pthread_self(), rankCtx->rank, rankCtx->sq, rankCtx->cq, rankCtx->globalCqes, rankCtx->globalBlkCount4Coll, daemonKernelParam.func, daemonKernelParam.stream, daemonKernelParam.args, rankCtx->collCount);

  checkRuntime(cudaLaunchKernel(daemonKernelParam.func, daemonKernelParam.gridDim, daemonKernelParam.blockDim, daemonKernelParam.args, daemonKernelParam.sharedMem, daemonKernelParam.stream));
}

void *startPoller(void *args) {
  ofcclRankCtx *rankCtx = ((PollerArgs *)args)->rankCtx;

  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, startPoller thread START", pthread_self(), rankCtx->rank);
  
  while (true) {
    // 原来的while (rankCtx->poll_start == 0)扩展成下边这样
    pthread_mutex_lock(&rankCtx->poller_mutex);
    if (rankCtx->poll_start == 1) {
      pthread_mutex_unlock(&rankCtx->poller_mutex);
      break;
    }
    pthread_mutex_unlock(&rankCtx->poller_mutex);

    sched_yield();
  }

  while (true) {
    pthread_mutex_lock(&rankCtx->poller_mutex);
    if (rankCtx->poll_stop == 1) {
      pthread_mutex_unlock(&rankCtx->poller_mutex);
      // TODO: 保证poller退出之前，cq已经空了。（应该没必要）
      break;
    }
    pthread_mutex_unlock(&rankCtx->poller_mutex);

    CQE target;
    if (cqRead(rankCtx->cq, &target, rankCtx->rank) == -1) {
      sched_yield();
    } else {
      int collId = target.collId;
      // OFCCL_LOG(OFCCL, "<%lu> Rank<%d> get cqe for coll_id = %d, will invoke callback", pthread_self(), rankCtx->rank, collId);
      // *(rankCtx->collCounters + 2 + collId * COLL_COUNTER_INNER_SIZE + 0 * MAX_LENGTH * COLL_COUNTER_INNER_SIZE) += 1;
      rankCtx->callbacks[collId](collId, rankCtx->callbackArgList[collId]);
    }
  }

  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, startPoller thread prepare to exit", pthread_self(), rankCtx->rank);
  return nullptr;
}

void *startKernel7SqObserver(void *args) {
  ofcclRankCtx *rankCtx = ((ObserverThrdArgs *)args)->rankCtx;

  while (true) {
    pthread_mutex_lock(&rankCtx->observer_mutex);
    int noMoreSqes = rankCtx->noMoreSqes;
    pthread_mutex_unlock(&rankCtx->observer_mutex);
    if (!noMoreSqes) { // 发出quitSqe的时候，主线程的sqWrite仍然会sem_post，但是observer已经没有必要等了，接下来只需要等待kernel最终退出就好了。不过实际大多数情况是，observer已经阻塞在sem_wait了，下一次再循环过来不会再阻塞而已。
      sem_wait(&rankCtx->getNewSqeSema);
      // 这个函数返回，说明等来了一个新的sqe写入。
      // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, new sqe come, sq->head = %llu, sq->tail = %llu", pthread_self(), rankCtx->rank, CpuLogicSqHead(rankCtx->sq), RingBufferLogicTail(rankCtx->sq));
    }

    // TODO: 按理说这里用cudaStreamQuery查状态应该是等价的，不过高频反复轮询，可能会导致cuda本身的一些问题吧，就卡住了。先放掉这个bug吧。
    // cudaError_t result = cudaStreamQuery(rankCtx->kernelStream);

    // if (result == cudaSuccess) { // kernel执行完了，退出来了。
    //   // kernel此时没在运行，有三种情况：没有启动；volunteer quit；以及看到了quit sqe，最终退出。
    //   // 除了最后一种情况，其他两种情况都需要启动kernel。
      
    //   OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, kernel exits, *rankCtx->finallyQuit = %d", pthread_self(), rankCtx->rank, *rankCtx->finallyQuit);
    //   if (*rankCtx->finallyQuit) {
    //     return nullptr;
    //   }

    //   startKernel(rankCtx);
    // } else if (result == cudaErrorNotReady) {// kernel还在执行，啥也不用做，满足预期：收到sqe的时候，kernel还在执行，有机会看到新写入的sqe，然后要保证rankCtx->finallyQuit被kernel设置，才舍得退出，应该保证了最终一致性。
    //   // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, kernel STILL RUN, *rankCtx->finallyQuit = %d", pthread_self(), rankCtx->rank, *rankCtx->finallyQuit);
    // } else { // unexpected
    //   checkRuntime(result);
    // }

    checkRuntime(cudaStreamSynchronize(rankCtx->kernelStream)); // 阻塞等待kernel执行，就算不收SQE了，也反复等，直到kernel自己看到quit sqe，这应该对了，保证最终一致性。
    // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, kernel exits or not started, *rankCtx->finallyQuit = %d", pthread_self(), rankCtx->rank, *rankCtx->finallyQuit);
    if (*rankCtx->finallyQuit) {
      break;
    }
    startKernel(rankCtx);
    // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, start Kernel", pthread_self(), rankCtx->rank);
  }
  return nullptr;
}

void printBarrierCnt(ofcclRankCtx *rankCtx, std::ofstream &file, int barrierId) {
  for (int bid = 0; bid < rankCtx->daemonKernelGridDim.x; ++bid) {
    file << " (" << bid << ")";
    for (int tid = 0; tid < rankCtx->daemonKernelBlockDim.x; tid += WARP_SIZE) { //WARP_SIZE
      file << " <" << tid << ">[";
      int printCnt = 2;
      if (barrierId == 11) {
        printCnt = 3;
      } 
      else if (barrierId == 10) {
        printCnt = 4;
      }
      for (int i = 0; i < printCnt; ++i) {
        file << *(rankCtx->barrierCnt + i + barrierId * BARCNT_INNER_SIZE + tid * NUM_BARRIERS * BARCNT_INNER_SIZE + bid * rankCtx->daemonKernelBlockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE);
        if (i < printCnt - 1) {
          file << "-";
        }
      }
      file << "] ";
    }
    file << std::endl;
  }
}

void printCollCounter(ofcclRankCtx *rankCtx, std::ofstream &file, int counterId) {
  int printBlkNum = rankCtx->daemonKernelGridDim.x;
  bool needSum = true;
  if (counterId == 2) {
    printBlkNum = 1;
  }
  if (counterId == 5 || counterId == 6 || counterId == 0) {
    needSum = false;
  }

  for (int bid = 0; bid < printBlkNum; ++bid) {
    file << " (" << bid << ")";
    for (int collId = 0; collId < rankCtx->collCount; ++collId) {
      file << " {" << collId << "}" << *(rankCtx->collCounters + counterId + collId * COLL_COUNTER_INNER_SIZE + bid * MAX_LENGTH * COLL_COUNTER_INNER_SIZE);
    }
    file << std::endl;
  }
  if (printBlkNum > 1 && needSum) {
    file << " (BLK SUM)";
    for (int collId = 0; collId < rankCtx->collCount; ++collId) {
      unsigned long long sumCounter = 0;
      for (int bid = 0; bid < printBlkNum; ++bid) {
        sumCounter += *(rankCtx->collCounters + counterId + collId * COLL_COUNTER_INNER_SIZE + bid * MAX_LENGTH * COLL_COUNTER_INNER_SIZE);
      }
      file << " {" << collId << "}" << sumCounter;
    }
    file << std::endl;
  }
}

void printCollCounterCompareBlock(ofcclRankCtx *rankCtx, std::ofstream &file, int counterId) {
  int printBlkNum = rankCtx->daemonKernelGridDim.x;

  for (int collId = 0; collId < rankCtx->collCount; ++collId) {
    file << " {" << collId << "}<";  
    for (int bid = 0; bid < printBlkNum; ++bid) {
      file << *(rankCtx->collCounters + counterId + collId * COLL_COUNTER_INNER_SIZE + bid * MAX_LENGTH * COLL_COUNTER_INNER_SIZE);
      if (bid < printBlkNum - 1) {
        file << "-";
      }
    }
    file << ">";
  }
  file << std::endl;
}

void *startBarrierCntPrinter(void *args) {
  ofcclRankCtx *rankCtx = ((ObserverThrdArgs *)args)->rankCtx;
  std::string fileName = "/home/panlichen/work2/ofccl/log/barrierCnt-";
  fileName.append(std::to_string(rankCtx->rank)).append(".log");

  std::ofstream clean(fileName, std::ios_base::out);
  clean << "";
  clean.close();

  std::ofstream file(fileName, std::ios_base::app);

  int stick_to_counter = 2;

  while (true) {
    if (rankCtx->noMoreSqes) {
      if (stick_to_counter-- < 0) {
        break;
      }
    }
  
    // file << "Rank " << rankCtx->rank << " barrier @ wroker wait fail 0:" << std::endl;
    // printBarrierCnt(rankCtx, file, 0);

    // file << "Rank " << rankCtx->rank << " barrier @ worker transmit done 1:" << std::endl;
    // printBarrierCnt(rankCtx, file, 1);
    
    // file << "Rank " << rankCtx->rank << " barrier @ controller 2:" << std::endl;
    // printBarrierCnt(rankCtx, file, 2);

    // file << "Rank " << rankCtx->rank << " barrier @ ~Primitives begin 3:" << std::endl;
    // printBarrierCnt(rankCtx, file, 3);

    // file << "Rank " << rankCtx->rank << " barrier @ ~Primitives end 4:" << std::endl;
    // printBarrierCnt(rankCtx, file, 4);
    
    // file << "Rank " << rankCtx->rank << " ofcclBarrier @ daemonKernel begin 8:" << std::endl;
    // printBarrierCnt(rankCtx, file, 8);

    file << "Rank " << rankCtx->rank << " # enter traverseTaskQ & # direct return & # return 11:" << std::endl;
    printBarrierCnt(rankCtx, file, 11);

    // file << "Rank " << rankCtx->rank << " enter traverse for & leave traverse for & collId & executing 10:" << std::endl;
    // printBarrierCnt(rankCtx, file, 10);

    // file << "Rank " << rankCtx->rank << " ofcclBarrier @ loadCollCtx end 6:" << std::endl;
    // printBarrierCnt(rankCtx, file, 6);

    // file << "Rank " << rankCtx->rank << " # invoke ofcclFunc & # ofcclFunc return 15:" << std::endl;
    // printBarrierCnt(rankCtx, file, 15);

    // file << "Rank " << rankCtx->rank << " # enter RunWork.run & # RunWork.run return 16:" << std::endl;
    // printBarrierCnt(rankCtx, file, 16);

    // file << "Rank " << rankCtx->rank << " runRing begin & return 14:" << std::endl;
    // printBarrierCnt(rankCtx, file, 14);
    
    // file << "Rank " << rankCtx->rank << " ofcclBarrier @ before traverse done 13:" << std::endl;
    // printBarrierCnt(rankCtx, file, 13);

    // file << "Rank " << rankCtx->rank << " ofcclBarrier @ after traverse done 7:" << std::endl;
    // printBarrierCnt(rankCtx, file, 7);
    
    // file << "Rank " << rankCtx->rank << " ofcclBarrier @ before checkSQ7TidyTaskQ 12:" << std::endl;
    // printBarrierCnt(rankCtx, file, 12);

    // file << "Rank " << rankCtx->rank << " ofcclBarrier @ after checkSQ7TidyTaskQ 9:" << std::endl;
    // printBarrierCnt(rankCtx, file, 9);

    file << "Rank " << rankCtx->rank << " daemonKernel # start & # quit 5:" << std::endl;
    printBarrierCnt(rankCtx, file, 5);

    for (int bid = 0; bid < rankCtx->daemonKernelGridDim.x; ++bid) {
      file << "Rank " << rankCtx->rank << " Block " << bid << " totalCtxSwitchCnt=" << 
        *(rankCtx->barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 33 * NUM_BARRIERS * BARCNT_INNER_SIZE + bid * rankCtx->daemonKernelBlockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) << " totalVolunteerQuitCnt=" << 
        *(rankCtx->barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 34 * NUM_BARRIERS * BARCNT_INNER_SIZE + bid * rankCtx->daemonKernelBlockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) << " numActiveColls=" << 
        *(rankCtx->barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + 35 * NUM_BARRIERS * BARCNT_INNER_SIZE + bid * rankCtx->daemonKernelBlockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) << std::endl;
    }


    // for (int bid = 0; bid < rankCtx->daemonKernelGridDim.x; ++bid) {
    //   file << "Rank " << rankCtx->rank << " Block " << bid << " TaskQ: [ ";
    //   for (int i = 0; i < PrintTestQNum; i++) {
    //     file << *(rankCtx->barrierCnt + 0 + 8 * BARCNT_INNER_SIZE + (36 + i) * NUM_BARRIERS * BARCNT_INNER_SIZE + bid * rankCtx->daemonKernelBlockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) << " ";
    //   }
    //   file << "]" << std::endl;
    // }

    // file << "Rank " << rankCtx->rank << " # block prepare cqe for coll CC-0:" << std::endl;
    // printCollCounter(rankCtx, file, 0);
    
    // file << "Rank " << rankCtx->rank << " # block put into cq cqe->collId CC-3:" << std::endl;
    // printCollCounter(rankCtx, file, 3);
    
    // file << "Rank " << rankCtx->rank << " # block put into cq RingBufferGetTail(cq)->collId CC-4:" << std::endl;
    // printCollCounter(rankCtx, file, 4);
    
    // file << "Rank " << rankCtx->rank << " block expect to write coll at CC-5:" << std::endl;
    // printCollCounterCompareBlock(rankCtx, file, 5);
    
    // file << "Rank " << rankCtx->rank << " block actually write coll at CC-6:" << std::endl;
    // printCollCounter(rankCtx, file, 6);

    // file << "Rank " << rankCtx->rank << " # block update cq->tail for coll CC-1:" << std::endl;
    // printCollCounter(rankCtx, file, 1);
    
    // file << "Rank " << rankCtx->rank << " # callback for coll invoked in CPU poller CC-2:" << std::endl;
    // printCollCounter(rankCtx, file, 2);

    file << std::endl << std::endl << std::endl;

    sleep(1);
  }
  file.close();
  return nullptr;
}

// 为了volunteer Quit进行的调整
NCCL_API(ncclResult_t, ofcclFinalizeRankCtx7StartHostThrds, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclFinalizeRankCtx7StartHostThrds(ofcclRankCtx_t rankCtx) {
  ncclResult_t ret = ncclSuccess;
  
  // OFCCL_LOG(OFCCL_INFO, "Rank %d registers %d colls", rankCtx->rank, rankCtx->collCount);

  // int front_of_panel = -1;

  // ***** ncclAsyncThreadPreconnect threads *****
  for (int i = 0; i < rankCtx->collCount; i++) {
    ofcclCommArgs *args = rankCtx->ofcclCommList + rankCtx->hostCollIds[i];
    // ***** 目前应该是不会执行 *****
    if (args->comm->connect) {
      pthread_create(&rankCtx->ofcclPrepareThreads[rankCtx->hostCollIds[i]], NULL, ofcclAsyncThreadPreconnect, &args);
    }
  }
  for (int i = 0; i < rankCtx->collCount; i++) {
    ofcclCommArgs *args = rankCtx->ofcclCommList + rankCtx->hostCollIds[i];
    if (args->comm->connect) {
      int err = pthread_join(rankCtx->ofcclPrepareThreads[rankCtx->hostCollIds[i]], NULL);
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
  for (int i = 0; i < rankCtx->collCount; i++) {
    int collId = rankCtx->hostCollIds[i];
    ofcclCommArgs *args = rankCtx->ofcclCommList + collId;
    ncclComm_t comm = args->comm;
    NCCLCHECKGOTO(ofcclSetupAsyncKernels(comm), ret, end);
  }
  
  // ***** second for loop *****
  for (int i = 0; i < rankCtx->collCount; i++) {
    ofcclCommArgs *args = rankCtx->ofcclCommList + rankCtx->hostCollIds[i];
    ncclComm_t comm = args->comm;

    // ***** omit cudaSetDevice related to stream *****
    // ***** omit ncclCudaGraphHostSetup *****

    // ***** ncclEnqueueHostSetup<0> *****
    ofcclEnqueueHostSetup(comm->enqueueInfo);

    // ***** omit ncclLaunchBarrier *****
  }

  rankCtx->daemonKernelGridDim.x = 0;
  rankCtx->daemonKernelBlockDim.x = 0;

  for (int i = 0; i < rankCtx->collCount; i++) {
    int collId = rankCtx->hostCollIds[i];
    ofcclCommArgs *args = rankCtx->ofcclCommList + collId;
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
    rankCtx->hostDevComm7WorkElems[collId].comm = comm->devComm;
    rankCtx->hostDevComm7WorkElems[collId].first = comm->args;
    
    struct cudaLaunchParams *params = comm->myParams;
    rankCtx->daemonKernelGridDim = dim3(std::max(rankCtx->daemonKernelGridDim.x, params->gridDim.x));
    rankCtx->daemonKernelBlockDim = dim3(std::max(rankCtx->daemonKernelBlockDim.x, params->blockDim.x));
    rankCtx->gridDim4Coll[collId] = params->gridDim;
    rankCtx->blockDim4Coll[collId] = params->blockDim;
    
    OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, coll_id = %d, gridDim.x=%d, blockDim.x=%d, nBytes = %lu", pthread_self(), rankCtx->rank, collId, params->gridDim.x, params->blockDim.x, comm->asyncOps->nBytes);

    rankCtx->hostCqes[collId].collId = collId;
    rankCtx->hostBlkCount4Coll[collId] = rankCtx->gridDim4Coll[collId].x;
    rankCtx->hostThrdCount4Coll[collId] = rankCtx->blockDim4Coll[collId].x;

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

  // OFCCL_LOG(OFCCL, "<%lu> device %d participate in %d colls, rankCtx->daemonKernelGridDim.x=%d, rankCtx->daemonKernelBlockDim.x=%d, sizeof(CollCtx)=%lu, sizeof(CollCtxGroup)=%lu, offsetof(struct CollCtx, redOpArgs)=%lu, sizeof(ncclDevComm)=%lu, sizeof(ncclChannel)=%lu, sizeof(ncclWork)=%lu, offsetof(struct CollCtx, work)=%lu, sizeof(struct ncclWorkElem)=%lu, alignof(ncclDevComm)=%lu, alignof(ncclChannel)=%lu, alignof(CollCtx)=%lu", pthread_self(), rankCtx->rank, rankCtx->collCount, rankCtx->daemonKernelGridDim.x, rankCtx->daemonKernelBlockDim.x, sizeof(CollCtx), sizeof(CollCtxGroup), offsetof(CollCtx, redOpArgs), sizeof(ncclDevComm), sizeof(ncclChannel), sizeof(ncclWork), offsetof(CollCtx, work), sizeof(struct ncclWorkElem), alignof(ncclDevComm), alignof(ncclChannel), alignof(CollCtx));
  
  rankCtx->sq = sqCreate(rankCtx->queueLength);
  rankCtx->cq = cqCreate(rankCtx->queueLength);

  // TODO: 之后考虑换成ofccl/src/include/alloc.h里的宏。
  checkRuntime(cudaMalloc(&rankCtx->globalCqes, MAX_LENGTH * sizeof(CQE)));
  checkRuntime(cudaMemcpy(rankCtx->globalCqes, rankCtx->hostCqes, MAX_LENGTH * sizeof(CQE), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&rankCtx->globalBlkCount4Coll, MAX_LENGTH * sizeof(int)));
  checkRuntime(cudaMemcpy(rankCtx->globalBlkCount4Coll, rankCtx->hostBlkCount4Coll, MAX_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&rankCtx->globalThrdCount4Coll, MAX_LENGTH * sizeof(int)));
  checkRuntime(cudaMemcpy(rankCtx->globalThrdCount4Coll, rankCtx->hostThrdCount4Coll, MAX_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&rankCtx->globalCollIds, MAX_LENGTH * sizeof(int)));
  checkRuntime(cudaMemcpy(rankCtx->globalCollIds, rankCtx->hostCollIds, MAX_LENGTH * sizeof(int), cudaMemcpyHostToDevice));

  checkRuntime(cudaMalloc(&rankCtx->globalDevComm7WorkElems, MAX_LENGTH * sizeof(DevComm7WorkElem)));
  checkRuntime(cudaMemcpy(rankCtx->globalDevComm7WorkElems, rankCtx->hostDevComm7WorkElems, MAX_LENGTH * sizeof(DevComm7WorkElem), cudaMemcpyHostToDevice));

  checkRuntime(cudaStreamCreate(&rankCtx->kernelStream));

  checkRuntime(cudaMalloc(&rankCtx->globalBlk2CollId2CollCtx, rankCtx->daemonKernelGridDim.x * MAX_LENGTH * sizeof(CollCtx)));

  checkRuntime(cudaMalloc(&rankCtx->globalVolunteerQuitCounter, sizeof(int)));

  checkRuntime(cudaMallocHost(&rankCtx->finallyQuit, sizeof(int)));
  *rankCtx->finallyQuit = 0;

  checkRuntime(cudaMalloc(&rankCtx->globalBlkStatus, rankCtx->daemonKernelGridDim.x * sizeof(BlkStatus)));

#ifdef ARRAY_DEBUG_ON
  checkRuntime(cudaMallocHost(&rankCtx->barrierCnt, rankCtx->daemonKernelGridDim.x * rankCtx->daemonKernelBlockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE * sizeof(unsigned long long int)));
  checkRuntime(cudaMallocHost(&rankCtx->collCounters, rankCtx->daemonKernelGridDim.x * MAX_LENGTH * COLL_COUNTER_INNER_SIZE * sizeof(unsigned long long int)));
#endif

  // make sure Memcpy to globalBlkCount4Coll finish
  checkRuntime(cudaDeviceSynchronize());

  rankCtx->noMoreSqes = 0;
  pthread_mutex_init(&rankCtx->observer_mutex, nullptr);
  sem_init(&rankCtx->getNewSqeSema, 0, 0);

  pthread_mutex_init(&rankCtx->poller_mutex, nullptr);
  rankCtx->pollerArgs = { rankCtx };
  pthread_create(&rankCtx->poller, nullptr, startPoller, &rankCtx->pollerArgs);

  rankCtx->observerThrdArgs = { rankCtx };
  pthread_create(&rankCtx->kernel7SqObserver, nullptr, startKernel7SqObserver, &rankCtx->observerThrdArgs);

#ifdef ARRAY_DEBUG_ON
  rankCtx->barrierCntPrinterArgs = { rankCtx };
  pthread_create(&rankCtx->barrierCntPrinter, nullptr, startBarrierCntPrinter, &rankCtx->barrierCntPrinterArgs);
#endif

end:
  return ret;
}

// !!!!!!!!!!!!!!!!! ALMOST DEPRECATED !!!!!!!!!!!!!!!!!
NCCL_API(ncclResult_t, ofcclPrepareDone, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclPrepareDone(ofcclRankCtx_t rankCtx) {
  // ***** ncclGroupEnd() *****
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(ofcclFinalizeRankCtx7StartHostThrds(rankCtx), ret, end);

  startKernel(rankCtx);

end:
  return ret;
}

NCCL_API(ncclResult_t, ofcclDestroy, ofcclRankCtx_t rankCtx);
ncclResult_t ofcclDestroy(ofcclRankCtx_t rankCtx) {
  // OFCCL_LOG1(OFCCL, "Enter ofcclDestroy");
  ncclResult_t ret = ncclSuccess;

  pthread_mutex_lock(&rankCtx->observer_mutex);
  rankCtx->noMoreSqes = 1;
  pthread_mutex_unlock(&rankCtx->observer_mutex);

  // 目前选择在client手动调用ofcclDestroy的时候，发送最终的quit
  SQE sqe = { -1, 0, nullptr, nullptr, true };
  sqWrite(rankCtx->sq, &sqe, rankCtx->rank, nullptr, nullptr, rankCtx);

  pthread_mutex_lock(&rankCtx->poller_mutex);
  rankCtx->poll_stop = 1;
  pthread_mutex_unlock(&rankCtx->poller_mutex);
  pthread_join(rankCtx->poller, nullptr);
  // OFCCL_LOG(OFCCL, "<%lu> Rank<%d>, pthread_join startPoller thread", pthread_self(), rankCtx->rank);

  pthread_join(rankCtx->kernel7SqObserver, nullptr);

#ifdef ARRAY_DEBUG_ON
  pthread_join(rankCtx->barrierCntPrinter, nullptr);
#endif

  checkRuntime(cudaFree(rankCtx->globalCqes));
  checkRuntime(cudaFree(rankCtx->globalBlkCount4Coll));
  checkRuntime(cudaFree(rankCtx->globalThrdCount4Coll));
  checkRuntime(cudaFree(rankCtx->globalCollIds));
  checkRuntime(cudaFree(rankCtx->globalDevComm7WorkElems));
  checkRuntime(cudaFree(rankCtx->globalBlk2CollId2CollCtx));
  checkRuntime(cudaFree(rankCtx->globalVolunteerQuitCounter));

#ifdef ARRAY_DEBUG_ON
  checkRuntime(cudaFreeHost(rankCtx->barrierCnt));
  checkRuntime(cudaFreeHost(rankCtx->collCounters));
#endif

  sqDestroy(rankCtx->sq);
  cqDestroy(rankCtx->cq);

  // ***** seems do not need to transverse ofcclCommList *****
  rankCtx->collCount = 0;
  return ret;
}








 














// 下边这部分主要是和单点的send、recv相关，所以目前没有支持。
//   for (int i = 0; i < collCount; i++) {
//     ofcclCommArgs *args = rankCtx->ofcclCommList + rankCtx->hostCollIds[i];
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