#include "devcomm.h"
#include "collectives_ofccl.h"
#include "ofccl_primitives.h"
#include "debug.h"

template<typename T, typename RedOp>
struct RunWork<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void runSend(const int tid, const int nthreads, const int group, struct ncclWorkElemP2p* args) {
  }

  __device__ __forceinline__ void runRecv(const int tid, const int nthreads, const int group, struct ncclWorkElemP2p* args) {
  }
  
  __device__ __forceinline__ void run(ncclWork *work) {
  }
};