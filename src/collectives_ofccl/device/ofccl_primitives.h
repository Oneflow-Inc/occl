#ifndef OFCCL_PRIMITIVES_H_
#define OFCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common_ofccl.h"

#define NCCL_SPINS_BEFORE_CHECK_ABORT 1000000 // 本身有退出的机制，不过没有恢复的方法

// TODO: now only Simple
/* Protocol classes: ProtoSimple, ProtoLL, ProtoLL128
 * We use these as template args to the Primtiives class instead of integral
 * enums (e.g. NCCL_PROTO_LL) because for SIMPLE we need to carry a few extra
 * numbers. Also these types hold methods which let us compute numbers important
 * to how that protocol operates with a consistent interface so that our
 * algorithm code can operate protocol parametrically.
 */
template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return sharedCollCtx.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 2;
  __device__ static int calcGroupWidth(bool send, int nthreads) {
    return send && nthreads-WARP_SIZE >= 64 ? 2 : 1;
  }
};

/* Fan (as in fan-in & fan-out) classes hold recv and send counts. The template
 * arguments are static bounds on the maximum values. Asymmetric counts are
 * independent. Symmetric is a static guarantee that nrecv==nsend, so it only
 * stores one value at runtime. This optimization save 32-bit register, but more
 * importantly uses fewer predicate registers when unrolling loops.
 */
template<int MaxRecv_, int MaxSend_>
struct FanAsymmetric {
  static constexpr int MaxRecv = MaxRecv_, MaxSend = MaxSend_;
  int nr, ns;
  FanAsymmetric() = default;
  __device__ FanAsymmetric(int nrecv, int nsend): nr(nrecv), ns(nsend) {
    // assert(nrecv <= MaxRecv && nsend <= MaxSend);
  }
  __device__ int nrecv() const { return MaxRecv ? nr : 0; }
  __device__ int nsend() const { return MaxSend ? ns : 0; }
};

template<int MaxArity>
struct FanSymmetric {
  static constexpr int MaxRecv = MaxArity, MaxSend = MaxArity;
  int n;
  FanSymmetric() = default;
  __device__ FanSymmetric(int nrecv, int nsend): n(nrecv) {
    // assert(nrecv == nsend && nrecv <= MaxArity);
  }
  __device__ int nrecv() const { return n; }
  __device__ int nsend() const { return n; }
};

// The primitives class. Specialized per protocol in the other headers.
template<typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p>
class Primitives;








#include "ofccl_prims_simple.h"
#endif