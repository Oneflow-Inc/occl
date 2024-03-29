#ifndef OFCCL_DEVICE_COMMON_H_
#define OFCCL_DEVICE_COMMON_H_

#include "collectives_ofccl.h"
#include "debug.h"
#include "devcomm.h"
#include "op128_ofccl.h"

extern __shared__ CollCtx sharedCollCtx[NUM_SHMEM_SLOT];
extern __shared__ BlkStatus blkStatus;

// ***** 这些文件夹内部用的宏，而且本身不带NCCL关键字的，可以不改名 *****
#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#else
#define COLL_UNROLL 4
#endif

#define OFCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

// ***** 接下来可以忽略所有和global函数定义相关的内容 *****
typedef void(*ofcclKern_t)();
extern __device__ ofcclKern_t ofcclFuncs[];

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

// Don't use barrier 0 as it's used by the final sync，8和15也都被占了。
inline __device__ void ofcclBarrier(int barId, int numThreads=blockDim.x) {
  asm volatile("bar.sync %0, %1;" :: "r"(barId), "r"(numThreads));
}

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run(ncclWorkElem *w) {
    // *(blkStatus.barrierCnt + 0 + 16 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;

    RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(w);

    // *(blkStatus.barrierCnt + 1 + 16 * BARCNT_INNER_SIZE + threadIdx.x * NUM_BARRIERS * BARCNT_INNER_SIZE + blockIdx.x * blockDim.x * NUM_BARRIERS * BARCNT_INNER_SIZE) += 1;
  }
};

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void OFCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
  RunWork<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&sharedCollCtx[blkStatus.currLoadedCollId % NUM_SHMEM_SLOT].staticCollCtx.workElem); \
}

#define IMPL_COLL4(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type)

#define IMPL_COLL3(func, devredop, type, ncclType) \
  IMPL_COLL4(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET, devredop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, double,   ncclFloat64)
#elif NCCL_TYPE == 9 && defined(__CUDA_BF16_TYPES_EXIST__)
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __nv_bfloat16, ncclBfloat16)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(func) IMPL_COLL2(func, Sum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(func) IMPL_COLL2(func, Prod);
#elif NCCL_OP == 2
#define IMPL_COLL_R(func) IMPL_COLL2(func, Min);
#elif NCCL_OP == 3
#define IMPL_COLL_R(func) IMPL_COLL2(func, Max);
#elif NCCL_OP == 4
#define IMPL_COLL_R(func) IMPL_COLL2(func, PreMulSum);
#elif NCCL_OP == 5
  #if NCCL_TYPE < 6
    #define IMPL_COLL_R(func) IMPL_COLL2(func, SumPostDiv);
  #else
    #define IMPL_COLL_R(func) // skip SumPostDiv for floating point
  #endif
#endif

// ***** 下边的还不会被用到 *****
#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

















#endif