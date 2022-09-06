#ifndef OFCCL_DEVICE_COMMON_H_
#define OFCCL_DEVICE_COMMON_H_

#include "collectives_ofccl.h"
#include "debug.h"
#include "devcomm.h"
#include "op128_ofccl.h"
#include "ofccl_contexts.h"

extern __shared__ ofcclShmemData ofcclShmem;

// ***** 这些文件夹内部用的宏，而且本身不带NCCL关键字的，可以不改名 *****
#if __CUDA_ARCH__ >= 800
#define COLL_UNROLL 8
#else
#define COLL_UNROLL 4
#endif

#define OFCCL_MAX_DEV_ARITY (NCCL_MAX_TREE_ARITY-1)  // Using balanced tree instead of split tree

// ***** 接下来可以忽略所有和global函数定义相关的内容 *****
typedef void(*ncclKern_t)();
extern __device__ ncclKern_t ncclFuncs[];

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkElement {
  __device__ void run(ncclWorkElem*) {
    // Put NOT IMPLEMENTED behavior here.
  }
};

template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWork {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run(ncclWork *w) {
    int wid = threadIdx.x / WARP_SIZE;
    int inc = w->header.type == ncclWorkTypeRegColl ? sizeof(ncclWorkElemReg) / sizeof(ncclWorkElem) : 1;
    #pragma unroll 1
    for(int e=0; e < NCCL_MAX_WORK_ELEMENTS && w->elems[e].header.type != ncclWorkTypeUnused; e += inc) {
      if (wid < w->header.nWarps)
        RunWorkElement<Fn, T, RedOp, Algo, Proto>().run(&w->elems[e]);
        // if (threadIdx.x == 0) {
        //   OFCCL_LOG(NCCL, "bid(%d), blockDim.x(%d), ncclWorkElem(%d)(with %d channels, %lu bytes each, lastChunkSize=%lu) done", blockIdx.x, blockDim.x, e, w->elems[e].nChannels, w->elems[e].count, w->elems[e].lastChunkSize);
        // }
    }
  }
};

// TODO: 参数里直接用ofcclShmem.work吗？
// Examples :     AllReduce, RING, LL,    Sum,   uint8
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
__device__ void OFCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
  /*OFCCL_LOG(OFCCL, "in IMPL_COLL_FUNC, %s, %s, %s, %s, %s\n", #func, #algo, #proto, #devredop, #type);*/ /* 这里传模板参数的时候，应该是还是得传ncclFunc前缀的，这时定义在devcomm.h里的，我们应该不需要改 */ \
  RunWork<ncclFunc##func, type, Func##devredop<type>, NCCL_ALGO_##algo, NCCL_PROTO_##proto>().run(&ofcclShmem.work); \
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