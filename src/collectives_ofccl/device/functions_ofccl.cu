#include "devcomm.h"
#include "collectives_ofccl.h"
#include "common_ofccl.h"

#define OFCCL_FUNC5(func, algo, devredop, type, nullify) \
  MACRO_IF(nullify, nullptr, OFCCL_FUNC_NAME(func, algo, LL,     devredop, type)), \
  MACRO_IF(nullify, nullptr, OFCCL_FUNC_NAME(func, algo, LL128,  devredop, type)), \
  MACRO_IF(nullify, nullptr, OFCCL_FUNC_NAME(func, algo, SIMPLE, devredop, type))

#define OFCCL_FUNC4(func, devredop, type, nullify) \
  OFCCL_FUNC5(func, TREE,    devredop, type, nullify), \
  OFCCL_FUNC5(func, RING,    devredop, type, nullify), \
  OFCCL_FUNC5(func, COLLNET, devredop, type, nullify)

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Must be consistent with ncclDataType_t
#define OFCCL_FUNCS3A(func, devredop, nullForFloat) \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, uint8_t, 0), \
  OFCCL_FUNC4(func, devredop, int32_t, 0), \
  OFCCL_FUNC4(func, devredop, uint32_t, 0), \
  OFCCL_FUNC4(func, devredop, int64_t, 0), \
  OFCCL_FUNC4(func, devredop, uint64_t, 0), \
  OFCCL_FUNC4(func, devredop, half, nullForFloat), \
  OFCCL_FUNC4(func, devredop, float, nullForFloat), \
  OFCCL_FUNC4(func, devredop, double, nullForFloat), \
  OFCCL_FUNC4(func, devredop, __nv_bfloat16, nullForFloat)
#define OFCCL_FUNCS3B(func, devredop) \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0)
#else
// Must be consistent with ncclDataType_t
#define OFCCL_FUNCS3A(func, devredop, nullForFloat) \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, uint8_t, 0), \
  OFCCL_FUNC4(func, devredop, int32_t, 0), \
  OFCCL_FUNC4(func, devredop, uint32_t, 0), \
  OFCCL_FUNC4(func, devredop, int64_t, 0), \
  OFCCL_FUNC4(func, devredop, uint64_t, 0), \
  OFCCL_FUNC4(func, devredop, half, nullForFloat), \
  OFCCL_FUNC4(func, devredop, float, nullForFloat), \
  OFCCL_FUNC4(func, devredop, double, nullForFloat)
#define OFCCL_FUNCS3B(func, devredop) \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0), \
  OFCCL_FUNC4(func, devredop, int8_t, 0)
#endif

// Must be consistent with ncclRedOp_t
#define OFCCL_FUNCS2A(func) \
  OFCCL_FUNCS3A(func, Sum,        /*nullForFloat=*/0), \
  OFCCL_FUNCS3A(func, Prod,       /*nullForFloat=*/0), \
  OFCCL_FUNCS3A(func, Max,        /*nullForFloat=*/0), \
  OFCCL_FUNCS3A(func, Min,        /*nullForFloat=*/0), \
  OFCCL_FUNCS3A(func, PreMulSum,  /*nullForFloat=*/0), \
  OFCCL_FUNCS3A(func, SumPostDiv, /*nullForFloat=*/1)

#define OFCCL_FUNCS2B(func) \
  OFCCL_FUNCS3B(func, Sum), \
  OFCCL_FUNCS3B(func, Sum), \
  OFCCL_FUNCS3B(func, Sum), \
  OFCCL_FUNCS3B(func, Sum), \
  OFCCL_FUNCS3B(func, Sum), \
  OFCCL_FUNCS3B(func, Sum)

// Must be consistent with the ncclFuncSet enum
__device__ ncclKern_t ofcclFuncs[1+ncclNumTypes+NCCL_NUM_FUNCTIONS*ncclNumDevRedOps*ncclNumTypes*NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS] = {
// Don't try to initialize the host shadow copy of this device-side global
// variable. There is no host pointer to a device-side function, which
// confuses clang. This will be fixed in the next clang release.
#if __CUDA_ARCH__
  OFCCL_FUNC_NAME(SendRecv, RING, SIMPLE, Sum, int8_t),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, half),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, float),
  OFCCL_ONERANK_REDUCE_NAME(PreMulSum, double),
  #if defined(__CUDA_BF16_TYPES_EXIST__)
    OFCCL_ONERANK_REDUCE_NAME(PreMulSum, __nv_bfloat16),
  #endif
  OFCCL_FUNCS2B(Broadcast),
  OFCCL_FUNCS2A(Reduce),
  OFCCL_FUNCS2B(AllGather),
  OFCCL_FUNCS2A(ReduceScatter),
  OFCCL_FUNCS2A(AllReduce)
#endif
};

// Workaround for https://reviews.llvm.org/D55580
__device__ void ofcclWorkaroundClangD55580() {}
