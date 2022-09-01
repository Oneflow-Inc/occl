#ifndef OFCCL_COLLECTIVES_H_
#define OFCCL_COLLECTIVES_H_

#include "collectives.h"

// ***** 先不要定义ofccl版本的ncclDevRedOp_t, ncclDevRedOpFull, 这个在其他地方有使用 *****

// ***** 保留FUNC_INDEX *****

// ***** 现在应该只需要个funcName *****
// TODO: 可能需要初始化和真正执行两份function name；也可能只搞一个function，不过通过参数区分是初始化还是真正执行。倾向于后者。
#define OFCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  ofcclFunction_##func##_##algo##_##proto##_##devredop##_##type

#define OFCCL_ONERANK_REDUCE_NAME(devredop, type) \
  ofcclFunction_OneRankReduce_##devredop##_##type

// ***** 我们只需要function，不需要kernel，即只需要device，不需要global *****
#define OFCCL_DECL5(func, algo, proto, devredop, type) \
  extern __device__ void OFCCL_FUNC_NAME(func, algo, proto, devredop, type)();

#define OFCCL_DECL4(func, algo, devredop, type, undef) \
  MACRO_IF(undef, /*undefined*/, OFCCL_DECL5(func, algo, SIMPLE, devredop, type)) \
  MACRO_IF(undef, /*undefined*/, OFCCL_DECL5(func, algo, LL,     devredop, type)) \
  MACRO_IF(undef, /*undefined*/, OFCCL_DECL5(func, algo, LL128,  devredop, type))

#define OFCCL_DECL3(func, devredop, type, undef) \
  OFCCL_DECL4(func, RING,    devredop, type, undef) \
  OFCCL_DECL4(func, TREE,    devredop, type, undef) \
  OFCCL_DECL4(func, COLLNET, devredop, type, undef)

#if defined(__CUDA_BF16_TYPES_EXIST__)
#define OFCCL_DECL2(func, devredop, undefForFloat) \
  OFCCL_DECL3(func, devredop, int8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, double, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, __nv_bfloat16, /*undef=*/undefForFloat)
#else
#define OFCCL_DECL2(func, devredop, undefForFloat) \
  OFCCL_DECL3(func, devredop, int8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint8_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint32_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, int64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, uint64_t, /*undef=*/0) \
  OFCCL_DECL3(func, devredop, half, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, float, /*undef=*/undefForFloat) \
  OFCCL_DECL3(func, devredop, double, /*undef=*/undefForFloat)
#endif

#define OFCCL_DECL(func) \
  OFCCL_DECL2(func, Sum, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, Prod, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, Min, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, Max, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, PreMulSum, /*undefForFloat=*/0) \
  OFCCL_DECL2(func, SumPostDiv, /*undefForFloat=*/1)

OFCCL_DECL2(Broadcast, Sum, /*undefForFloat=*/0)
OFCCL_DECL(Reduce)
OFCCL_DECL2(AllGather, Sum, /*undefForFloat=*/0)
OFCCL_DECL(ReduceScatter)
OFCCL_DECL(AllReduce)
OFCCL_DECL5(SendRecv, RING, SIMPLE, Sum, int8_t)


extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int8_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint8_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int32_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint32_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, int64_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, uint64_t)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, half)();
#if defined(__CUDA_BF16_TYPES_EXIST__)
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, __nv_bfloat16)();
#endif
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, float)();
extern __device__ void OFCCL_ONERANK_REDUCE_NAME(PreMulSum, double)();

#endif