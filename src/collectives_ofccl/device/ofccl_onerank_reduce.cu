#include "devcomm.h"
#include "collectives_ofccl.h"
#include "reduce_kernel.h"
#include "common_ofccl.h"

namespace {
  template<typename T, typename RedOp>
  __device__ __forceinline__ void oneRankReduce() {
  } 
}

#define INSTANTIATE(devredop, type) \
  __device__ void OFCCL_ONERANK_REDUCE_NAME(devredop, type)() { \
    oneRankReduce<type, Func##devredop<type>>(); \
  }

INSTANTIATE(PreMulSum, int8_t)
INSTANTIATE(PreMulSum, uint8_t)
INSTANTIATE(PreMulSum, int32_t)
INSTANTIATE(PreMulSum, uint32_t)
INSTANTIATE(PreMulSum, int64_t)
INSTANTIATE(PreMulSum, uint64_t)
INSTANTIATE(PreMulSum, half)
#if defined(__CUDA_BF16_TYPES_EXIST__)
INSTANTIATE(PreMulSum, __nv_bfloat16)
#endif
INSTANTIATE(PreMulSum, float)
INSTANTIATE(PreMulSum, double)
