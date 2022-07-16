#include "ofccl_all_reduce.h"
// #include "common.h"
#include "collectives_ofccl.h"

__global__ void try_make_kern() {
  printf("gridDim.x=%d, blockDim.x=%d, blockIdx=%d, threadIdx=%d\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x);
}