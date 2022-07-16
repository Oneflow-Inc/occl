/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue_ofccl.h"
#include "argcheck.h"
#include "coll_net.h"
#include "debug.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "channel.h"
#include "nccl.h"

#include <cstring> // std::memcpy

namespace {
void try_make() {
  try_make_kern<<<8, 4>>>();
}

} // namespace

ncclResult_t ofcclEnqueueCheck(struct ncclInfo* info) {
  try_make();
  return ncclSuccess;
}
