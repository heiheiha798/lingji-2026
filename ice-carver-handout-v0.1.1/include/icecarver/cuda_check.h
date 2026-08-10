#pragma once

#include <cuda_runtime_api.h>

#include "icecarver/api.h"

namespace icecarver {

inline int cuda_status(cudaError_t status) noexcept {
  return status == cudaSuccess ? kSuccess : kCudaFailure;
}

inline int cuda_launch_status() noexcept {
  return cuda_status(cudaPeekAtLastError());
}

}  // namespace icecarver

#define ICECARVER_RETURN_IF_CUDA_ERROR(expression)                           \
  do {                                                                      \
    const cudaError_t icecarver_cuda_status_ = (expression);                \
    if (icecarver_cuda_status_ != cudaSuccess) {                            \
      return ::icecarver::kCudaFailure;                                     \
    }                                                                       \
  } while (false)

#define ICECARVER_RETURN_IF_LAUNCH_ERROR()                                  \
  do {                                                                      \
    if (cudaPeekAtLastError() != cudaSuccess) {                             \
      return ::icecarver::kCudaFailure;                                     \
    }                                                                       \
  } while (false)
