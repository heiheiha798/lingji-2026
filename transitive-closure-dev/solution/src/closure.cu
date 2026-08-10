#include "closure_api.h"

#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

struct DeviceResources {
  std::uint64_t *reachability = nullptr;
  std::uint64_t *pivot_rows = nullptr;
  std::uint64_t *pivot_masks = nullptr;
  cudaStream_t stream = nullptr;

  ~DeviceResources() {
    cudaFree(pivot_masks);
    cudaFree(pivot_rows);
    cudaFree(reachability);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

bool CudaOk(cudaError_t status) { return status == cudaSuccess; }

}  // namespace

extern "C" int closure_run(const std::uint64_t *adjacency,
                            std::uint64_t *reachability, int vertices,
                            int words_per_row) {
  if (adjacency == nullptr || reachability == nullptr || vertices < 1 ||
      words_per_row != (vertices + 63) / 64) {
    return 1;
  }
  const std::size_t bytes = static_cast<std::size_t>(vertices) *
                            words_per_row * sizeof(std::uint64_t);
  constexpr int kPivotBlock = 128;
  const std::size_t pivot_bytes = static_cast<std::size_t>(kPivotBlock) *
                                  words_per_row * sizeof(std::uint64_t);
  DeviceResources d;
  if (!CudaOk(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking)) ||
      !CudaOk(cudaMalloc(&d.reachability, bytes)) ||
      !CudaOk(cudaMalloc(&d.pivot_rows, pivot_bytes)) ||
      !CudaOk(cudaMalloc(&d.pivot_masks,
                         2 * kPivotBlock * sizeof(std::uint64_t)))) {
    return 2;
  }
  if (words_per_row <= 512) {
    cudaStreamAttrValue stream_attribute{};
    stream_attribute.accessPolicyWindow.base_ptr = d.pivot_rows;
    stream_attribute.accessPolicyWindow.num_bytes = pivot_bytes;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0F;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    if (!CudaOk(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                   pivot_bytes)) ||
        !CudaOk(cudaStreamSetAttribute(
            d.stream, cudaStreamAttributeAccessPolicyWindow,
            &stream_attribute))) {
      return 2;
    }
  }
  if (!CudaOk(cudaMemcpyAsync(d.reachability, adjacency, bytes,
                              cudaMemcpyHostToDevice, d.stream))) {
    return 2;
  }
  LaunchInitialize(d.reachability, vertices, words_per_row, d.stream);
  if (!CudaOk(cudaGetLastError())) {
    return 2;
  }
  for (int block_start = 0; block_start < vertices;
       block_start += kPivotBlock) {
    LaunchPivotBlock(d.reachability, d.pivot_rows, d.pivot_masks, vertices,
                     words_per_row, block_start, d.stream);
  }
  if (!CudaOk(cudaGetLastError()) ||
      !CudaOk(cudaMemcpyAsync(reachability, d.reachability, bytes,
                              cudaMemcpyDeviceToHost, d.stream)) ||
      !CudaOk(cudaStreamSynchronize(d.stream))) {
    return 2;
  }
  return 0;
}
