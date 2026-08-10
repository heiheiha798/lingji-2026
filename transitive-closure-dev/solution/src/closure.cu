#include "closure_api.h"

#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

struct DeviceResources {
  std::uint64_t *adjacency = nullptr;
  std::uint64_t *reachability = nullptr;
  cudaStream_t stream = nullptr;

  ~DeviceResources() {
    cudaFree(reachability);
    cudaFree(adjacency);
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
  DeviceResources d;
  if (!CudaOk(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking)) ||
      !CudaOk(cudaMalloc(&d.adjacency, bytes)) ||
      !CudaOk(cudaMalloc(&d.reachability, bytes)) ||
      !CudaOk(cudaMemcpyAsync(d.adjacency, adjacency, bytes,
                              cudaMemcpyHostToDevice, d.stream))) {
    return 2;
  }
  LaunchInitialize(d.adjacency, d.reachability, vertices, words_per_row,
                   d.stream);
  if (!CudaOk(cudaGetLastError())) {
    return 2;
  }
  for (int pivot = 0; pivot < vertices; ++pivot) {
    LaunchPivot(d.reachability, vertices, words_per_row, pivot, d.stream);
  }
  if (!CudaOk(cudaGetLastError()) ||
      !CudaOk(cudaMemcpyAsync(reachability, d.reachability, bytes,
                              cudaMemcpyDeviceToHost, d.stream)) ||
      !CudaOk(cudaStreamSynchronize(d.stream))) {
    return 2;
  }
  return 0;
}
