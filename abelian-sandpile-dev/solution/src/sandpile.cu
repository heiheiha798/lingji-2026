#include "sandpile_api.h"

#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

struct DeviceResources {
  std::uint32_t *initial = nullptr;
  std::uint8_t *stable = nullptr;
  std::uint64_t *odometer = nullptr;
  std::uint64_t *height_a = nullptr;
  std::uint64_t *height_b = nullptr;
  int *active = nullptr;
  cudaStream_t stream = nullptr;

  ~DeviceResources() {
    cudaFree(active);
    cudaFree(height_b);
    cudaFree(height_a);
    cudaFree(odometer);
    cudaFree(stable);
    cudaFree(initial);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

bool CudaOk(cudaError_t status) { return status == cudaSuccess; }

}  // namespace

extern "C" int sandpile_run(const std::uint32_t *initial,
                             std::uint8_t *stable,
                             std::uint64_t *odometer, int rows, int cols) {
  if (initial == nullptr || stable == nullptr || odometer == nullptr ||
      rows < 1 || cols < 1) {
    return 1;
  }
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  DeviceResources d;
  if (!CudaOk(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking)) ||
      !CudaOk(cudaMalloc(&d.initial, n * sizeof(std::uint32_t))) ||
      !CudaOk(cudaMalloc(&d.stable, n * sizeof(std::uint8_t))) ||
      !CudaOk(cudaMalloc(&d.odometer, n * sizeof(std::uint64_t))) ||
      !CudaOk(cudaMalloc(&d.height_a, n * sizeof(std::uint64_t))) ||
      !CudaOk(cudaMalloc(&d.height_b, n * sizeof(std::uint64_t))) ||
      !CudaOk(cudaMalloc(&d.active, sizeof(int))) ||
      !CudaOk(cudaMemcpyAsync(d.initial, initial, n * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice, d.stream))) {
    return 2;
  }

  auto *a = d.height_a;
  auto *b = d.height_b;
  LaunchInitialize(d.initial, a, d.odometer, n, d.stream);
  if (!CudaOk(cudaGetLastError())) {
    return 2;
  }
  int host_active = 1;
  while (host_active != 0) {
    host_active = 0;
    if (!CudaOk(cudaMemsetAsync(d.active, 0, sizeof(int), d.stream))) {
      return 2;
    }
    LaunchSweep(a, b, d.odometer, rows, cols, d.active, d.stream);
    if (!CudaOk(cudaGetLastError()) ||
        !CudaOk(cudaMemcpyAsync(&host_active, d.active, sizeof(int),
                                cudaMemcpyDeviceToHost, d.stream)) ||
        !CudaOk(cudaStreamSynchronize(d.stream))) {
      return 2;
    }
    auto *tmp = a;
    a = b;
    b = tmp;
  }
  LaunchStore(a, d.stable, n, d.stream);
  if (!CudaOk(cudaGetLastError()) ||
      !CudaOk(cudaMemcpyAsync(stable, d.stable, n * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToHost, d.stream)) ||
      !CudaOk(cudaMemcpyAsync(odometer, d.odometer,
                              n * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost, d.stream)) ||
      !CudaOk(cudaStreamSynchronize(d.stream))) {
    return 2;
  }
  return 0;
}
