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
  void *height_a = nullptr;
  void *height_b = nullptr;
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
  std::uint32_t max_initial = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (initial[i] > max_initial) max_initial = initial[i];
  }
  const bool use_short_heights = max_initial <= 65535U;
  int sampled_active = 0;
  for (std::size_t sample = 0; sample < 256; ++sample) {
    const std::size_t i = sample * n / 256;
    sampled_active += initial[i] >= 4U;
  }
  const bool dense = sampled_active >= 4;
  const std::size_t height_bytes =
      n * (use_short_heights ? sizeof(std::uint16_t) : sizeof(std::uint32_t));
  DeviceResources d;
  if (!CudaOk(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking)) ||
      !CudaOk(cudaMalloc(&d.initial, n * sizeof(std::uint32_t))) ||
      !CudaOk(cudaMalloc(&d.stable, n * sizeof(std::uint8_t))) ||
      !CudaOk(cudaMalloc(&d.odometer, n * sizeof(std::uint64_t))) ||
      !CudaOk(cudaMalloc(&d.height_a, height_bytes)) ||
      !CudaOk(cudaMalloc(&d.height_b, height_bytes)) ||
      !CudaOk(cudaMalloc(&d.active, 5 * sizeof(int))) ||
      !CudaOk(cudaMemcpyAsync(d.initial, initial, n * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice, d.stream))) {
    return 2;
  }

  auto *a = d.height_a;
  auto *b = d.height_b;
  int host_control[5] = {0, cols, rows, 0, 0};
  if (!dense) {
    if (!CudaOk(cudaMemcpyAsync(d.active, host_control, sizeof(host_control),
                                cudaMemcpyHostToDevice, d.stream))) {
      return 2;
    }
  }
  LaunchInitialize(d.initial, a, b, d.odometer, n, rows, cols, d.active + 1,
                   !dense, use_short_heights, d.stream);
  if (!CudaOk(cudaGetLastError())) {
    return 2;
  }
  if (!dense) {
    if (!CudaOk(cudaMemcpyAsync(host_control + 1, d.active + 1,
                                4 * sizeof(int), cudaMemcpyDeviceToHost,
                                d.stream)) ||
        !CudaOk(cudaStreamSynchronize(d.stream))) {
      return 2;
    }
  }
  int x_begin = dense ? 0 : host_control[1];
  int y_begin = dense ? 0 : host_control[2];
  int x_end = dense ? cols : host_control[3];
  int y_end = dense ? rows : host_control[4];
  int host_active = x_begin < x_end ? 1 : 0;
  while (host_active != 0) {
    host_active = 0;
    if (!CudaOk(cudaMemsetAsync(d.active, 0, sizeof(int), d.stream))) {
      return 2;
    }
    for (int sweep = 0; sweep < 16; ++sweep) {
      if (x_begin > 0) --x_begin;
      if (y_begin > 0) --y_begin;
      if (x_end < cols) ++x_end;
      if (y_end < rows) ++y_end;
      LaunchSweep(a, b, d.odometer, rows, cols, x_begin, y_begin, x_end, y_end,
                  x_begin != 0 || y_begin != 0 || x_end != cols || y_end != rows,
                  sweep == 15 ? d.active : nullptr, use_short_heights,
                  d.stream);
      if (!CudaOk(cudaGetLastError())) {
        return 2;
      }
      auto *tmp = a;
      a = b;
      b = tmp;
    }
    if (!CudaOk(cudaMemcpyAsync(&host_active, d.active, sizeof(int),
                                cudaMemcpyDeviceToHost, d.stream)) ||
        !CudaOk(cudaStreamSynchronize(d.stream))) {
      return 2;
    }
  }
  LaunchStore(a, d.stable, n, use_short_heights, d.stream);
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
