#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void InitializeKernel(const std::uint32_t *input,
                                 std::uint32_t *height,
                                 std::uint64_t *odometer, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    height[i] = input[i];
    odometer[i] = 0;
  }
}

template <bool CheckActive>
__global__ void SweepKernel(const std::uint32_t *__restrict__ input,
                            std::uint32_t *__restrict__ output,
                            std::uint64_t *__restrict__ odometer, int rows,
                            int cols, int *active) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cols || y >= rows) return;

  const std::size_t i = static_cast<std::size_t>(y) * cols + x;
  const std::uint32_t h = input[i];
  const std::uint32_t q = h >> 2;
  std::uint32_t next = h & 3U;
  if (x > 0) next += input[i - 1] >> 2;
  if (x + 1 < cols) next += input[i + 1] >> 2;
  if (y > 0) next += input[i - cols] >> 2;
  if (y + 1 < rows) next += input[i + cols] >> 2;
  output[i] = next;
  if (q != 0) {
    odometer[i] += q;
    if constexpr (CheckActive) {
      atomicExch(active, 1);
    }
  }
}

__global__ void StoreKernel(const std::uint32_t *height,
                            std::uint8_t *stable, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) stable[i] = static_cast<std::uint8_t>(height[i]);
}

}  // namespace

void LaunchInitialize(const std::uint32_t *input, std::uint32_t *height,
                      std::uint64_t *odometer, std::size_t n,
                      cudaStream_t stream) {
  InitializeKernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      input, height, odometer, n);
}

void LaunchSweep(const std::uint32_t *input, std::uint32_t *output,
                 std::uint64_t *odometer, int rows, int cols, int *active,
                 cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((cols + block.x - 1) / block.x,
                  (rows + block.y - 1) / block.y);
  if (active == nullptr) {
    SweepKernel<false><<<grid, block, 0, stream>>>(input, output, odometer, rows,
                                                   cols, nullptr);
  } else {
    SweepKernel<true><<<grid, block, 0, stream>>>(input, output, odometer, rows,
                                                  cols, active);
  }
}

void LaunchStore(const std::uint32_t *height, std::uint8_t *stable,
                 std::size_t n, cudaStream_t stream) {
  StoreKernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      height, stable, n);
}
