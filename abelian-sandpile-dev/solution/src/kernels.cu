#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

namespace {

__global__ void InitializeDenseKernel(const std::uint32_t *input,
                                      std::uint32_t *height,
                                      std::uint64_t *odometer,
                                      std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    height[i] = input[i];
    odometer[i] = 0;
  }
}

__global__ void InitializeBoundedKernel(const std::uint32_t *input,
                                        std::uint32_t *height_a,
                                        std::uint32_t *height_b,
                                        std::uint64_t *odometer, int rows,
                                        int cols, int *bounds) {
  __shared__ int block_bounds[4];
  const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  if (thread_id == 0) block_bounds[0] = cols;
  if (thread_id == 1) block_bounds[1] = rows;
  if (thread_id == 2) block_bounds[2] = 0;
  if (thread_id == 3) block_bounds[3] = 0;
  __syncthreads();

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const bool valid = x < cols && y < rows;
  const std::size_t i = static_cast<std::size_t>(y) * cols + x;
  const unsigned int unstable_mask =
      __ballot_sync(0xffffffffU, valid && input[i] >= 4U);
  if (threadIdx.x == 0 && unstable_mask != 0U) {
    atomicMin(block_bounds,
              blockIdx.x * blockDim.x + __ffs(unstable_mask) - 1);
    atomicMin(block_bounds + 1, y);
    atomicMax(block_bounds + 2,
              blockIdx.x * blockDim.x + 32 - __clz(unstable_mask));
    atomicMax(block_bounds + 3, y + 1);
  }
  if (valid) {
    height_a[i] = input[i];
    height_b[i] = input[i];
    odometer[i] = 0;
  }
  __syncthreads();
  if (thread_id == 0 && block_bounds[0] < block_bounds[2]) {
    atomicMin(bounds, block_bounds[0]);
    atomicMin(bounds + 1, block_bounds[1]);
    atomicMax(bounds + 2, block_bounds[2]);
    atomicMax(bounds + 3, block_bounds[3]);
  }
}

template <bool CheckActive, bool Bounded>
__global__ void SweepKernel(const std::uint32_t *__restrict__ input,
                            std::uint32_t *__restrict__ output,
                            std::uint64_t *__restrict__ odometer, int rows,
                            int cols, int x_begin, int y_begin, int x_end,
                            int y_end, int *active) {
  const int x = (Bounded ? x_begin : 0) +
                blockIdx.x * blockDim.x + threadIdx.x;
  const int y = (Bounded ? y_begin : 0) +
                blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= (Bounded ? x_end : cols) || y >= (Bounded ? y_end : rows)) return;

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

void LaunchInitialize(const std::uint32_t *input, std::uint32_t *height_a,
                      std::uint32_t *height_b, std::uint64_t *odometer,
                      std::size_t n, int rows, int cols, int *bounds,
                      bool find_bounds, cudaStream_t stream) {
  if (find_bounds) {
    const dim3 block(32, 8);
    const dim3 grid((cols + block.x - 1) / block.x,
                    (rows + block.y - 1) / block.y);
    InitializeBoundedKernel<<<grid, block, 0, stream>>>(
        input, height_a, height_b, odometer, rows, cols, bounds);
  } else {
    InitializeDenseKernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0,
                            stream>>>(input, height_a, odometer, n);
  }
}

void LaunchSweep(const std::uint32_t *input, std::uint32_t *output,
                 std::uint64_t *odometer, int rows, int cols, int x_begin,
                 int y_begin, int x_end, int y_end, bool bounded, int *active,
                 cudaStream_t stream) {
  const dim3 block(32, 8);
  const unsigned int width = bounded ? x_end - x_begin : cols;
  const unsigned int height = bounded ? y_end - y_begin : rows;
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  if (bounded && active == nullptr) {
    SweepKernel<false, true><<<grid, block, 0, stream>>>(
        input, output, odometer, rows, cols, x_begin, y_begin, x_end, y_end,
        nullptr);
  } else if (bounded) {
    SweepKernel<true, true><<<grid, block, 0, stream>>>(
        input, output, odometer, rows, cols, x_begin, y_begin, x_end, y_end,
        active);
  } else if (active == nullptr) {
    SweepKernel<false, false><<<grid, block, 0, stream>>>(
        input, output, odometer, rows, cols, 0, 0, cols, rows, nullptr);
  } else {
    SweepKernel<true, false><<<grid, block, 0, stream>>>(
        input, output, odometer, rows, cols, 0, 0, cols, rows, active);
  }
}

void LaunchStore(const std::uint32_t *height, std::uint8_t *stable,
                 std::size_t n, cudaStream_t stream) {
  StoreKernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
      height, stable, n);
}
