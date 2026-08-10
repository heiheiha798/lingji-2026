#include "reference.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace reference_impl {

constexpr std::size_t kAlignment = 256;

std::size_t AlignUp(std::size_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

__global__ void Initialize(const std::uint32_t *input, std::uint64_t *height,
                           std::uint64_t *odometer, std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    height[i] = input[i];
    odometer[i] = 0;
  }
}

__global__ void Sweep(const std::uint64_t *__restrict__ input,
                      std::uint64_t *__restrict__ output,
                      std::uint64_t *__restrict__ odometer, int rows, int cols,
                      int *active) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= cols || y >= rows) return;
  const std::size_t i = static_cast<std::size_t>(y) * cols + x;
  const std::uint64_t h = input[i];
  const std::uint64_t q = h >> 2;
  std::uint64_t next = h & 3U;
  if (x > 0) next += input[i - 1] >> 2;
  if (x + 1 < cols) next += input[i + 1] >> 2;
  if (y > 0) next += input[i - cols] >> 2;
  if (y + 1 < rows) next += input[i + cols] >> 2;
  output[i] = next;
  odometer[i] += q;
  if (q != 0) atomicExch(active, 1);
}

__global__ void Store(const std::uint64_t *height, std::uint8_t *stable,
                      std::size_t n) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) stable[i] = static_cast<std::uint8_t>(height[i]);
}

}  // namespace reference_impl

std::size_t ReferenceWorkspaceSize(int rows, int cols) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  return reference_impl::AlignUp(2 * n * sizeof(std::uint64_t)) +
         reference_impl::kAlignment;
}

void ReferenceSolve(const std::uint32_t *d_initial, std::uint8_t *d_stable,
                    std::uint64_t *d_odometer, int rows, int cols,
                    void *d_workspace, cudaStream_t stream,
                    std::uint64_t *iterations) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  auto *a = static_cast<std::uint64_t *>(d_workspace);
  auto *b = a + n;
  auto *active = reinterpret_cast<int *>(
      static_cast<std::uint8_t *>(d_workspace) +
      reference_impl::AlignUp(2 * n * sizeof(std::uint64_t)));
  reference_impl::Initialize<<<static_cast<unsigned>((n + 255) / 256), 256, 0,
                               stream>>>(d_initial, a, d_odometer, n);
  const dim3 block(32, 8);
  const dim3 grid((cols + block.x - 1) / block.x,
                  (rows + block.y - 1) / block.y);
  int host_active = 1;
  std::uint64_t rounds = 0;
  while (host_active != 0) {
    host_active = 0;
    cudaMemsetAsync(active, 0, sizeof(int), stream);
    reference_impl::Sweep<<<grid, block, 0, stream>>>(a, b, d_odometer, rows,
                                                       cols, active);
    cudaMemcpyAsync(&host_active, active, sizeof(int), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    auto *tmp = a;
    a = b;
    b = tmp;
    ++rounds;
  }
  reference_impl::Store<<<static_cast<unsigned>((n + 255) / 256), 256, 0,
                          stream>>>(a, d_stable, n);
  if (iterations != nullptr) *iterations = rounds;
}

