#include "reference.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void AddDiagonalAndMask(std::uint64_t *reach, int vertices,
                                   int words) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= vertices) return;
  reach[static_cast<std::size_t>(row) * words + row / 64] |=
      1ULL << (row & 63);
  if ((vertices & 63) != 0)
    reach[static_cast<std::size_t>(row) * words + words - 1] &=
        (1ULL << (vertices & 63)) - 1ULL;
}

__global__ void WarshallStep(std::uint64_t *reach, int vertices, int words,
                             int pivot) {
  const int row = blockIdx.x;
  if (row >= vertices) return;
  const std::size_t base = static_cast<std::size_t>(row) * words;
  if ((reach[base + pivot / 64] & (1ULL << (pivot & 63))) == 0) return;
  const std::size_t pivot_base = static_cast<std::size_t>(pivot) * words;
  for (int word = threadIdx.x; word < words; word += blockDim.x)
    reach[base + word] |= reach[pivot_base + word];
}

}  // namespace

void ReferenceSolve(const std::uint64_t *d_adjacency,
                    std::uint64_t *d_reachability, int vertices,
                    int words_per_row, cudaStream_t stream) {
  const std::size_t bytes =
      static_cast<std::size_t>(vertices) * words_per_row * sizeof(std::uint64_t);
  cudaMemcpyAsync(d_reachability, d_adjacency, bytes, cudaMemcpyDeviceToDevice,
                  stream);
  AddDiagonalAndMask<<<(vertices + 255) / 256, 256, 0, stream>>>(
      d_reachability, vertices, words_per_row);
  for (int pivot = 0; pivot < vertices; ++pivot)
    WarshallStep<<<vertices, 128, 0, stream>>>(d_reachability, vertices,
                                               words_per_row, pivot);
}
