#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kPivotBlock = 64;

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

__global__ void ClosePivotBlock(const std::uint64_t *reach,
                                std::uint64_t *pivot_masks, int words,
                                int block_start, int block_size) {
  __shared__ std::uint64_t current_pivot;
  const int row = threadIdx.x;
  std::uint64_t row_mask = 0;
  if (row < block_size) {
    row_mask = reach[static_cast<std::size_t>(block_start + row) * words +
                     block_start / kPivotBlock];
    row_mask &= block_size == kPivotBlock
                    ? ~0ULL
                    : (1ULL << block_size) - 1ULL;
    row_mask |= 1ULL << row;
  }

  for (int pivot = 0; pivot < block_size; ++pivot) {
    if (row == pivot) current_pivot = row_mask;
    __syncthreads();
    if (row < block_size && (row_mask & (1ULL << pivot)) != 0)
      row_mask |= current_pivot;
    __syncthreads();
  }
  if (row < block_size) pivot_masks[row] = row_mask;
}

__global__ void BuildPivotRows(const std::uint64_t *reach,
                               std::uint64_t *pivot_rows,
                               const std::uint64_t *pivot_masks, int words,
                               int block_start, int block_size) {
  const int row = blockIdx.x;
  if (row >= block_size) return;
  const std::uint64_t mask = pivot_masks[row];
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    std::uint64_t output = 0;
    std::uint64_t remaining = mask;
    while (remaining != 0) {
      const int source = __ffsll(static_cast<long long>(remaining)) - 1;
      output |= reach[static_cast<std::size_t>(block_start + source) * words +
                      word];
      remaining &= remaining - 1;
    }
    pivot_rows[static_cast<std::size_t>(row) * words + word] = output;
  }
}

template <bool SkipEmpty>
__global__ void ApplyPivotBlock(std::uint64_t *reach,
                                const std::uint64_t *pivot_rows,
                                const std::uint64_t *pivot_masks,
                                int vertices, int words, int block_start,
                                int block_size) {
  const int row = blockIdx.x;
  if (row >= vertices) return;
  const std::size_t base = static_cast<std::size_t>(row) * words;
  std::uint64_t row_mask = reach[base + block_start / kPivotBlock];
  row_mask &= block_size == kPivotBlock
                  ? ~0ULL
                  : (1ULL << block_size) - 1ULL;
  if constexpr (SkipEmpty) {
    if (row_mask == 0) return;
  }
  std::uint64_t selected_mask = 0;
  if ((threadIdx.x & 31) == 0) {
    std::uint64_t remaining = row_mask;
    while (remaining != 0) {
      const int source = __ffsll(static_cast<long long>(remaining)) - 1;
      selected_mask |= 1ULL << source;
      remaining &= ~pivot_masks[source];
    }
  }
  selected_mask = __shfl_sync(0xffffffffU, selected_mask, 0);
  if (selected_mask == 0) return;

  if ((selected_mask & (selected_mask - 1)) == 0) {
    const int source = __ffsll(static_cast<long long>(selected_mask)) - 1;
    for (int word = threadIdx.x; word < words; word += blockDim.x) {
      reach[base + word] |=
          pivot_rows[static_cast<std::size_t>(source) * words + word];
    }
    return;
  }

  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    std::uint64_t output = reach[base + word];
    std::uint64_t remaining = selected_mask;
    while (remaining != 0) {
      const int source = __ffsll(static_cast<long long>(remaining)) - 1;
      output |= pivot_rows[static_cast<std::size_t>(source) * words + word];
      remaining &= remaining - 1;
    }
    reach[base + word] = output;
  }
}

}  // namespace

void LaunchInitialize(std::uint64_t *reachability, int vertices, int words,
                      cudaStream_t stream) {
  AddDiagonalAndMask<<<(vertices + 255) / 256, 256, 0, stream>>>(
      reachability, vertices, words);
}

void LaunchPivotBlock(std::uint64_t *reachability,
                      std::uint64_t *pivot_rows,
                      std::uint64_t *pivot_masks, int vertices, int words,
                      int block_start, cudaStream_t stream) {
  const int remaining = vertices - block_start;
  const int block_size = remaining < kPivotBlock ? remaining : kPivotBlock;
  ClosePivotBlock<<<1, kPivotBlock, 0, stream>>>(
      reachability, pivot_masks, words, block_start, block_size);
  BuildPivotRows<<<block_size, 128, 0, stream>>>(
      reachability, pivot_rows, pivot_masks, words, block_start, block_size);
  if (words <= 256) {
    ApplyPivotBlock<true><<<vertices, 128, 0, stream>>>(
        reachability, pivot_rows, pivot_masks, vertices, words, block_start,
        block_size);
  } else {
    ApplyPivotBlock<false><<<vertices, 128, 0, stream>>>(
        reachability, pivot_rows, pivot_masks, vertices, words, block_start,
        block_size);
  }
}
