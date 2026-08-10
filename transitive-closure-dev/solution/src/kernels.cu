#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kPivotBlock = 128;

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
  __shared__ std::uint64_t current_pivot[2];
  const int row = threadIdx.x;
  std::uint64_t row_mask0 = 0;
  std::uint64_t row_mask1 = 0;
  if (row < block_size) {
    const std::size_t base =
        static_cast<std::size_t>(block_start + row) * words;
    const int block_word = block_start / 64;
    row_mask0 = reach[base + block_word];
    if (block_size > 64) row_mask1 = reach[base + block_word + 1];
    if (block_size < 64)
      row_mask0 &= (1ULL << block_size) - 1ULL;
    if (block_size > 64 && block_size < kPivotBlock)
      row_mask1 &= (1ULL << (block_size - 64)) - 1ULL;
    if (row < 64) {
      row_mask0 |= 1ULL << row;
    } else {
      row_mask1 |= 1ULL << (row - 64);
    }
  }

  for (int pivot = 0; pivot < block_size; ++pivot) {
    if (row == pivot) {
      current_pivot[0] = row_mask0;
      current_pivot[1] = row_mask1;
    }
    __syncthreads();
    const bool has_pivot =
        pivot < 64 ? (row_mask0 & (1ULL << pivot)) != 0
                   : (row_mask1 & (1ULL << (pivot - 64))) != 0;
    if (row < block_size && has_pivot) {
      row_mask0 |= current_pivot[0];
      row_mask1 |= current_pivot[1];
    }
    __syncthreads();
  }
  const bool row_identity =
      row >= block_size ||
      (row < 64
           ? row_mask0 == (1ULL << row) && row_mask1 == 0
           : row_mask0 == 0 && row_mask1 == (1ULL << (row - 64)));
  const int block_identity = __syncthreads_and(row_identity);
  if (row < block_size) {
    pivot_masks[2 * row] = row_mask0;
    pivot_masks[2 * row + 1] = row_mask1;
  }
  if (row == 0) pivot_masks[2 * kPivotBlock] = block_identity;
}

__global__ void BuildPivotRows(const std::uint64_t *reach,
                               std::uint64_t *pivot_rows,
                               const std::uint64_t *pivot_masks, int words,
                               int block_start, int block_size) {
  const int row = blockIdx.x;
  if (row >= block_size) return;
  const std::uint64_t mask0 = pivot_masks[2 * row];
  const std::uint64_t mask1 = pivot_masks[2 * row + 1];
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    std::uint64_t output = 0;
    std::uint64_t remaining0 = mask0;
    while (remaining0 != 0) {
      const int source = __ffsll(static_cast<long long>(remaining0)) - 1;
      output |= reach[static_cast<std::size_t>(block_start + source) * words +
                      word];
      remaining0 &= remaining0 - 1;
    }
    std::uint64_t remaining1 = mask1;
    while (remaining1 != 0) {
      const int source =
          64 + __ffsll(static_cast<long long>(remaining1)) - 1;
      output |= reach[static_cast<std::size_t>(block_start + source) * words +
                      word];
      remaining1 &= remaining1 - 1;
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
  const int block_word = block_start / 64;
  std::uint64_t row_mask0 = reach[base + block_word];
  std::uint64_t row_mask1 =
      block_size > 64 ? reach[base + block_word + 1] : 0;
  if (block_size < 64)
    row_mask0 &= (1ULL << block_size) - 1ULL;
  if (block_size > 64 && block_size < kPivotBlock)
    row_mask1 &= (1ULL << (block_size - 64)) - 1ULL;
  if constexpr (SkipEmpty) {
    if ((row_mask0 | row_mask1) == 0) return;
  }
  std::uint64_t selected_mask0 = 0;
  std::uint64_t selected_mask1 = 0;
  if (pivot_masks[2 * kPivotBlock] != 0) {
    selected_mask0 = row_mask0;
    selected_mask1 = row_mask1;
  } else {
    if ((threadIdx.x & 31) == 0) {
      std::uint64_t remaining0 = row_mask0;
      std::uint64_t remaining1 = row_mask1;
      while ((remaining0 | remaining1) != 0) {
        int source;
        if (remaining0 != 0) {
          source = __ffsll(static_cast<long long>(remaining0)) - 1;
          selected_mask0 |= 1ULL << source;
        } else {
          source = 64 + __ffsll(static_cast<long long>(remaining1)) - 1;
          selected_mask1 |= 1ULL << (source - 64);
        }
        remaining0 &= ~pivot_masks[2 * source];
        remaining1 &= ~pivot_masks[2 * source + 1];
      }
    }
    selected_mask0 = __shfl_sync(0xffffffffU, selected_mask0, 0);
    selected_mask1 = __shfl_sync(0xffffffffU, selected_mask1, 0);
  }
  if ((selected_mask0 | selected_mask1) == 0) return;

  const bool single_generator =
      (selected_mask1 == 0 &&
       (selected_mask0 & (selected_mask0 - 1)) == 0) ||
      (selected_mask0 == 0 &&
       (selected_mask1 & (selected_mask1 - 1)) == 0);
  if (single_generator) {
    const int source =
        selected_mask0 != 0
            ? __ffsll(static_cast<long long>(selected_mask0)) - 1
            : 64 + __ffsll(static_cast<long long>(selected_mask1)) - 1;
    for (int word = threadIdx.x; word < words; word += blockDim.x) {
      const std::uint64_t input = reach[base + word];
      const std::uint64_t output =
          input |
          pivot_rows[static_cast<std::size_t>(source) * words + word];
      if (output != input) reach[base + word] = output;
    }
    return;
  }

  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    const std::uint64_t input = reach[base + word];
    std::uint64_t output = input;
    std::uint64_t remaining0 = selected_mask0;
    while (remaining0 != 0) {
      const int source = __ffsll(static_cast<long long>(remaining0)) - 1;
      output |= pivot_rows[static_cast<std::size_t>(source) * words + word];
      remaining0 &= remaining0 - 1;
    }
    std::uint64_t remaining1 = selected_mask1;
    while (remaining1 != 0) {
      const int source =
          64 + __ffsll(static_cast<long long>(remaining1)) - 1;
      output |= pivot_rows[static_cast<std::size_t>(source) * words + word];
      remaining1 &= remaining1 - 1;
    }
    if (output != input) reach[base + word] = output;
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
