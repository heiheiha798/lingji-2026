#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kPivotBlock = 256;
constexpr int kPivotMaskWords = kPivotBlock / 64;
constexpr int kNarrowPivotBlock = 128;

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
  __shared__ std::uint64_t current_pivot[kPivotMaskWords];
  const int row = threadIdx.x;
  std::uint64_t row_mask0 = 0;
  std::uint64_t row_mask1 = 0;
  std::uint64_t row_mask2 = 0;
  std::uint64_t row_mask3 = 0;
  if (row < block_size) {
    const std::size_t base =
        static_cast<std::size_t>(block_start + row) * words;
    const int block_word = block_start / 64;
    row_mask0 = reach[base + block_word];
    if (block_size > 64) row_mask1 = reach[base + block_word + 1];
    if (block_size > 128) row_mask2 = reach[base + block_word + 2];
    if (block_size > 192) row_mask3 = reach[base + block_word + 3];
    const int tail_bits = block_size & 63;
    if (tail_bits != 0) {
      const std::uint64_t tail_mask = (1ULL << tail_bits) - 1ULL;
      if (block_size <= 64) {
        row_mask0 &= tail_mask;
      } else if (block_size <= 128) {
        row_mask1 &= tail_mask;
      } else if (block_size <= 192) {
        row_mask2 &= tail_mask;
      } else {
        row_mask3 &= tail_mask;
      }
    }
    if (row < 64) {
      row_mask0 |= 1ULL << row;
    } else if (row < 128) {
      row_mask1 |= 1ULL << (row - 64);
    } else if (row < 192) {
      row_mask2 |= 1ULL << (row - 128);
    } else {
      row_mask3 |= 1ULL << (row - 192);
    }
  }

  for (int pivot = 0; pivot < block_size; ++pivot) {
    if (row == pivot) {
      current_pivot[0] = row_mask0;
      current_pivot[1] = row_mask1;
      current_pivot[2] = row_mask2;
      current_pivot[3] = row_mask3;
    }
    __syncthreads();
    const bool has_pivot =
        pivot < 64 ? (row_mask0 & (1ULL << pivot)) != 0
        : pivot < 128 ? (row_mask1 & (1ULL << (pivot - 64))) != 0
        : pivot < 192 ? (row_mask2 & (1ULL << (pivot - 128))) != 0
                      : (row_mask3 & (1ULL << (pivot - 192))) != 0;
    if (row < block_size && has_pivot) {
      row_mask0 |= current_pivot[0];
      row_mask1 |= current_pivot[1];
      row_mask2 |= current_pivot[2];
      row_mask3 |= current_pivot[3];
    }
    __syncthreads();
  }
  const bool has_diagonal =
      row < 64
          ? (row_mask0 & (1ULL << row)) != 0
          : row < 128
                ? (row_mask1 & (1ULL << (row - 64))) != 0
                : row < 192
                      ? (row_mask2 & (1ULL << (row - 128))) != 0
                      : (row_mask3 & (1ULL << (row - 192))) != 0;
  const bool row_identity =
      row >= block_size ||
      (has_diagonal &&
       __popcll(static_cast<unsigned long long>(row_mask0)) +
               __popcll(static_cast<unsigned long long>(row_mask1)) +
               __popcll(static_cast<unsigned long long>(row_mask2)) +
               __popcll(static_cast<unsigned long long>(row_mask3)) ==
           1);
  const int block_identity = __syncthreads_and(row_identity);
  if (row < block_size) {
    pivot_masks[kPivotMaskWords * row] = row_mask0;
    pivot_masks[kPivotMaskWords * row + 1] = row_mask1;
    pivot_masks[kPivotMaskWords * row + 2] = row_mask2;
    pivot_masks[kPivotMaskWords * row + 3] = row_mask3;
  }
  if (row == 0)
    pivot_masks[kPivotMaskWords * kPivotBlock] = block_identity;
}

__global__ void ClosePivotBlock128(const std::uint64_t *reach,
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
    if (block_size > 64 && block_size < kNarrowPivotBlock)
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
    pivot_masks[kPivotMaskWords * row] = row_mask0;
    pivot_masks[kPivotMaskWords * row + 1] = row_mask1;
    pivot_masks[kPivotMaskWords * row + 2] = 0;
    pivot_masks[kPivotMaskWords * row + 3] = 0;
  }
  if (row == 0)
    pivot_masks[kPivotMaskWords * kPivotBlock] = block_identity;
}

__global__ void BuildPivotRows(const std::uint64_t *reach,
                               std::uint64_t *pivot_rows,
                               const std::uint64_t *pivot_masks, int words,
                               int block_start, int block_size) {
  const int row = blockIdx.x;
  if (row >= block_size) return;
  const int word = blockIdx.y * blockDim.x + threadIdx.x;
  if (word >= words) return;
  std::uint64_t output = 0;
#pragma unroll
  for (int mask_word = 0; mask_word < kPivotMaskWords; ++mask_word) {
    std::uint64_t remaining =
        pivot_masks[kPivotMaskWords * row + mask_word];
    while (remaining != 0) {
      const int source =
          mask_word * 64 + __ffsll(static_cast<long long>(remaining)) - 1;
      output |= reach[static_cast<std::size_t>(block_start + source) * words +
                      word];
      remaining &= remaining - 1;
    }
  }
  pivot_rows[static_cast<std::size_t>(row) * words + word] = output;
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
  std::uint64_t row_mask2 =
      block_size > 128 ? reach[base + block_word + 2] : 0;
  std::uint64_t row_mask3 =
      block_size > 192 ? reach[base + block_word + 3] : 0;
  const int tail_bits = block_size & 63;
  if (tail_bits != 0) {
    const std::uint64_t tail_mask = (1ULL << tail_bits) - 1ULL;
    if (block_size <= 64) {
      row_mask0 &= tail_mask;
    } else if (block_size <= 128) {
      row_mask1 &= tail_mask;
    } else if (block_size <= 192) {
      row_mask2 &= tail_mask;
    } else {
      row_mask3 &= tail_mask;
    }
  }
  if constexpr (SkipEmpty) {
    if ((row_mask0 | row_mask1 | row_mask2 | row_mask3) == 0) return;
  }
  std::uint64_t selected_mask0 = 0;
  std::uint64_t selected_mask1 = 0;
  std::uint64_t selected_mask2 = 0;
  std::uint64_t selected_mask3 = 0;
  if (pivot_masks[kPivotMaskWords * kPivotBlock] != 0) {
    selected_mask0 = row_mask0;
    selected_mask1 = row_mask1;
    selected_mask2 = row_mask2;
    selected_mask3 = row_mask3;
  } else {
    if ((threadIdx.x & 31) == 0) {
      std::uint64_t remaining0 = row_mask0;
      std::uint64_t remaining1 = row_mask1;
      std::uint64_t remaining2 = row_mask2;
      std::uint64_t remaining3 = row_mask3;
      while ((remaining0 | remaining1 | remaining2 | remaining3) != 0) {
        int source;
        if (remaining0 != 0) {
          source = __ffsll(static_cast<long long>(remaining0)) - 1;
          selected_mask0 |= 1ULL << source;
        } else if (remaining1 != 0) {
          source = 64 + __ffsll(static_cast<long long>(remaining1)) - 1;
          selected_mask1 |= 1ULL << (source - 64);
        } else if (remaining2 != 0) {
          source = 128 + __ffsll(static_cast<long long>(remaining2)) - 1;
          selected_mask2 |= 1ULL << (source - 128);
        } else {
          source = 192 + __ffsll(static_cast<long long>(remaining3)) - 1;
          selected_mask3 |= 1ULL << (source - 192);
        }
        remaining0 &= ~pivot_masks[kPivotMaskWords * source];
        remaining1 &= ~pivot_masks[kPivotMaskWords * source + 1];
        remaining2 &= ~pivot_masks[kPivotMaskWords * source + 2];
        remaining3 &= ~pivot_masks[kPivotMaskWords * source + 3];
      }
    }
    selected_mask0 = __shfl_sync(0xffffffffU, selected_mask0, 0);
    selected_mask1 = __shfl_sync(0xffffffffU, selected_mask1, 0);
    selected_mask2 = __shfl_sync(0xffffffffU, selected_mask2, 0);
    selected_mask3 = __shfl_sync(0xffffffffU, selected_mask3, 0);
  }
  if ((selected_mask0 | selected_mask1 | selected_mask2 | selected_mask3) ==
      0)
    return;

  const bool single_generator =
      __popcll(static_cast<unsigned long long>(selected_mask0)) +
          __popcll(static_cast<unsigned long long>(selected_mask1)) +
          __popcll(static_cast<unsigned long long>(selected_mask2)) +
          __popcll(static_cast<unsigned long long>(selected_mask3)) ==
      1;
  if (single_generator) {
    int source;
    if (selected_mask0 != 0) {
      source = __ffsll(static_cast<long long>(selected_mask0)) - 1;
    } else if (selected_mask1 != 0) {
      source = 64 + __ffsll(static_cast<long long>(selected_mask1)) - 1;
    } else if (selected_mask2 != 0) {
      source = 128 + __ffsll(static_cast<long long>(selected_mask2)) - 1;
    } else {
      source = 192 + __ffsll(static_cast<long long>(selected_mask3)) - 1;
    }
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
    std::uint64_t remaining2 = selected_mask2;
    while (remaining2 != 0) {
      const int source =
          128 + __ffsll(static_cast<long long>(remaining2)) - 1;
      output |= pivot_rows[static_cast<std::size_t>(source) * words + word];
      remaining2 &= remaining2 - 1;
    }
    std::uint64_t remaining3 = selected_mask3;
    while (remaining3 != 0) {
      const int source =
          192 + __ffsll(static_cast<long long>(remaining3)) - 1;
      output |= pivot_rows[static_cast<std::size_t>(source) * words + word];
      remaining3 &= remaining3 - 1;
    }
    if (output != input) reach[base + word] = output;
  }
}

__global__ void ApplyPivotBlock128(std::uint64_t *reach,
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
  if (block_size > 64 && block_size < kNarrowPivotBlock)
    row_mask1 &= (1ULL << (block_size - 64)) - 1ULL;

  std::uint64_t selected_mask0 = 0;
  std::uint64_t selected_mask1 = 0;
  if (pivot_masks[kPivotMaskWords * kPivotBlock] != 0) {
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
        remaining0 &= ~pivot_masks[kPivotMaskWords * source];
        remaining1 &= ~pivot_masks[kPivotMaskWords * source + 1];
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
  const int pivot_width =
      words <= 512 ? kPivotBlock : kNarrowPivotBlock;
  const int block_size = remaining < pivot_width ? remaining : pivot_width;
  if (words <= 512) {
    ClosePivotBlock<<<1, kPivotBlock, 0, stream>>>(
        reachability, pivot_masks, words, block_start, block_size);
  } else {
    ClosePivotBlock128<<<1, kNarrowPivotBlock, 0, stream>>>(
        reachability, pivot_masks, words, block_start, block_size);
  }
  BuildPivotRows<<<dim3(block_size, (words + 127) / 128), 128, 0, stream>>>(
      reachability, pivot_rows, pivot_masks, words, block_start, block_size);
  if (words <= 256) {
    ApplyPivotBlock<true><<<vertices, 128, 0, stream>>>(
        reachability, pivot_rows, pivot_masks, vertices, words, block_start,
        block_size);
  } else if (words <= 512) {
    ApplyPivotBlock<false><<<vertices, 128, 0, stream>>>(
        reachability, pivot_rows, pivot_masks, vertices, words, block_start,
        block_size);
  } else {
    ApplyPivotBlock128<<<vertices, 128, 0, stream>>>(
        reachability, pivot_rows, pivot_masks, vertices, words, block_start,
        block_size);
  }
}
