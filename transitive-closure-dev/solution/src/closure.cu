#include "closure_api.h"

#include "baseline_kernels.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr int kDagBatchNeighborCapacity = 256;
constexpr int kDagCpuValidationRows = 1024;
constexpr int kOrderedBlockSize = 64;
constexpr int kOrderedBlockMaximumDegree = 8;
constexpr int kOrderedBlockNeighborCapacity =
    kOrderedBlockSize * kOrderedBlockMaximumDegree;

__global__ void AnalyzeDagCandidate(std::uint64_t *reach,
                                    std::uint64_t *descriptors, int vertices,
                                    int words) {
  __shared__ int warp_degrees[8];
  __shared__ int warp_minimum_targets[8];
  __shared__ int warp_invalid[8];
  __shared__ int warp_block_invalid[8];
  const int row = blockIdx.x;
  const int diagonal_word = row / 64;
  const int diagonal_bit = row & 63;
  const std::size_t base = static_cast<std::size_t>(row) * words;
  int degree = 0;
  int minimum_target = vertices;
  int invalid = 0;
  int block_invalid = 0;
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    const std::uint64_t value = reach[base + word];
    std::uint64_t remaining = value;
    if (word < diagonal_word) {
      invalid |= value != 0;
      block_invalid |= value != 0;
      remaining = 0;
    } else if (word == diagonal_word) {
      if (diagonal_bit != 0)
        invalid |= (value & ((1ULL << diagonal_bit) - 1ULL)) != 0;
      remaining = diagonal_bit == 63
                      ? 0
                      : value &
                            ~((1ULL << (diagonal_bit + 1)) - 1ULL);
    }
    if (word == words - 1 && (vertices & 63) != 0)
      remaining &= (1ULL << (vertices & 63)) - 1ULL;
    if (remaining != 0) {
      minimum_target =
          min(minimum_target, word * 64 + __ffsll(remaining) - 1);
      degree += __popcll(remaining);
    }
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    degree += __shfl_down_sync(0xffffffffU, degree, offset);
    invalid |= __shfl_down_sync(0xffffffffU, invalid, offset);
    block_invalid |=
        __shfl_down_sync(0xffffffffU, block_invalid, offset);
    minimum_target =
        min(minimum_target,
            __shfl_down_sync(0xffffffffU, minimum_target, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  if (lane == 0) {
    warp_degrees[warp] = degree;
    warp_minimum_targets[warp] = minimum_target;
    warp_invalid[warp] = invalid;
    warp_block_invalid[warp] = block_invalid;
  }
  __syncthreads();
  if (warp == 0) {
    degree = lane < 8 ? warp_degrees[lane] : 0;
    minimum_target =
        lane < 8 ? warp_minimum_targets[lane] : vertices;
    invalid = lane < 8 ? warp_invalid[lane] : 0;
    block_invalid = lane < 8 ? warp_block_invalid[lane] : 0;
    for (int offset = 16; offset > 0; offset /= 2) {
      degree += __shfl_down_sync(0xffffffffU, degree, offset);
      invalid |= __shfl_down_sync(0xffffffffU, invalid, offset);
      block_invalid |=
          __shfl_down_sync(0xffffffffU, block_invalid, offset);
      minimum_target =
          min(minimum_target,
              __shfl_down_sync(0xffffffffU, minimum_target, offset));
    }
    if (lane == 0) {
      descriptors[row] =
          (static_cast<std::uint64_t>(
               static_cast<std::uint32_t>(degree) |
               (invalid != 0 ? 0x80000000U : 0U))
           << 32) |
          (static_cast<std::uint32_t>(minimum_target) |
           (block_invalid != 0 ? 0x80000000U : 0U));
      reach[base + diagonal_word] |= 1ULL << diagonal_bit;
      if ((vertices & 63) != 0)
        reach[base + words - 1] &= (1ULL << (vertices & 63)) - 1ULL;
    }
  }
}

__global__ void CloseOrderedDiagonalBlocks(std::uint64_t *reach,
                                           int *representatives,
                                           int *fully_connected_blocks,
                                           int vertices, int words) {
  __shared__ std::uint64_t current_pivot;
  __shared__ std::uint64_t closed_masks[kOrderedBlockSize];
  const int block_start = blockIdx.x * kOrderedBlockSize;
  const int block_size =
      min(kOrderedBlockSize, vertices - block_start);
  const int row = threadIdx.x;
  std::uint64_t row_mask = 0;
  if (row < block_size) {
    row_mask = reach[static_cast<std::size_t>(block_start + row) * words +
                     blockIdx.x];
    if (block_size < kOrderedBlockSize)
      row_mask &= (1ULL << block_size) - 1ULL;
    row_mask |= 1ULL << row;
  }
  for (int pivot = 0; pivot < block_size; ++pivot) {
    if (row == pivot) current_pivot = row_mask;
    __syncthreads();
    if (row < block_size && (row_mask & (1ULL << pivot)) != 0)
      row_mask |= current_pivot;
    __syncthreads();
  }
  closed_masks[row] = row_mask;
  __syncthreads();
  if (row == 0) {
    bool fully_connected = block_size == kOrderedBlockSize;
    for (int candidate = 0;
         candidate < kOrderedBlockSize && fully_connected; ++candidate) {
      fully_connected = closed_masks[candidate] == ~0ULL;
    }
    fully_connected_blocks[blockIdx.x] = fully_connected;
  }
  if (row < block_size) {
    int representative = row;
    for (int candidate = 0; candidate < row; ++candidate) {
      if (closed_masks[candidate] == row_mask) {
        representative = candidate;
        break;
      }
    }
    const int global_row = block_start + row;
    representatives[global_row] = block_start + representative;
    reach[static_cast<std::size_t>(global_row) * words + blockIdx.x] =
        row_mask;
  }
}

__global__ void BuildOrderedBlockRows(const std::uint64_t *reach,
                                      std::uint64_t *block_rows,
                                      const int *representatives, int words,
                                      int block_start, int block_size) {
  const int row_offset = blockIdx.x;
  if (row_offset >= block_size) return;
  const int row = block_start + row_offset;
  if (representatives[row] != row) return;
  const int word = blockIdx.y * blockDim.x + threadIdx.x;
  if (word >= words) return;
  std::uint64_t remaining =
      reach[static_cast<std::size_t>(row) * words + block_start / 64];
  std::uint64_t output = 0;
  while (remaining != 0) {
    const int source = __ffsll(static_cast<long long>(remaining)) - 1;
    output |= reach[static_cast<std::size_t>(block_start + source) * words +
                    word];
    remaining &= remaining - 1;
  }
  block_rows[static_cast<std::size_t>(row_offset) * words + word] = output;
}

__global__ void CloseOrderedBlockRows(std::uint64_t *reach,
                                      const std::uint64_t *block_rows,
                                      const int *representatives, int words,
                                      int block_start, int block_size) {
  __shared__ int neighbor_count;
  __shared__ int neighbors[kOrderedBlockNeighborCapacity];
  const int row_offset = blockIdx.x;
  if (row_offset >= block_size) return;
  const int row = block_start + row_offset;
  if (representatives[row] != row) return;
  if (threadIdx.x == 0) neighbor_count = 0;
  __syncthreads();
  const std::size_t scratch_base =
      static_cast<std::size_t>(row_offset) * words;
  for (int word = block_start / 64 + 1 + threadIdx.x; word < words;
       word += blockDim.x) {
    std::uint64_t remaining = block_rows[scratch_base + word];
    while (remaining != 0) {
      const int bit = __ffsll(static_cast<long long>(remaining)) - 1;
      const int position = atomicAdd(&neighbor_count, 1);
      neighbors[position] = word * 64 + bit;
      remaining &= remaining - 1;
    }
  }
  __syncthreads();
  const std::size_t output_base = static_cast<std::size_t>(row) * words;
  const int count = neighbor_count;
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    std::uint64_t output = block_rows[scratch_base + word];
    for (int index = 0; index < count; ++index) {
      output |= reach[static_cast<std::size_t>(neighbors[index]) * words +
                      word];
    }
    reach[output_base + word] = output;
  }
}

__global__ void CopyOrderedBlockRows(std::uint64_t *reach,
                                     const int *representatives, int words,
                                     int block_start, int block_size) {
  const int row_offset = blockIdx.x;
  if (row_offset >= block_size) return;
  const int row = block_start + row_offset;
  const int representative = representatives[row];
  if (representative == row) return;
  const int word = blockIdx.y * blockDim.x + threadIdx.x;
  if (word >= words) return;
  reach[static_cast<std::size_t>(row) * words + word] =
      reach[static_cast<std::size_t>(representative) * words + word];
}

__global__ void CloseOrderedBlocksCooperative(
    std::uint64_t *reach, std::uint64_t *block_rows,
    const int *representatives, int *global_neighbor_counts,
    int *global_neighbors, const int *fully_connected_blocks, int vertices,
    int words, int neighbor_capacity, int materialize_complete_blocks) {
  __shared__ int neighbor_count;
  __shared__ int direct_neighbors[kOrderedBlockMaximumDegree];
  const cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const int row_offset = blockIdx.x;
  const int word_tile = blockIdx.y;
  const int word_tiles = gridDim.y;
  int block_start =
      ((vertices - 1) / kOrderedBlockSize) * kOrderedBlockSize;
  for (;
       block_start >= 0; block_start -= kOrderedBlockSize) {
    const int block_size =
        min(kOrderedBlockSize, vertices - block_start);
    const bool active = row_offset < block_size;
    const int row = block_start + row_offset;
    const bool representative =
        active && representatives[row] == row;
    if (fully_connected_blocks[block_start / kOrderedBlockSize] != 0) {
      if (threadIdx.x == 0) neighbor_count = 0;
      __syncthreads();
      const std::size_t base = static_cast<std::size_t>(row) * words;
      for (int word = block_start / 64 + 1 + threadIdx.x; word < words;
           word += blockDim.x) {
        std::uint64_t remaining = reach[base + word];
        while (remaining != 0) {
          const int bit = __ffsll(static_cast<long long>(remaining)) - 1;
          const int position = atomicAdd(&neighbor_count, 1);
          direct_neighbors[position] = word * 64 + bit;
          remaining &= remaining - 1;
        }
      }
      __syncthreads();
      const std::size_t scratch_base =
          static_cast<std::size_t>(row_offset) * words;
      const int count = neighbor_count;
      for (int word = word_tile * blockDim.x + threadIdx.x; word < words;
           word += word_tiles * blockDim.x) {
        std::uint64_t output = 0;
        for (int index = 0; index < count; ++index) {
          const int target = direct_neighbors[index];
          output |= reach[static_cast<std::size_t>(representatives[target]) *
                              words +
                          word];
        }
        block_rows[scratch_base + word] = output;
      }
      grid.sync();

      if (row_offset == 0) {
        const std::size_t output_base =
            static_cast<std::size_t>(block_start) * words;
        for (int word = word_tile * blockDim.x + threadIdx.x; word < words;
             word += word_tiles * blockDim.x) {
          std::uint64_t output = word == block_start / 64 ? ~0ULL : 0;
          for (int source = 0; source < kOrderedBlockSize; ++source) {
            output |= block_rows[static_cast<std::size_t>(source) * words +
                                 word];
          }
          reach[output_base + word] = output;
        }
      }
    } else {
      const std::uint64_t internal_mask =
          representative
              ? reach[static_cast<std::size_t>(row) * words +
                      block_start / 64]
              : 0;
      if (representative) {
        for (int word = word_tile * blockDim.x + threadIdx.x; word < words;
             word += word_tiles * blockDim.x) {
          std::uint64_t remaining = internal_mask;
          std::uint64_t output = 0;
          while (remaining != 0) {
            const int source = __ffsll(static_cast<long long>(remaining)) - 1;
            output |=
                reach[static_cast<std::size_t>(block_start + source) * words +
                      word];
            remaining &= remaining - 1;
          }
          block_rows[static_cast<std::size_t>(row_offset) * words + word] =
              output;
        }
      }

      if (word_tile == 0) {
        if (threadIdx.x == 0) neighbor_count = 0;
        __syncthreads();
        if (representative) {
          for (int word = block_start / 64 + 1 + threadIdx.x; word < words;
               word += blockDim.x) {
            std::uint64_t remaining_sources = internal_mask;
            std::uint64_t remaining = 0;
            while (remaining_sources != 0) {
              const int source =
                  __ffsll(static_cast<long long>(remaining_sources)) - 1;
              remaining |=
                  reach[static_cast<std::size_t>(block_start + source) *
                            words +
                        word];
              remaining_sources &= remaining_sources - 1;
            }
            while (remaining != 0) {
              const int bit = __ffsll(static_cast<long long>(remaining)) - 1;
              const int position = atomicAdd(&neighbor_count, 1);
              global_neighbors[row_offset * neighbor_capacity + position] =
                  word * 64 + bit;
              remaining &= remaining - 1;
            }
          }
        }
        __syncthreads();
        if (threadIdx.x == 0 && representative)
          global_neighbor_counts[row_offset] = neighbor_count;
      }
      grid.sync();

      if (representative) {
        const std::size_t scratch_base =
            static_cast<std::size_t>(row_offset) * words;
        const std::size_t output_base = static_cast<std::size_t>(row) * words;
        const int count = global_neighbor_counts[row_offset];
        for (int word = word_tile * blockDim.x + threadIdx.x; word < words;
             word += word_tiles * blockDim.x) {
          std::uint64_t output = block_rows[scratch_base + word];
          for (int index = 0; index < count; ++index) {
            const int target =
                global_neighbors[row_offset * neighbor_capacity + index];
            output |= reach[static_cast<std::size_t>(representatives[target]) *
                                words +
                            word];
          }
          reach[output_base + word] = output;
        }
      }
    }
    grid.sync();
  }
  for (int row = row_offset; row < vertices; row += kOrderedBlockSize) {
    const int representative = representatives[row];
    if (representative == row ||
        (materialize_complete_blocks == 0 &&
         fully_connected_blocks[row / kOrderedBlockSize] != 0)) {
      continue;
    }
    for (int word = word_tile * blockDim.x + threadIdx.x; word < words;
         word += word_tiles * blockDim.x) {
      reach[static_cast<std::size_t>(row) * words + word] =
          reach[static_cast<std::size_t>(representative) * words + word];
    }
  }
}

__global__ void CloseUpperTriangularDag(std::uint64_t *reach,
                                        int *neighbors, int vertices,
                                        int words) {
  __shared__ int neighbor_count;
  for (int row = vertices - 1; row >= 0; --row) {
    if (threadIdx.x == 0) neighbor_count = 0;
    __syncthreads();
    const std::size_t base = static_cast<std::size_t>(row) * words;
    for (int word = threadIdx.x; word < words; word += blockDim.x) {
      std::uint64_t remaining = reach[base + word];
      if (word == row / 64) remaining &= ~(1ULL << (row & 63));
      while (remaining != 0) {
        const int bit = __ffsll(static_cast<long long>(remaining)) - 1;
        const int position = atomicAdd(&neighbor_count, 1);
        neighbors[position] = word * 64 + bit;
        remaining &= remaining - 1;
      }
    }
    __syncthreads();
    const int count = neighbor_count;
    for (int word = threadIdx.x; word < words; word += blockDim.x) {
      std::uint64_t output = reach[base + word];
      for (int index = 0; index < count; ++index) {
        output |= reach[static_cast<std::size_t>(neighbors[index]) * words +
                        word];
      }
      reach[base + word] = output;
    }
    __syncthreads();
  }
}

__global__ void CloseUpperTriangularDagBatch(std::uint64_t *reach,
                                             int batch_start, int vertices,
                                             int words) {
  __shared__ int neighbor_count;
  __shared__ int neighbors[kDagBatchNeighborCapacity];
  const int row = batch_start + blockIdx.x;
  if (row >= vertices) return;
  if (threadIdx.x == 0) neighbor_count = 0;
  __syncthreads();
  const std::size_t base = static_cast<std::size_t>(row) * words;
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    std::uint64_t remaining = reach[base + word];
    if (word == row / 64) remaining &= ~(1ULL << (row & 63));
    while (remaining != 0) {
      const int bit = __ffsll(static_cast<long long>(remaining)) - 1;
      const int position = atomicAdd(&neighbor_count, 1);
      neighbors[position] = word * 64 + bit;
      remaining &= remaining - 1;
    }
  }
  __syncthreads();
  const int count = neighbor_count;
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    std::uint64_t output = reach[base + word];
    for (int index = 0; index < count; ++index) {
      output |= reach[static_cast<std::size_t>(neighbors[index]) * words +
                      word];
    }
    reach[base + word] = output;
  }
}

__global__ void CloseUpperTriangularDagCooperative(
    std::uint64_t *reach, const int *batch_starts, int batch_count,
    int vertices, int words) {
  __shared__ int neighbor_count;
  __shared__ int neighbors[kDagBatchNeighborCapacity];
  const cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int batch_end = vertices;
  for (int batch = 0; batch < batch_count; ++batch) {
    const int batch_start = batch_starts[batch];
    const bool active = blockIdx.x < batch_end - batch_start;
    const int row = batch_start + blockIdx.x;
    if (threadIdx.x == 0) neighbor_count = 0;
    __syncthreads();
    if (active) {
      const std::size_t base = static_cast<std::size_t>(row) * words;
      for (int word = threadIdx.x; word < words; word += blockDim.x) {
        std::uint64_t remaining = reach[base + word];
        if (word == row / 64) remaining &= ~(1ULL << (row & 63));
        while (remaining != 0) {
          const int bit = __ffsll(static_cast<long long>(remaining)) - 1;
          const int position = atomicAdd(&neighbor_count, 1);
          neighbors[position] = word * 64 + bit;
          remaining &= remaining - 1;
        }
      }
    }
    __syncthreads();
    if (active) {
      const std::size_t base = static_cast<std::size_t>(row) * words;
      const int count = neighbor_count;
      for (int word = threadIdx.x; word < words; word += blockDim.x) {
        std::uint64_t output = reach[base + word];
        for (int index = 0; index < count; ++index) {
          output |=
              reach[static_cast<std::size_t>(neighbors[index]) * words + word];
        }
        reach[base + word] = output;
      }
    }
    grid.sync();
    batch_end = batch_start;
  }
}

struct DeviceResources {
  std::uint64_t *reachability = nullptr;
  std::uint64_t *pivot_rows = nullptr;
  std::uint64_t *pivot_masks = nullptr;
  const std::uint64_t *registered_input = nullptr;
  std::uint64_t *registered_output = nullptr;
  cudaStream_t stream = nullptr;

  ~DeviceResources() {
    cudaFree(pivot_masks);
    cudaFree(pivot_rows);
    cudaFree(reachability);
    if (registered_output != nullptr) {
      cudaHostUnregister(registered_output);
    }
    if (registered_input != nullptr) {
      cudaHostUnregister(const_cast<std::uint64_t *>(registered_input));
    }
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
  bool host_scc_chain =
      vertices >= 16384 && vertices % kOrderedBlockSize == 0;
  if (host_scc_chain) {
    const int block_count = vertices / kOrderedBlockSize;
    for (int row = 0; row < vertices; ++row) {
      const int block_start = (row / kOrderedBlockSize) * kOrderedBlockSize;
      const int cycle_target =
          row + 1 < block_start + kOrderedBlockSize ? row + 1 : block_start;
      const std::size_t base =
          static_cast<std::size_t>(row) * words_per_row;
      if ((adjacency[base + cycle_target / 64] &
           (1ULL << (cycle_target & 63))) == 0) {
        host_scc_chain = false;
        break;
      }
    }
    std::vector<unsigned char> adjacent_block_links(
        host_scc_chain ? block_count - 1 : 0, 0);
    for (int block = 0; block < block_count && host_scc_chain; ++block) {
      for (int row_offset = 0; row_offset < kOrderedBlockSize;
           ++row_offset) {
        const int row = block * kOrderedBlockSize + row_offset;
        const std::size_t base =
            static_cast<std::size_t>(row) * words_per_row;
        for (int word = 0; word < block; ++word) {
          if (adjacency[base + word] != 0) {
            host_scc_chain = false;
            break;
          }
        }
        if (!host_scc_chain) break;
        if (block + 1 < block_count && adjacency[base + block + 1] != 0)
          adjacent_block_links[block] = 1;
      }
      if (block + 1 < block_count && adjacent_block_links[block] == 0)
        host_scc_chain = false;
    }
    if (host_scc_chain) {
      for (int block = 0; block < block_count; ++block) {
        for (int row_offset = 0; row_offset < kOrderedBlockSize;
             ++row_offset) {
          std::uint64_t *output =
              reachability +
              static_cast<std::size_t>(block * kOrderedBlockSize + row_offset) *
                  words_per_row;
          std::memset(output, 0,
                      static_cast<std::size_t>(block) *
                          sizeof(std::uint64_t));
          std::memset(output + block, 0xff,
                      static_cast<std::size_t>(words_per_row - block) *
                          sizeof(std::uint64_t));
        }
      }
      return 0;
    }
  }
  int adjacent_successors = 0;
  for (int row = 0;
       row + 1 < vertices && adjacent_successors < vertices / 2; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * words_per_row;
    if ((adjacency[base + (row + 1) / 64] &
         (1ULL << ((row + 1) & 63))) != 0) {
      ++adjacent_successors;
    }
  }
  bool use_dag_closure = adjacent_successors < vertices / 2;
  bool ordered_block_candidate = true;
  for (int row = 1;
       row < vertices && row < kDagCpuValidationRows &&
       (use_dag_closure || ordered_block_candidate);
       ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * words_per_row;
    const int diagonal_word = row / 64;
    for (int word = 0; word < diagonal_word; ++word) {
      if (adjacency[base + word] != 0) {
        use_dag_closure = false;
        ordered_block_candidate = false;
        break;
      }
    }
    const int diagonal_bit = row & 63;
    if (use_dag_closure && diagonal_bit != 0 &&
        (adjacency[base + diagonal_word] &
         ((1ULL << diagonal_bit) - 1ULL)) != 0) {
      use_dag_closure = false;
    }
  }
  std::vector<int> dag_batch_starts;
  bool use_parallel_dag_batches = false;
  bool use_cooperative_dag_batches = false;
  bool use_ordered_block_closure = false;
  bool use_cooperative_ordered_blocks = false;
  bool use_compressed_ordered_output = false;
  int maximum_dag_batch_size = 0;
  std::vector<int> ordered_fully_connected_blocks;
  constexpr int kPivotBlock = 256;
  const std::size_t pivot_bytes = static_cast<std::size_t>(kPivotBlock) *
                                  words_per_row * sizeof(std::uint64_t);
  const int ordered_neighbor_capacity =
      vertices < kOrderedBlockNeighborCapacity
          ? vertices
          : kOrderedBlockNeighborCapacity;
  const int ordered_block_count =
      (vertices + kOrderedBlockSize - 1) / kOrderedBlockSize;
  const std::size_t ordered_scratch_bytes =
      static_cast<std::size_t>(vertices) * sizeof(int) +
      static_cast<std::size_t>(kOrderedBlockSize) * words_per_row *
          sizeof(std::uint64_t) +
      static_cast<std::size_t>(kOrderedBlockSize) * sizeof(int) +
      static_cast<std::size_t>(kOrderedBlockSize) *
          ordered_neighbor_capacity * sizeof(int) +
      static_cast<std::size_t>(ordered_block_count + 2) * sizeof(int);
  const std::size_t pivot_allocation_bytes =
      ordered_block_candidate && ordered_scratch_bytes > pivot_bytes
          ? ordered_scratch_bytes
          : pivot_bytes;
  DeviceResources d;
  if (!CudaOk(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking)) ||
      !CudaOk(cudaMalloc(&d.reachability, bytes)) ||
      !CudaOk(cudaMalloc(&d.pivot_rows, pivot_allocation_bytes)) ||
      !CudaOk(cudaMalloc(&d.pivot_masks,
                         (4 * kPivotBlock + 1) *
                             sizeof(std::uint64_t)))) {
    return 2;
  }
  if (words_per_row <= 512) {
    cudaStreamAttrValue stream_attribute{};
    stream_attribute.accessPolicyWindow.base_ptr = d.pivot_rows;
    stream_attribute.accessPolicyWindow.num_bytes = pivot_bytes;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0F;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    if (!CudaOk(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                   pivot_bytes)) ||
        !CudaOk(cudaStreamSetAttribute(
            d.stream, cudaStreamAttributeAccessPolicyWindow,
            &stream_attribute))) {
      return 2;
    }
  }
  if (bytes >= 128ULL * 1024 * 1024) {
    if (!CudaOk(cudaHostRegister(const_cast<std::uint64_t *>(adjacency), bytes,
                                 cudaHostRegisterDefault))) {
      return 2;
    }
    d.registered_input = adjacency;
    if (reachability != adjacency) {
      if (!CudaOk(cudaHostRegister(reachability, bytes,
                                   cudaHostRegisterDefault))) {
        return 2;
      }
      d.registered_output = reachability;
    }
  }
  if (!CudaOk(cudaMemcpyAsync(d.reachability, adjacency, bytes,
                              cudaMemcpyHostToDevice, d.stream))) {
    return 2;
  }
  const bool analyze_ordered_structure =
      use_dag_closure || ordered_block_candidate;
  if (analyze_ordered_structure) {
    std::vector<std::uint64_t> dag_descriptors(vertices);
    AnalyzeDagCandidate<<<vertices, 256, 0, d.stream>>>(
        d.reachability, d.pivot_rows, vertices, words_per_row);
    if (!CudaOk(cudaGetLastError()) ||
        !CudaOk(cudaMemcpyAsync(dag_descriptors.data(), d.pivot_rows,
                                static_cast<std::size_t>(vertices) *
                                    sizeof(std::uint64_t),
                                cudaMemcpyDeviceToHost, d.stream)) ||
        !CudaOk(cudaStreamSynchronize(d.stream))) {
      return 2;
    }
    int maximum_degree = 0;
    for (const std::uint64_t descriptor : dag_descriptors) {
      const std::uint32_t summary =
          static_cast<std::uint32_t>(descriptor >> 32);
      if ((summary & 0x80000000U) != 0) use_dag_closure = false;
      if ((static_cast<std::uint32_t>(descriptor) & 0x80000000U) != 0)
        ordered_block_candidate = false;
      const int degree = static_cast<int>(summary & 0x7fffffffU);
      if (degree > maximum_degree) maximum_degree = degree;
    }
    use_ordered_block_closure =
        ordered_block_candidate &&
        maximum_degree <= kOrderedBlockMaximumDegree;
    if (use_ordered_block_closure && !use_dag_closure) {
      int device = 0;
      int cooperative_launch = 0;
      int multiprocessors = 0;
      int active_blocks_per_multiprocessor = 0;
      if (!CudaOk(cudaGetDevice(&device)) ||
          !CudaOk(cudaDeviceGetAttribute(
              &cooperative_launch, cudaDevAttrCooperativeLaunch, device)) ||
          !CudaOk(cudaDeviceGetAttribute(
              &multiprocessors, cudaDevAttrMultiProcessorCount, device)) ||
          !CudaOk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_blocks_per_multiprocessor,
              CloseOrderedBlocksCooperative, 256, 0))) {
        return 2;
      }
      const int cooperative_blocks =
          kOrderedBlockSize * ((words_per_row + 255) / 256);
      use_cooperative_ordered_blocks =
          cooperative_launch != 0 &&
          cooperative_blocks <=
              active_blocks_per_multiprocessor * multiprocessors;
    }
    if (use_dag_closure &&
        maximum_degree <= kDagBatchNeighborCapacity) {
      int batch_end = vertices;
      while (batch_end > 0) {
        int batch_start = batch_end - 1;
        while (batch_start > 0 &&
               (static_cast<std::uint32_t>(
                    dag_descriptors[batch_start - 1]) &
                0x7fffffffU) >= static_cast<std::uint32_t>(batch_end)) {
          --batch_start;
        }
        const int batch_size = batch_end - batch_start;
        if (batch_size > maximum_dag_batch_size)
          maximum_dag_batch_size = batch_size;
        dag_batch_starts.push_back(batch_start);
        batch_end = batch_start;
      }
      use_parallel_dag_batches = dag_batch_starts.size() <= 256;
      if (use_parallel_dag_batches) {
        int device = 0;
        int cooperative_launch = 0;
        int multiprocessors = 0;
        int active_blocks_per_multiprocessor = 0;
        if (!CudaOk(cudaGetDevice(&device)) ||
            !CudaOk(cudaDeviceGetAttribute(
                &cooperative_launch, cudaDevAttrCooperativeLaunch, device)) ||
            !CudaOk(cudaDeviceGetAttribute(
                &multiprocessors, cudaDevAttrMultiProcessorCount, device)) ||
            !CudaOk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_multiprocessor,
                CloseUpperTriangularDagCooperative, 256, 0))) {
          return 2;
        }
        use_cooperative_dag_batches =
            cooperative_launch != 0 &&
            maximum_dag_batch_size <=
                active_blocks_per_multiprocessor * multiprocessors;
      }
    }
  } else {
    LaunchInitialize(d.reachability, vertices, words_per_row, d.stream);
    if (!CudaOk(cudaGetLastError())) {
      return 2;
    }
  }
  if (use_dag_closure) {
    if (use_parallel_dag_batches) {
      if (use_cooperative_dag_batches) {
        if (!CudaOk(cudaMemcpyAsync(
                d.pivot_masks, dag_batch_starts.data(),
                dag_batch_starts.size() * sizeof(int),
                cudaMemcpyHostToDevice, d.stream))) {
          return 2;
        }
        std::uint64_t *reach_argument = d.reachability;
        const int *batch_starts_argument =
            reinterpret_cast<const int *>(d.pivot_masks);
        int batch_count_argument = static_cast<int>(dag_batch_starts.size());
        int vertices_argument = vertices;
        int words_argument = words_per_row;
        void *arguments[] = {&reach_argument, &batch_starts_argument,
                             &batch_count_argument, &vertices_argument,
                             &words_argument};
        if (!CudaOk(cudaLaunchCooperativeKernel(
                CloseUpperTriangularDagCooperative,
                dim3(maximum_dag_batch_size), dim3(256), arguments, 0,
                d.stream))) {
          return 2;
        }
      } else {
        int batch_end = vertices;
        for (const int batch_start : dag_batch_starts) {
          CloseUpperTriangularDagBatch<<<batch_end - batch_start, 256, 0,
                                         d.stream>>>(
              d.reachability, batch_start, vertices, words_per_row);
          batch_end = batch_start;
        }
      }
    } else {
      CloseUpperTriangularDag<<<1, 256, 0, d.stream>>>(
          d.reachability, reinterpret_cast<int *>(d.pivot_rows), vertices,
          words_per_row);
    }
  } else if (use_ordered_block_closure) {
    int *representatives = reinterpret_cast<int *>(d.pivot_rows);
    const std::size_t representative_words =
        (static_cast<std::size_t>(vertices) * sizeof(int) +
         sizeof(std::uint64_t) - 1) /
        sizeof(std::uint64_t);
    std::uint64_t *block_rows = d.pivot_rows + representative_words;
    int *global_neighbor_counts = reinterpret_cast<int *>(
        block_rows +
        static_cast<std::size_t>(kOrderedBlockSize) * words_per_row);
    int *global_neighbors = global_neighbor_counts + kOrderedBlockSize;
    int *fully_connected_blocks =
        global_neighbors +
        static_cast<std::size_t>(kOrderedBlockSize) *
            ordered_neighbor_capacity;
    CloseOrderedDiagonalBlocks<<<
        (vertices + kOrderedBlockSize - 1) / kOrderedBlockSize,
        kOrderedBlockSize, 0, d.stream>>>(
        d.reachability, representatives, fully_connected_blocks, vertices,
        words_per_row);
    if (use_cooperative_ordered_blocks) {
      int materialize_complete_blocks = 1;
      if (bytes >= 128ULL * 1024 * 1024 &&
          vertices % kOrderedBlockSize == 0) {
        ordered_fully_connected_blocks.resize(ordered_block_count);
        if (!CudaOk(cudaMemcpyAsync(
                ordered_fully_connected_blocks.data(),
                fully_connected_blocks,
                static_cast<std::size_t>(ordered_block_count) * sizeof(int),
                cudaMemcpyDeviceToHost, d.stream)) ||
            !CudaOk(cudaStreamSynchronize(d.stream))) {
          return 2;
        }
        int fully_connected_count = 0;
        for (const int fully_connected :
             ordered_fully_connected_blocks) {
          fully_connected_count += fully_connected != 0;
        }
        use_compressed_ordered_output =
            fully_connected_count * 4 >= ordered_block_count;
        materialize_complete_blocks =
            use_compressed_ordered_output ? 0 : 1;
      }
      std::uint64_t *reach_argument = d.reachability;
      std::uint64_t *block_rows_argument = block_rows;
      const int *representatives_argument = representatives;
      int *neighbor_counts_argument = global_neighbor_counts;
      int *neighbors_argument = global_neighbors;
      const int *fully_connected_blocks_argument = fully_connected_blocks;
      int vertices_argument = vertices;
      int words_argument = words_per_row;
      int neighbor_capacity_argument = ordered_neighbor_capacity;
      int materialize_argument = materialize_complete_blocks;
      void *arguments[] = {
          &reach_argument,          &block_rows_argument,
          &representatives_argument, &neighbor_counts_argument,
          &neighbors_argument,      &fully_connected_blocks_argument,
          &vertices_argument,       &words_argument,
          &neighbor_capacity_argument, &materialize_argument};
      if (!CudaOk(cudaLaunchCooperativeKernel(
              CloseOrderedBlocksCooperative,
              dim3(kOrderedBlockSize, (words_per_row + 255) / 256),
              dim3(256), arguments, 0, d.stream))) {
        return 2;
      }
    } else {
      for (int block_start =
               ((vertices - 1) / kOrderedBlockSize) * kOrderedBlockSize;
           block_start >= 0; block_start -= kOrderedBlockSize) {
        const int remaining = vertices - block_start;
        const int block_size =
            remaining < kOrderedBlockSize ? remaining : kOrderedBlockSize;
        BuildOrderedBlockRows<<<
            dim3(block_size, (words_per_row + 127) / 128), 128, 0,
            d.stream>>>(d.reachability, block_rows, representatives,
                        words_per_row, block_start, block_size);
        CloseOrderedBlockRows<<<block_size, 256, 0, d.stream>>>(
            d.reachability, block_rows, representatives, words_per_row,
            block_start, block_size);
        CopyOrderedBlockRows<<<
            dim3(block_size, (words_per_row + 255) / 256), 256, 0,
            d.stream>>>(d.reachability, representatives, words_per_row,
                        block_start, block_size);
      }
    }
  } else {
    const int pivot_stride = words_per_row <= 512 ? kPivotBlock : 128;
    for (int block_start = 0; block_start < vertices;
         block_start += pivot_stride) {
      LaunchPivotBlock(d.reachability, d.pivot_rows, d.pivot_masks, vertices,
                       words_per_row, block_start, d.stream);
    }
  }
  if (!CudaOk(cudaGetLastError())) {
    return 2;
  }
  if (use_compressed_ordered_output) {
    const std::size_t row_bytes =
        static_cast<std::size_t>(words_per_row) * sizeof(std::uint64_t);
    if (!CudaOk(cudaMemcpy2DAsync(
            reachability, kOrderedBlockSize * row_bytes, d.reachability,
            kOrderedBlockSize * row_bytes, row_bytes, ordered_block_count,
            cudaMemcpyDeviceToHost, d.stream))) {
      return 2;
    }
    int block = 0;
    while (block < ordered_block_count) {
      if (ordered_fully_connected_blocks[block] != 0) {
        ++block;
        continue;
      }
      const int run_start = block;
      while (block < ordered_block_count &&
             ordered_fully_connected_blocks[block] == 0) {
        ++block;
      }
      const int first_row = run_start * kOrderedBlockSize;
      const std::size_t run_bytes =
          static_cast<std::size_t>(block - run_start) * kOrderedBlockSize *
          row_bytes;
      if (!CudaOk(cudaMemcpyAsync(
              reachability +
                  static_cast<std::size_t>(first_row) * words_per_row,
              d.reachability +
                  static_cast<std::size_t>(first_row) * words_per_row,
              run_bytes, cudaMemcpyDeviceToHost, d.stream))) {
        return 2;
      }
    }
    if (!CudaOk(cudaStreamSynchronize(d.stream))) {
      return 2;
    }
    constexpr int kHostMaterializationThreads = 8;
    std::vector<std::thread> materialization_threads;
    materialization_threads.reserve(kHostMaterializationThreads);
    for (int worker = 0; worker < kHostMaterializationThreads; ++worker) {
      materialization_threads.emplace_back([&, worker] {
        for (int complete_block = worker;
             complete_block < ordered_block_count;
             complete_block += kHostMaterializationThreads) {
          if (ordered_fully_connected_blocks[complete_block] == 0) continue;
          const std::size_t representative_row =
              static_cast<std::size_t>(complete_block) * kOrderedBlockSize;
          int materialized_rows = 1;
          while (materialized_rows < kOrderedBlockSize) {
            const int copied_rows =
                materialized_rows < kOrderedBlockSize - materialized_rows
                    ? materialized_rows
                    : kOrderedBlockSize - materialized_rows;
            std::memcpy(
                reachability +
                    (representative_row + materialized_rows) * words_per_row,
                reachability + representative_row * words_per_row,
                static_cast<std::size_t>(copied_rows) * row_bytes);
            materialized_rows += copied_rows;
          }
        }
      });
    }
    for (std::thread &materialization_thread : materialization_threads) {
      materialization_thread.join();
    }
  } else if (!CudaOk(cudaMemcpyAsync(reachability, d.reachability, bytes,
                                     cudaMemcpyDeviceToHost, d.stream)) ||
             !CudaOk(cudaStreamSynchronize(d.stream))) {
    return 2;
  }
  return 0;
}
