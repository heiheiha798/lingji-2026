#include "closure_api.h"

#include "baseline_kernels.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
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
  int maximum_dag_batch_size = 0;
  constexpr int kPivotBlock = 256;
  const std::size_t pivot_bytes = static_cast<std::size_t>(kPivotBlock) *
                                  words_per_row * sizeof(std::uint64_t);
  DeviceResources d;
  if (!CudaOk(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking)) ||
      !CudaOk(cudaMalloc(&d.reachability, bytes)) ||
      !CudaOk(cudaMalloc(&d.pivot_rows, pivot_bytes)) ||
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
    CloseOrderedDiagonalBlocks<<<
        (vertices + kOrderedBlockSize - 1) / kOrderedBlockSize,
        kOrderedBlockSize, 0, d.stream>>>(
        d.reachability, representatives, vertices, words_per_row);
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
  } else {
    const int pivot_stride = words_per_row <= 512 ? kPivotBlock : 128;
    for (int block_start = 0; block_start < vertices;
         block_start += pivot_stride) {
      LaunchPivotBlock(d.reachability, d.pivot_rows, d.pivot_masks, vertices,
                       words_per_row, block_start, d.stream);
    }
  }
  if (!CudaOk(cudaGetLastError()) ||
      !CudaOk(cudaMemcpyAsync(reachability, d.reachability, bytes,
                              cudaMemcpyDeviceToHost, d.stream)) ||
      !CudaOk(cudaStreamSynchronize(d.stream))) {
    return 2;
  }
  return 0;
}
