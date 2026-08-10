#include "closure_api.h"

#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

constexpr int kDagBatchNeighborCapacity = 256;

__global__ void AnalyzeUpperTriangularDag(std::uint64_t *reach,
                                          std::uint64_t *descriptors,
                                          int vertices, int words) {
  __shared__ int warp_degrees[8];
  __shared__ int warp_minimum_targets[8];
  const int row = blockIdx.x;
  const int diagonal_word = row / 64;
  const int diagonal_bit = row & 63;
  const std::size_t base = static_cast<std::size_t>(row) * words;
  int degree = 0;
  int minimum_target = vertices;
  for (int word = diagonal_word + threadIdx.x; word < words;
       word += blockDim.x) {
    std::uint64_t remaining = reach[base + word];
    if (word == diagonal_word) {
      remaining = diagonal_bit == 63
                      ? 0
                      : remaining &
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
    minimum_target =
        min(minimum_target,
            __shfl_down_sync(0xffffffffU, minimum_target, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  if (lane == 0) {
    warp_degrees[warp] = degree;
    warp_minimum_targets[warp] = minimum_target;
  }
  __syncthreads();
  if (warp == 0) {
    degree = lane < 8 ? warp_degrees[lane] : 0;
    minimum_target =
        lane < 8 ? warp_minimum_targets[lane] : vertices;
    for (int offset = 16; offset > 0; offset /= 2) {
      degree += __shfl_down_sync(0xffffffffU, degree, offset);
      minimum_target =
          min(minimum_target,
              __shfl_down_sync(0xffffffffU, minimum_target, offset));
    }
    if (lane == 0) {
      descriptors[row] =
          (static_cast<std::uint64_t>(degree) << 32) |
          static_cast<std::uint32_t>(minimum_target);
      reach[base + diagonal_word] |= 1ULL << diagonal_bit;
      if ((vertices & 63) != 0)
        reach[base + words - 1] &= (1ULL << (vertices & 63)) - 1ULL;
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
  for (int row = 1; row < vertices && use_dag_closure; ++row) {
    const std::size_t base = static_cast<std::size_t>(row) * words_per_row;
    const int diagonal_word = row / 64;
    for (int word = 0; word < diagonal_word; ++word) {
      if (adjacency[base + word] != 0) {
        use_dag_closure = false;
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
  if (use_dag_closure) {
    std::vector<std::uint64_t> dag_descriptors(vertices);
    AnalyzeUpperTriangularDag<<<vertices, 256, 0, d.stream>>>(
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
      const int degree = static_cast<int>(descriptor >> 32);
      if (degree > maximum_degree) maximum_degree = degree;
    }
    if (maximum_degree <= kDagBatchNeighborCapacity) {
      int batch_end = vertices;
      while (batch_end > 0) {
        int batch_start = batch_end - 1;
        while (batch_start > 0 &&
               static_cast<std::uint32_t>(
                   dag_descriptors[batch_start - 1]) >= batch_end) {
          --batch_start;
        }
        dag_batch_starts.push_back(batch_start);
        batch_end = batch_start;
      }
      use_parallel_dag_batches = dag_batch_starts.size() <= 256;
    }
  } else {
    LaunchInitialize(d.reachability, vertices, words_per_row, d.stream);
    if (!CudaOk(cudaGetLastError())) {
      return 2;
    }
  }
  if (use_dag_closure) {
    if (use_parallel_dag_batches) {
      int batch_end = vertices;
      for (const int batch_start : dag_batch_starts) {
        CloseUpperTriangularDagBatch<<<batch_end - batch_start, 256, 0,
                                       d.stream>>>(
            d.reachability, batch_start, vertices, words_per_row);
        batch_end = batch_start;
      }
    } else {
      CloseUpperTriangularDag<<<1, 256, 0, d.stream>>>(
          d.reachability, reinterpret_cast<int *>(d.pivot_rows), vertices,
          words_per_row);
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
