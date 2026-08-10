#include "closure_api.h"

#include "baseline_kernels.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void CloseUpperTriangularDag(std::uint64_t *reach,
                                        int *neighbors, int vertices,
                                        int words) {
  __shared__ int neighbor_count;
  const int tail_bits = vertices & 63;
  for (int row = vertices - 1; row >= 0; --row) {
    if (threadIdx.x == 0) neighbor_count = 0;
    __syncthreads();
    const std::size_t base = static_cast<std::size_t>(row) * words;
    for (int word = threadIdx.x; word < words; word += blockDim.x) {
      std::uint64_t remaining = reach[base + word];
      if (tail_bits != 0 && word == words - 1)
        remaining &= (1ULL << tail_bits) - 1ULL;
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
      if (word == row / 64) output |= 1ULL << (row & 63);
      for (int index = 0; index < count; ++index) {
        output |= reach[static_cast<std::size_t>(neighbors[index]) * words +
                        word];
      }
      if (tail_bits != 0 && word == words - 1)
        output &= (1ULL << tail_bits) - 1ULL;
      reach[base + word] = output;
    }
    __syncthreads();
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
  constexpr int kPivotBlock = 256;
  const std::size_t pivot_bytes =
      use_dag_closure
          ? static_cast<std::size_t>(vertices) * sizeof(int)
          : static_cast<std::size_t>(kPivotBlock) * words_per_row *
                sizeof(std::uint64_t);
  DeviceResources d;
  if (!CudaOk(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking)) ||
      !CudaOk(cudaMalloc(&d.reachability, bytes)) ||
      !CudaOk(cudaMalloc(&d.pivot_rows, pivot_bytes)) ||
      (!use_dag_closure &&
       !CudaOk(cudaMalloc(&d.pivot_masks,
                          (4 * kPivotBlock + 1) *
                              sizeof(std::uint64_t))))) {
    return 2;
  }
  if (!use_dag_closure && words_per_row <= 512) {
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
    CloseUpperTriangularDag<<<1, 256, 0, d.stream>>>(
        d.reachability, reinterpret_cast<int *>(d.pivot_rows), vertices,
        words_per_row);
  } else {
    LaunchInitialize(d.reachability, vertices, words_per_row, d.stream);
    if (!CudaOk(cudaGetLastError())) {
      return 2;
    }
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
