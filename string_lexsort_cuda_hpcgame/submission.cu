#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>


namespace {

constexpr int kThreads = 256;


__device__ __forceinline__ bool index_less(
    int32_t lhs,
    int32_t rhs,
    const uint8_t* __restrict__ strings,
    int32_t width) {
  const uint8_t* lhs_string = strings + static_cast<int64_t>(lhs) * width;
  const uint8_t* rhs_string = strings + static_cast<int64_t>(rhs) * width;
  for (int32_t offset = 0; offset < width; offset += 4) {
    const uint32_t lhs_word =
        *reinterpret_cast<const uint32_t*>(lhs_string + offset);
    const uint32_t rhs_word =
        *reinterpret_cast<const uint32_t*>(rhs_string + offset);
    if (lhs_word != rhs_word) {
      const uint32_t lhs_big_endian = __byte_perm(lhs_word, 0, 0x0123);
      const uint32_t rhs_big_endian = __byte_perm(rhs_word, 0, 0x0123);
      return lhs_big_endian < rhs_big_endian;
    }
  }
  return lhs < rhs;
}


__global__ void sort_tiles(
    const uint8_t* __restrict__ strings,
    int32_t* __restrict__ indices,
    int32_t n,
    int32_t width) {
  __shared__ int32_t tile[kThreads];

  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t index = static_cast<int32_t>(blockIdx.x * kThreads) + lane;
  tile[lane] = index < n ? index : INT32_MAX;
  __syncthreads();

  for (int32_t size = 2; size <= kThreads; size <<= 1) {
    for (int32_t stride = size >> 1; stride > 0; stride >>= 1) {
      const int32_t partner = lane ^ stride;
      if (partner > lane) {
        const int32_t lhs = tile[lane];
        const int32_t rhs = tile[partner];
        const bool ascending = (lane & size) == 0;
        bool exchange;
        if (ascending) {
          exchange = lhs == INT32_MAX
              ? rhs != INT32_MAX
              : rhs != INT32_MAX && index_less(rhs, lhs, strings, width);
        } else {
          exchange = rhs == INT32_MAX
              ? lhs != INT32_MAX
              : lhs != INT32_MAX && index_less(lhs, rhs, strings, width);
        }
        if (exchange) {
          tile[lane] = rhs;
          tile[partner] = lhs;
        }
      }
      __syncthreads();
    }
  }

  if (index < n) {
    indices[index] = tile[lane];
  }
}


__global__ void merge_pass(
    const uint8_t* __restrict__ strings,
    const int32_t* __restrict__ source,
    int32_t* __restrict__ destination,
    int32_t n,
    int32_t width,
    int32_t run_length) {
  __shared__ int32_t tile[kThreads];
  __shared__ int32_t block_partitions[2];

  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t block_begin = static_cast<int32_t>(blockIdx.x * kThreads);
  const int32_t pair_length = run_length << 1;
  const int32_t pair_begin = (block_begin / pair_length) * pair_length;
  const int32_t left_length = min(run_length, n - pair_begin);
  const int32_t right_begin = pair_begin + left_length;
  const int32_t right_length = min(run_length, n - right_begin);
  const int32_t block_diagonal = block_begin - pair_begin;
  const int32_t output_count = min(kThreads, n - block_begin);

  if (lane < 2) {
    const int32_t diagonal = block_diagonal + lane * output_count;
    int32_t low = max(0, diagonal - right_length);
    int32_t high = min(diagonal, left_length);
    while (low < high) {
      const int32_t left = (low + high) >> 1;
      const int32_t right = diagonal - left;
      if (left < left_length && right > 0 &&
          index_less(
              source[pair_begin + left],
              source[right_begin + right - 1],
              strings,
              width)) {
        low = left + 1;
      } else {
        high = left;
      }
    }
    block_partitions[lane] = low;
  }
  __syncthreads();

  const int32_t left_begin = block_partitions[0];
  const int32_t left_count = block_partitions[1] - left_begin;
  const int32_t right_offset = block_diagonal - left_begin;
  const int32_t right_count = output_count - left_count;
  if (lane < left_count) {
    tile[lane] = source[pair_begin + left_begin + lane];
  } else if (lane < output_count) {
    tile[lane] = source[right_begin + right_offset + lane - left_count];
  }
  __syncthreads();

  if (lane >= output_count) {
    return;
  }

  int32_t low = max(0, lane - right_count);
  int32_t high = min(lane, left_count);
  while (low < high) {
    const int32_t left = (low + high) >> 1;
    const int32_t right = lane - left;
    if (left < left_count && right > 0 &&
        index_less(
            tile[left],
            tile[left_count + right - 1],
            strings,
            width)) {
      low = left + 1;
    } else {
      high = left;
    }
  }

  const int32_t left = low;
  const int32_t right = lane - left;
  if (left < left_count &&
      (right >= right_count ||
       index_less(
           tile[left],
           tile[left_count + right],
           strings,
           width))) {
    destination[block_begin + lane] = tile[left];
  } else {
    destination[block_begin + lane] = tile[left_count + right];
  }
}

}  // namespace


int64_t workspace_bytes(int64_t n, int64_t width) {
  (void)width;
  return n * static_cast<int64_t>(sizeof(int32_t));
}


void lexsort_cuda(
    torch::Tensor strings,
    torch::Tensor lengths,
    torch::Tensor indices_out,
    torch::Tensor workspace) {
  const int64_t n64 = strings.size(0);
  const int64_t width64 = strings.size(1);
  TORCH_CHECK(n64 <= INT32_MAX, "N exceeds the supported range");
  TORCH_CHECK(width64 <= INT32_MAX, "width exceeds the supported range");
  TORCH_CHECK(
      workspace.numel() >= n64 * static_cast<int64_t>(sizeof(int32_t)),
      "workspace is too small");
  (void)lengths;

  const int32_t n = static_cast<int32_t>(n64);
  const int32_t width = static_cast<int32_t>(width64);
  int32_t* output = indices_out.data_ptr<int32_t>();
  int32_t* scratch = reinterpret_cast<int32_t*>(workspace.data_ptr<uint8_t>());
  const uint8_t* input_strings = strings.data_ptr<uint8_t>();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int32_t blocks = (n + kThreads - 1) / kThreads;

  sort_tiles<<<blocks, kThreads, 0, stream>>>(input_strings, output, n, width);

  const int32_t* source = output;
  int32_t* destination = scratch;
  for (int32_t run_length = kThreads;
       run_length < n;
       run_length <<= 1) {
    merge_pass<<<blocks, kThreads, 0, stream>>>(
        input_strings,
        source,
        destination,
        n,
        width,
        run_length);
    const int32_t* completed = destination;
    destination = const_cast<int32_t*>(source);
    source = completed;
  }

  if (source != output) {
    const cudaError_t copy_status = cudaMemcpyAsync(
        output,
        source,
        static_cast<size_t>(n) * sizeof(int32_t),
        cudaMemcpyDeviceToDevice,
        stream);
    TORCH_CHECK(
        copy_status == cudaSuccess,
        "failed to copy final indices: ",
        cudaGetErrorString(copy_status));
  }
  const cudaError_t launch_status = cudaGetLastError();
  TORCH_CHECK(
      launch_status == cudaSuccess,
      "CUDA kernel launch failed: ",
      cudaGetErrorString(launch_status));
}
