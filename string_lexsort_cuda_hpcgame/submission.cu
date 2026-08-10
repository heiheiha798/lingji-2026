#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>


namespace {

constexpr int kThreads = 256;


struct GraphEntry {
  const uint8_t* strings;
  int32_t* output;
  int32_t* workspace;
  int32_t n;
  int32_t width;
  cudaGraphExec_t executable;
};


template <int kWidth, int kStart = 0>
__device__ __forceinline__ bool index_less(
    int32_t lhs,
    int32_t rhs,
    const uint8_t* __restrict__ strings) {
  const uint8_t* lhs_string = strings + static_cast<int64_t>(lhs) * kWidth;
  const uint8_t* rhs_string = strings + static_cast<int64_t>(rhs) * kWidth;
#pragma unroll
  for (int32_t offset = kStart; offset < kWidth; offset += 4) {
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


template <int kWidth>
__device__ __forceinline__ bool shared_index_less(
    int32_t lhs_position,
    int32_t rhs_position,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ keys) {
#pragma unroll
  for (int32_t word = 0; word < kWidth / 4; ++word) {
    const uint32_t lhs_word = keys[word * kThreads + lhs_position];
    const uint32_t rhs_word = keys[word * kThreads + rhs_position];
    if (lhs_word != rhs_word) {
      return lhs_word < rhs_word;
    }
  }
  return indices[lhs_position] < indices[rhs_position];
}


template <int kWidth>
__device__ __forceinline__ bool shared_global_suffix_less(
    int32_t lhs,
    int32_t rhs,
    int32_t block_begin,
    const uint32_t* __restrict__ suffixes) {
  const int32_t lhs_position = lhs - block_begin;
  const int32_t rhs_position = rhs - block_begin;
#pragma unroll
  for (int32_t word = 1; word < kWidth / 4; ++word) {
    const uint32_t lhs_word =
        suffixes[(word - 1) * kThreads + lhs_position];
    const uint32_t rhs_word =
        suffixes[(word - 1) * kThreads + rhs_position];
    if (lhs_word != rhs_word) {
      return lhs_word < rhs_word;
    }
  }
  return lhs < rhs;
}


template <int kWidth, bool kCacheSuffix = false>
__global__ void sort_tiles(
    const uint8_t* __restrict__ strings,
    int32_t* __restrict__ indices,
    int32_t n) {
  __shared__ int32_t tile[kThreads];
  __shared__ uint32_t prefixes[kThreads];
  extern __shared__ uint32_t suffixes[];

  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  const int32_t block_begin = static_cast<int32_t>(blockIdx.x * kThreads);
  const int32_t index = block_begin + lane;
  tile[lane] = index < n ? index : INT32_MAX;
  if (index < n) {
    const uint32_t first_word = *reinterpret_cast<const uint32_t*>(
        strings + static_cast<int64_t>(index) * kWidth);
    prefixes[lane] = __byte_perm(first_word, 0, 0x0123);
    if constexpr (kCacheSuffix) {
#pragma unroll
      for (int32_t word = 1; word < kWidth / 4; ++word) {
        const uint32_t key_word = *reinterpret_cast<const uint32_t*>(
            strings + static_cast<int64_t>(index) * kWidth + word * 4);
        suffixes[(word - 1) * kThreads + lane] =
            __byte_perm(key_word, 0, 0x0123);
      }
    }
  } else {
    prefixes[lane] = UINT32_MAX;
  }
  __syncthreads();

  for (int32_t size = 2; size <= kThreads; size <<= 1) {
    for (int32_t stride = size >> 1; stride > 0; stride >>= 1) {
      const int32_t partner = lane ^ stride;
      if (partner > lane) {
        const int32_t lhs = tile[lane];
        const int32_t rhs = tile[partner];
        const uint32_t lhs_prefix = prefixes[lane];
        const uint32_t rhs_prefix = prefixes[partner];
        const bool ascending = (lane & size) == 0;
        bool exchange;
        if (ascending) {
          exchange = lhs == INT32_MAX
              ? rhs != INT32_MAX
              : rhs != INT32_MAX &&
                    (rhs_prefix < lhs_prefix ||
                     (rhs_prefix == lhs_prefix &&
                      (kCacheSuffix
                           ? shared_global_suffix_less<kWidth>(
                                 rhs, lhs, block_begin, suffixes)
                           : index_less<kWidth, 4>(
                                 rhs, lhs, strings))));
        } else {
          exchange = rhs == INT32_MAX
              ? lhs != INT32_MAX
              : lhs != INT32_MAX &&
                    (lhs_prefix < rhs_prefix ||
                     (lhs_prefix == rhs_prefix &&
                      (kCacheSuffix
                           ? shared_global_suffix_less<kWidth>(
                                 lhs, rhs, block_begin, suffixes)
                           : index_less<kWidth, 4>(
                                 lhs, rhs, strings))));
        }
        if (exchange) {
          tile[lane] = rhs;
          tile[partner] = lhs;
          prefixes[lane] = rhs_prefix;
          prefixes[partner] = lhs_prefix;
        }
      }
      __syncthreads();
    }
  }

  if (index < n) {
    indices[index] = tile[lane];
  }
}


template <
    int kWidth,
    bool kCachePrefix = false,
    bool kCacheFullKey = false>
__global__ void merge_pass(
    const uint8_t* __restrict__ strings,
    const int32_t* __restrict__ source,
    int32_t* __restrict__ destination,
    int32_t n,
    int32_t run_length) {
  __shared__ int32_t tile[kThreads];
  __shared__ uint32_t tile_prefixes[kCachePrefix ? kThreads : 1];
  __shared__ int32_t block_partitions[2];
  extern __shared__ uint32_t tile_keys[];
  static_assert(!(kCachePrefix && kCacheFullKey));

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
          index_less<kWidth>(
              source[pair_begin + left],
              source[right_begin + right - 1],
              strings)) {
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
  if (lane < output_count) {
    const int32_t index = lane < left_count
        ? source[pair_begin + left_begin + lane]
        : source[right_begin + right_offset + lane - left_count];
    tile[lane] = index;
    if constexpr (kCachePrefix) {
      const uint32_t first_word = *reinterpret_cast<const uint32_t*>(
          strings + static_cast<int64_t>(index) * kWidth);
      tile_prefixes[lane] = __byte_perm(first_word, 0, 0x0123);
    } else if constexpr (kCacheFullKey) {
      static_assert(kWidth == 32);
      const uint8_t* string =
          strings + static_cast<int64_t>(index) * kWidth;
      const uint4 first_half =
          *reinterpret_cast<const uint4*>(string);
      const uint4 second_half =
          *reinterpret_cast<const uint4*>(string + sizeof(uint4));
      tile_keys[0 * kThreads + lane] =
          __byte_perm(first_half.x, 0, 0x0123);
      tile_keys[1 * kThreads + lane] =
          __byte_perm(first_half.y, 0, 0x0123);
      tile_keys[2 * kThreads + lane] =
          __byte_perm(first_half.z, 0, 0x0123);
      tile_keys[3 * kThreads + lane] =
          __byte_perm(first_half.w, 0, 0x0123);
      tile_keys[4 * kThreads + lane] =
          __byte_perm(second_half.x, 0, 0x0123);
      tile_keys[5 * kThreads + lane] =
          __byte_perm(second_half.y, 0, 0x0123);
      tile_keys[6 * kThreads + lane] =
          __byte_perm(second_half.z, 0, 0x0123);
      tile_keys[7 * kThreads + lane] =
          __byte_perm(second_half.w, 0, 0x0123);
    }
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
    if (left < left_count && right > 0) {
      bool left_before_right;
      if constexpr (kCacheFullKey) {
        left_before_right = shared_index_less<kWidth>(
            left,
            left_count + right - 1,
            tile,
            tile_keys);
      } else if constexpr (kCachePrefix) {
        const uint32_t left_prefix = tile_prefixes[left];
        const uint32_t right_prefix = tile_prefixes[left_count + right - 1];
        left_before_right = left_prefix < right_prefix ||
            (left_prefix == right_prefix &&
             index_less<kWidth, 4>(
                 tile[left],
                 tile[left_count + right - 1],
                 strings));
      } else {
        left_before_right = index_less<kWidth>(
               tile[left],
               tile[left_count + right - 1],
               strings);
      }
      if (left_before_right) {
        low = left + 1;
      } else {
        high = left;
      }
    } else {
      high = left;
    }
  }

  const int32_t left = low;
  const int32_t right = lane - left;
  bool take_left = left < left_count && right >= right_count;
  if (left < left_count && right < right_count) {
    if constexpr (kCacheFullKey) {
      take_left = shared_index_less<kWidth>(
          left, left_count + right, tile, tile_keys);
    } else if constexpr (kCachePrefix) {
      const uint32_t left_prefix = tile_prefixes[left];
      const uint32_t right_prefix = tile_prefixes[left_count + right];
      take_left = left_prefix < right_prefix ||
          (left_prefix == right_prefix &&
           index_less<kWidth, 4>(
               tile[left], tile[left_count + right], strings));
    } else {
      take_left = index_less<kWidth>(
          tile[left], tile[left_count + right], strings);
    }
  }
  if (take_left) {
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
  static thread_local std::vector<GraphEntry> graph_entries;
  for (const GraphEntry& entry : graph_entries) {
    if (entry.strings == input_strings &&
        entry.output == output &&
        entry.workspace == scratch &&
        entry.n == n &&
        entry.width == width) {
      const cudaError_t replay_status =
          cudaGraphLaunch(entry.executable, stream);
      TORCH_CHECK(
          replay_status == cudaSuccess,
          "CUDA graph replay failed: ",
          cudaGetErrorString(replay_status));
      return;
    }
  }

  cudaGraph_t graph = nullptr;
  cudaStream_t capture_stream = nullptr;
  const cudaError_t stream_status = cudaStreamCreateWithFlags(
      &capture_stream, cudaStreamNonBlocking);
  TORCH_CHECK(
      stream_status == cudaSuccess,
      "CUDA graph capture stream creation failed: ",
      cudaGetErrorString(stream_status));
  const cudaError_t begin_status = cudaStreamBeginCapture(
      capture_stream, cudaStreamCaptureModeThreadLocal);
  TORCH_CHECK(
      begin_status == cudaSuccess,
      "CUDA graph capture failed to start: ",
      cudaGetErrorString(begin_status));

  if (width == 16) {
    sort_tiles<16><<<blocks, kThreads, 0, capture_stream>>>(
        input_strings, output, n);
  } else if (width == 32) {
    if (n == 65536) {
      sort_tiles<32, true><<<
          blocks,
          kThreads,
          (32 / 4 - 1) * kThreads * sizeof(uint32_t),
          capture_stream>>>(
          input_strings, output, n);
    } else {
      sort_tiles<32><<<blocks, kThreads, 0, capture_stream>>>(
          input_strings, output, n);
    }
  } else {
    TORCH_CHECK(width == 64, "unsupported string width: ", width);
    sort_tiles<64><<<blocks, kThreads, 0, capture_stream>>>(
        input_strings, output, n);
  }

  const int32_t* source = output;
  int32_t* destination = scratch;
  for (int32_t run_length = kThreads;
       run_length < n;
       run_length <<= 1) {
    if (width == 16) {
      merge_pass<16><<<blocks, kThreads, 0, capture_stream>>>(
          input_strings, source, destination, n, run_length);
    } else if (width == 32) {
      merge_pass<32, false, true><<<
          blocks,
          kThreads,
          (32 / 4) * kThreads * sizeof(uint32_t),
          capture_stream>>>(
          input_strings, source, destination, n, run_length);
    } else {
      merge_pass<64, true><<<blocks, kThreads, 0, capture_stream>>>(
          input_strings, source, destination, n, run_length);
    }
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
        capture_stream);
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
  const cudaError_t end_status = cudaStreamEndCapture(capture_stream, &graph);
  TORCH_CHECK(
      end_status == cudaSuccess,
      "CUDA graph capture failed to finish: ",
      cudaGetErrorString(end_status));
  cudaGraphExec_t executable = nullptr;
  const cudaError_t instantiate_status =
      cudaGraphInstantiate(&executable, graph, 0);
  const cudaError_t destroy_status = cudaGraphDestroy(graph);
  const cudaError_t stream_destroy_status = cudaStreamDestroy(capture_stream);
  TORCH_CHECK(
      instantiate_status == cudaSuccess,
      "CUDA graph instantiation failed: ",
      cudaGetErrorString(instantiate_status));
  TORCH_CHECK(
      destroy_status == cudaSuccess,
      "CUDA graph destruction failed: ",
      cudaGetErrorString(destroy_status));
  TORCH_CHECK(
      stream_destroy_status == cudaSuccess,
      "CUDA graph capture stream destruction failed: ",
      cudaGetErrorString(stream_destroy_status));
  graph_entries.push_back(
      {input_strings, output, scratch, n, width, executable});
  const cudaError_t first_launch_status = cudaGraphLaunch(executable, stream);
  TORCH_CHECK(
      first_launch_status == cudaSuccess,
      "CUDA graph launch failed: ",
      cudaGetErrorString(first_launch_status));
}
