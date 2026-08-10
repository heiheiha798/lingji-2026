#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cub/device/device_merge_sort.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>


namespace {

template <int kWidth>
struct IndexLess {
  const uint8_t* strings;

  __device__ __forceinline__ bool operator()(int32_t lhs, int32_t rhs) const {
    const uint8_t* lhs_string =
        strings + static_cast<int64_t>(lhs) * kWidth;
    const uint8_t* rhs_string =
        strings + static_cast<int64_t>(rhs) * kWidth;
#pragma unroll
    for (int32_t offset = 0; offset < kWidth; offset += 4) {
      const uint32_t lhs_word =
          *reinterpret_cast<const uint32_t*>(lhs_string + offset);
      const uint32_t rhs_word =
          *reinterpret_cast<const uint32_t*>(rhs_string + offset);
      if (lhs_word != rhs_word) {
        return __byte_perm(lhs_word, 0, 0x0123) <
            __byte_perm(rhs_word, 0, 0x0123);
      }
    }
    return lhs < rhs;
  }
};


template <int kWidth>
size_t query_workspace(int32_t n) {
  size_t bytes = 0;
  thrust::counting_iterator<int32_t> input(0);
  const cudaError_t status = cub::DeviceMergeSort::SortKeysCopy(
      nullptr,
      bytes,
      input,
      static_cast<int32_t*>(nullptr),
      n,
      IndexLess<kWidth>{nullptr});
  TORCH_CHECK(
      status == cudaSuccess,
      "CUB workspace query failed: ",
      cudaGetErrorString(status));
  return bytes;
}


template <int kWidth>
void run_cub_sort(
    const uint8_t* strings,
    int32_t* output,
    int32_t n,
    void* workspace,
    size_t workspace_bytes,
    cudaStream_t stream) {
  thrust::counting_iterator<int32_t> input(0);
  const cudaError_t status = cub::DeviceMergeSort::SortKeysCopy(
      workspace,
      workspace_bytes,
      input,
      output,
      n,
      IndexLess<kWidth>{strings},
      stream);
  TORCH_CHECK(
      status == cudaSuccess,
      "CUB DeviceMergeSort failed: ",
      cudaGetErrorString(status));
}

}  // namespace


int64_t workspace_bytes(int64_t n, int64_t width) {
  TORCH_CHECK(n >= 0 && n <= INT32_MAX, "N exceeds the supported range");
  if (width == 16) {
    return static_cast<int64_t>(query_workspace<16>(static_cast<int32_t>(n)));
  }
  if (width == 32) {
    return static_cast<int64_t>(query_workspace<32>(static_cast<int32_t>(n)));
  }
  TORCH_CHECK(width == 64, "unsupported string width: ", width);
  return static_cast<int64_t>(query_workspace<64>(static_cast<int32_t>(n)));
}


void lexsort_cuda(
    torch::Tensor strings,
    torch::Tensor lengths,
    torch::Tensor indices_out,
    torch::Tensor workspace) {
  const int64_t n64 = strings.size(0);
  const int64_t width64 = strings.size(1);
  TORCH_CHECK(n64 <= INT32_MAX, "N exceeds the supported range");
  (void)lengths;

  const int32_t n = static_cast<int32_t>(n64);
  const uint8_t* input_strings = strings.data_ptr<uint8_t>();
  int32_t* output = indices_out.data_ptr<int32_t>();
  void* scratch = workspace.data_ptr<uint8_t>();
  size_t scratch_bytes = static_cast<size_t>(workspace.numel());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (width64 == 16) {
    run_cub_sort<16>(
        input_strings, output, n, scratch, scratch_bytes, stream);
  } else if (width64 == 32) {
    run_cub_sort<32>(
        input_strings, output, n, scratch, scratch_bytes, stream);
  } else {
    TORCH_CHECK(width64 == 64, "unsupported string width: ", width64);
    run_cub_sort<64>(
        input_strings, output, n, scratch, scratch_bytes, stream);
  }
}
