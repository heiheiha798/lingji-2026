#include <torch/extension.h>

#include <cstdint>


int64_t workspace_bytes(int64_t n, int64_t width) {
  return 0;
}


void lexsort_cuda(
    torch::Tensor strings,
    torch::Tensor lengths,
    torch::Tensor indices_out,
    torch::Tensor workspace) {
  TORCH_CHECK(false, "lexsort_cuda is not implemented");
}
