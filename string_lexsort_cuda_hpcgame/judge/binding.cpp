#include <torch/extension.h>

#include <cstdint>


int64_t workspace_bytes(int64_t n, int64_t width);

void lexsort_cuda(
    torch::Tensor strings,
    torch::Tensor lengths,
    torch::Tensor indices_out,
    torch::Tensor workspace);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "workspace_bytes",
      &workspace_bytes,
      "Return the required CUDA workspace size in bytes");
  module.def(
      "lexsort_cuda",
      &lexsort_cuda,
      "Sort variable-length byte strings on the current CUDA stream");
}
