#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "evaluator/validator.h"

int main() {
  using namespace icecarver;

  GuardedOutput guarded{};
  std::string error;
  if (!AllocateGuardedOutput(1, 1, &guarded, &error)) {
    std::cerr << "AllocateGuardedOutput failed: " << error << '\n';
    return 1;
  }

  const std::size_t expected_count_bytes =
      static_cast<std::size_t>(kMaxIsovalues) * sizeof(std::uint64_t);
  if (guarded.count_payload_bytes != expected_count_bytes) {
    std::cerr << "count payload does not implement the public eight-slot ABI\n";
    FreeGuardedOutput(&guarded);
    return 1;
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::cerr << "cudaStreamCreateWithFlags failed\n";
    FreeGuardedOutput(&guarded);
    return 1;
  }
  if (!ResetGuardedOutput(1, &guarded, stream, &error)) {
    std::cerr << "ResetGuardedOutput failed: " << error << '\n';
    cudaStreamDestroy(stream);
    FreeGuardedOutput(&guarded);
    return 1;
  }

  // Model a conforming one-isovalue solver that initializes every public
  // triangle_counts slot.  The write must remain entirely before the guard.
  if (cudaMemsetAsync(guarded.output.triangle_counts, 0,
                      expected_count_bytes, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "device write or synchronization failed\n";
    cudaStreamDestroy(stream);
    FreeGuardedOutput(&guarded);
    return 1;
  }

  std::vector<unsigned char> trailing_guard(kGuardBytes);
  const unsigned char* trailing_guard_device =
      guarded.count_allocation + kGuardBytes + guarded.count_payload_bytes;
  if (cudaMemcpy(trailing_guard.data(), trailing_guard_device, kGuardBytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::cerr << "copying the trailing guard failed\n";
    cudaStreamDestroy(stream);
    FreeGuardedOutput(&guarded);
    return 1;
  }
  const bool intact =
      std::all_of(trailing_guard.begin(), trailing_guard.end(),
                  [](unsigned char value) { return value == kGuardPattern; });

  cudaStreamDestroy(stream);
  FreeGuardedOutput(&guarded);
  if (!intact) {
    std::cerr << "eight-slot count initialization corrupted the guard\n";
    return 1;
  }
  return 0;
}
