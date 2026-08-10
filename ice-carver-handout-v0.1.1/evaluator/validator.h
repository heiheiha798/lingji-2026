#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "icecarver/api.h"

namespace icecarver {

constexpr std::size_t kGuardBytes = 4096;
constexpr unsigned char kGuardPattern = 0xa5;

struct GuardedOutput {
  Output output{};
  unsigned char* count_allocation = nullptr;
  std::size_t count_payload_bytes = 0;
  std::array<unsigned char*, kMaxIsovalues> triangle_allocations{};
  std::array<std::size_t, kMaxIsovalues> triangle_payload_bytes{};
};

struct ValidationReport {
  bool correct = false;
  bool counts_match = false;
  bool capacity_ok = false;
  bool finite = false;
  bool guards_intact = false;
  bool values_within_tolerance = false;
  std::uint64_t mismatched_values = 0;
  std::uint64_t nonfinite_values = 0;
  std::uint64_t guard_corruptions = 0;
  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;
  std::vector<std::uint64_t> candidate_counts;
  std::vector<std::uint64_t> reference_counts;
  std::string message;
};

bool AllocateGuardedOutput(int num_isovalues, std::uint64_t per_iso_capacity,
                           GuardedOutput* guarded, std::string* error);
bool ResetGuardedOutput(int num_isovalues, GuardedOutput* guarded,
                        cudaStream_t stream, std::string* error);
void FreeGuardedOutput(GuardedOutput* guarded);

ValidationReport ValidateOutput(const GuardedOutput& candidate,
                                const GuardedOutput& reference,
                                int num_isovalues, double abs_tolerance,
                                double rel_tolerance, cudaStream_t stream);

}  // namespace icecarver
