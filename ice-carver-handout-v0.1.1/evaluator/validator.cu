#include "validator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace icecarver {
namespace {

struct DeviceValidationStats {
  unsigned long long mismatched_values;
  unsigned long long nonfinite_values;
  unsigned long long guard_corruptions;
  unsigned int max_abs_error_bits;
  unsigned int max_rel_error_bits;
};

__device__ __forceinline__ void AtomicMaxNonnegativeFloat(
    unsigned int* destination, float value) {
  atomicMax(destination, __float_as_uint(value));
}

__global__ void CompareTrianglesKernel(const float* candidate,
                                       const float* reference,
                                       std::uint64_t value_count,
                                       float abs_tolerance,
                                       float rel_tolerance,
                                       DeviceValidationStats* stats) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= value_count) {
    return;
  }

  const float got = candidate[index];
  const float expected = reference[index];
  if (!isfinite(got) || !isfinite(expected)) {
    atomicAdd(&stats->mismatched_values, 1ull);
    return;
  }

  const float absolute_error = fabsf(got - expected);
  const float denominator = fmaxf(fabsf(expected), 1.0e-30f);
  const float relative_error = absolute_error / denominator;
  AtomicMaxNonnegativeFloat(&stats->max_abs_error_bits, absolute_error);
  AtomicMaxNonnegativeFloat(&stats->max_rel_error_bits, relative_error);
  if (!(absolute_error <= abs_tolerance || relative_error <= rel_tolerance)) {
    atomicAdd(&stats->mismatched_values, 1ull);
  }
}

__global__ void CheckFiniteKernel(const float* values,
                                  std::uint64_t value_count,
                                  DeviceValidationStats* stats) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < value_count && !isfinite(values[index])) {
    atomicAdd(&stats->nonfinite_values, 1ull);
  }
}

__global__ void CheckGuardKernel(const unsigned char* allocation,
                                 std::size_t payload_bytes,
                                 DeviceValidationStats* stats) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= 2 * kGuardBytes) {
    return;
  }
  const unsigned char observed =
      index < kGuardBytes
          ? allocation[index]
          : allocation[kGuardBytes + payload_bytes + (index - kGuardBytes)];
  if (observed != kGuardPattern) {
    atomicAdd(&stats->guard_corruptions, 1ull);
  }
}

bool SetCudaError(const char* operation, cudaError_t status,
                  std::string* error) {
  if (error != nullptr) {
    std::ostringstream stream;
    stream << operation << " failed: " << cudaGetErrorString(status);
    *error = stream.str();
  }
  return false;
}

bool CheckedAllocationSize(std::uint64_t capacity, std::size_t* payload,
                           std::string* error) {
  if (capacity >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                 sizeof(Triangle))) {
    if (error != nullptr) {
      *error = "triangle capacity overflows size_t";
    }
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(capacity) *
                            static_cast<std::size_t>(sizeof(Triangle));
  if (bytes > std::numeric_limits<std::size_t>::max() - 2 * kGuardBytes) {
    if (error != nullptr) {
      *error = "guarded allocation size overflows size_t";
    }
    return false;
  }
  *payload = bytes;
  return true;
}

void LaunchGuardCheck(const unsigned char* allocation,
                      std::size_t payload_bytes,
                      DeviceValidationStats* device_stats,
                      cudaStream_t stream) {
  constexpr int kThreads = 256;
  constexpr int kBlocks =
      static_cast<int>((2 * kGuardBytes + kThreads - 1) / kThreads);
  CheckGuardKernel<<<kBlocks, kThreads, 0, stream>>>(allocation, payload_bytes,
                                                     device_stats);
}

}  // namespace

bool AllocateGuardedOutput(int num_isovalues, std::uint64_t per_iso_capacity,
                           GuardedOutput* guarded, std::string* error) {
  if (guarded == nullptr || num_isovalues <= 0 ||
      num_isovalues > kMaxIsovalues || per_iso_capacity == 0) {
    if (error != nullptr) {
      *error = "invalid guarded-output allocation arguments";
    }
    return false;
  }

  *guarded = GuardedOutput{};
  // Output::triangle_counts is part of the public ABI and is documented as a
  // kMaxIsovalues-element device array.  Keep all eight slots inside the
  // guarded payload even when the current case uses fewer isovalues, so a
  // conforming solver may initialize the complete public array without
  // touching the trailing guard.
  guarded->count_payload_bytes =
      static_cast<std::size_t>(kMaxIsovalues) * sizeof(std::uint64_t);
  cudaError_t status = cudaMalloc(
      reinterpret_cast<void**>(&guarded->count_allocation),
      guarded->count_payload_bytes + 2 * kGuardBytes);
  if (status != cudaSuccess) {
    return SetCudaError("cudaMalloc(counts)", status, error);
  }
  guarded->output.triangle_counts = reinterpret_cast<std::uint64_t*>(
      guarded->count_allocation + kGuardBytes);

  for (int iso = 0; iso < num_isovalues; ++iso) {
    std::size_t payload = 0;
    if (!CheckedAllocationSize(per_iso_capacity, &payload, error)) {
      FreeGuardedOutput(guarded);
      return false;
    }
    status = cudaMalloc(
        reinterpret_cast<void**>(&guarded->triangle_allocations[iso]),
        payload + 2 * kGuardBytes);
    if (status != cudaSuccess) {
      SetCudaError("cudaMalloc(triangles)", status, error);
      FreeGuardedOutput(guarded);
      return false;
    }
    guarded->triangle_payload_bytes[iso] = payload;
    guarded->output.capacities[iso] = per_iso_capacity;
    guarded->output.triangles[iso] = reinterpret_cast<Triangle*>(
        guarded->triangle_allocations[iso] + kGuardBytes);
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

bool ResetGuardedOutput(int num_isovalues, GuardedOutput* guarded,
                        cudaStream_t stream, std::string* error) {
  if (guarded == nullptr || guarded->count_allocation == nullptr ||
      num_isovalues <= 0 || num_isovalues > kMaxIsovalues) {
    if (error != nullptr) {
      *error = "invalid guarded output passed to reset";
    }
    return false;
  }

  cudaError_t status = cudaMemsetAsync(
      guarded->count_allocation, kGuardPattern,
      guarded->count_payload_bytes + 2 * kGuardBytes, stream);
  if (status != cudaSuccess) {
    return SetCudaError("cudaMemsetAsync(count allocation)", status, error);
  }
  status = cudaMemsetAsync(guarded->output.triangle_counts, 0,
                           guarded->count_payload_bytes, stream);
  if (status != cudaSuccess) {
    return SetCudaError("cudaMemsetAsync(count payload)", status, error);
  }

  for (int iso = 0; iso < num_isovalues; ++iso) {
    unsigned char* allocation = guarded->triangle_allocations[iso];
    if (allocation == nullptr) {
      if (error != nullptr) {
        *error = "missing triangle allocation";
      }
      return false;
    }
    const std::size_t payload = guarded->triangle_payload_bytes[iso];
    status = cudaMemsetAsync(allocation, kGuardPattern,
                             payload + 2 * kGuardBytes, stream);
    if (status != cudaSuccess) {
      return SetCudaError("cudaMemsetAsync(triangle allocation)", status,
                          error);
    }
    // 0xff bytes form a NaN on IEEE-754 systems. If a solver reports a count
    // without writing every coordinate, the finite-value check catches it.
    status = cudaMemsetAsync(guarded->output.triangles[iso], 0xff, payload,
                             stream);
    if (status != cudaSuccess) {
      return SetCudaError("cudaMemsetAsync(triangle payload)", status, error);
    }
  }
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

void FreeGuardedOutput(GuardedOutput* guarded) {
  if (guarded == nullptr) {
    return;
  }
  for (unsigned char*& allocation : guarded->triangle_allocations) {
    if (allocation != nullptr) {
      cudaFree(allocation);
      allocation = nullptr;
    }
  }
  if (guarded->count_allocation != nullptr) {
    cudaFree(guarded->count_allocation);
  }
  *guarded = GuardedOutput{};
}

ValidationReport ValidateOutput(const GuardedOutput& candidate,
                                const GuardedOutput& reference,
                                int num_isovalues, double abs_tolerance,
                                double rel_tolerance, cudaStream_t stream) {
  ValidationReport report;
  if (num_isovalues <= 0 || num_isovalues > kMaxIsovalues ||
      candidate.output.triangle_counts == nullptr ||
      reference.output.triangle_counts == nullptr) {
    report.message = "invalid validation arguments";
    return report;
  }

  report.candidate_counts.resize(static_cast<std::size_t>(num_isovalues));
  report.reference_counts.resize(static_cast<std::size_t>(num_isovalues));
  const std::size_t count_bytes =
      static_cast<std::size_t>(num_isovalues) * sizeof(std::uint64_t);
  cudaError_t status = cudaMemcpyAsync(
      report.candidate_counts.data(), candidate.output.triangle_counts,
      count_bytes, cudaMemcpyDeviceToHost, stream);
  if (status == cudaSuccess) {
    status = cudaMemcpyAsync(report.reference_counts.data(),
                             reference.output.triangle_counts, count_bytes,
                             cudaMemcpyDeviceToHost, stream);
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  if (status != cudaSuccess) {
    report.message = std::string("copying triangle counts failed: ") +
                     cudaGetErrorString(status);
    return report;
  }

  report.counts_match = true;
  report.capacity_ok = true;
  for (int iso = 0; iso < num_isovalues; ++iso) {
    const std::uint64_t got = report.candidate_counts[iso];
    const std::uint64_t expected = report.reference_counts[iso];
    report.counts_match = report.counts_match && got == expected;
    report.capacity_ok =
        report.capacity_ok && got <= candidate.output.capacities[iso] &&
        expected <= reference.output.capacities[iso];
  }

  DeviceValidationStats* device_stats = nullptr;
  status = cudaMalloc(reinterpret_cast<void**>(&device_stats),
                      sizeof(DeviceValidationStats));
  if (status != cudaSuccess) {
    report.message = std::string("allocating validation stats failed: ") +
                     cudaGetErrorString(status);
    return report;
  }
  status = cudaMemsetAsync(device_stats, 0, sizeof(DeviceValidationStats),
                           stream);
  if (status != cudaSuccess) {
    cudaFree(device_stats);
    report.message = std::string("resetting validation stats failed: ") +
                     cudaGetErrorString(status);
    return report;
  }

  LaunchGuardCheck(candidate.count_allocation, candidate.count_payload_bytes,
                   device_stats, stream);
  LaunchGuardCheck(reference.count_allocation, reference.count_payload_bytes,
                   device_stats, stream);
  for (int iso = 0; iso < num_isovalues; ++iso) {
    LaunchGuardCheck(candidate.triangle_allocations[iso],
                     candidate.triangle_payload_bytes[iso], device_stats,
                     stream);
    LaunchGuardCheck(reference.triangle_allocations[iso],
                     reference.triangle_payload_bytes[iso], device_stats,
                     stream);

    constexpr int kThreads = 256;
    const std::uint64_t bounded_candidate_count =
        std::min(report.candidate_counts[iso],
                 candidate.output.capacities[iso]);
    const std::uint64_t bounded_reference_count =
        std::min(report.reference_counts[iso],
                 reference.output.capacities[iso]);
    for (int side = 0; side < 2; ++side) {
      const std::uint64_t triangle_count =
          side == 0 ? bounded_candidate_count : bounded_reference_count;
      const float* values = reinterpret_cast<const float*>(
          side == 0 ? candidate.output.triangles[iso]
                    : reference.output.triangles[iso]);
      const std::uint64_t value_count = triangle_count * 9ull;
      if (value_count != 0) {
        const std::uint64_t blocks =
            (value_count + static_cast<std::uint64_t>(kThreads) - 1ull) /
            static_cast<std::uint64_t>(kThreads);
        if (blocks > static_cast<std::uint64_t>(0x7fffffff)) {
          report.capacity_ok = false;
        } else {
          CheckFiniteKernel<<<static_cast<unsigned int>(blocks), kThreads, 0,
                              stream>>>(values, value_count, device_stats);
        }
      }
    }

    if (report.candidate_counts[iso] != report.reference_counts[iso] ||
        report.candidate_counts[iso] > candidate.output.capacities[iso] ||
        report.reference_counts[iso] > reference.output.capacities[iso]) {
      continue;
    }
    const std::uint64_t value_count = report.reference_counts[iso] * 9ull;
    if (value_count == 0) continue;
    const std::uint64_t blocks =
        (value_count + static_cast<std::uint64_t>(kThreads) - 1ull) /
        static_cast<std::uint64_t>(kThreads);
    if (blocks > static_cast<std::uint64_t>(0x7fffffff)) {
      report.capacity_ok = false;
      continue;
    }
    CompareTrianglesKernel<<<static_cast<unsigned int>(blocks), kThreads, 0,
                             stream>>>(
        reinterpret_cast<const float*>(candidate.output.triangles[iso]),
        reinterpret_cast<const float*>(reference.output.triangles[iso]),
        value_count, static_cast<float>(abs_tolerance),
        static_cast<float>(rel_tolerance), device_stats);
  }

  status = cudaGetLastError();
  DeviceValidationStats host_stats{};
  if (status == cudaSuccess) {
    status = cudaMemcpyAsync(&host_stats, device_stats,
                             sizeof(DeviceValidationStats),
                             cudaMemcpyDeviceToHost, stream);
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream);
  }
  cudaFree(device_stats);
  if (status != cudaSuccess) {
    report.message = std::string("GPU validation failed: ") +
                     cudaGetErrorString(status);
    return report;
  }

  report.mismatched_values = host_stats.mismatched_values;
  report.nonfinite_values = host_stats.nonfinite_values;
  report.guard_corruptions = host_stats.guard_corruptions;
  std::memcpy(&report.max_abs_error, &host_stats.max_abs_error_bits,
              sizeof(float));
  std::memcpy(&report.max_rel_error, &host_stats.max_rel_error_bits,
              sizeof(float));
  report.finite = report.nonfinite_values == 0;
  report.guards_intact = report.guard_corruptions == 0;
  report.values_within_tolerance =
      report.counts_match && report.capacity_ok &&
      report.mismatched_values == 0;
  report.correct = report.counts_match && report.capacity_ok && report.finite &&
                   report.guards_intact && report.values_within_tolerance;

  if (report.correct) {
    report.message = "ok";
  } else {
    std::ostringstream message;
    bool separator = false;
    auto append = [&](const std::string& item) {
      if (separator) {
        message << "; ";
      }
      message << item;
      separator = true;
    };
    if (!report.counts_match) append("triangle counts differ");
    if (!report.capacity_ok) append("reported count exceeds capacity");
    if (!report.finite) append("non-finite coordinate detected");
    if (!report.guards_intact) append("guard region corrupted");
    if (!report.values_within_tolerance) append("coordinate mismatch");
    report.message = message.str();
  }
  return report;
}

}  // namespace icecarver
