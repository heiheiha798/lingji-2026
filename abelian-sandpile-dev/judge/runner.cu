#include "cases.hpp"
#include "reference.hpp"
#include "sandpile_api.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void CheckCuda(cudaError_t error, const char *expression, const char *file,
               int line) {
  if (error == cudaSuccess) {
    return;
  }
  char message[1024];
  std::snprintf(message, sizeof(message), "%s:%d: %s failed: %s", file, line,
                expression, cudaGetErrorString(error));
  throw std::runtime_error(message);
}

#define CUDA_CHECK(expr) CheckCuda((expr), #expr, __FILE__, __LINE__)

using RunFn = int (*)(const std::uint32_t *, std::uint8_t *, std::uint64_t *,
                      int, int);

class SubmissionLibrary {
 public:
  explicit SubmissionLibrary(const char *path) {
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      throw std::runtime_error(std::string("cannot load submission library: ") +
                               dlerror());
    }
    run = reinterpret_cast<RunFn>(dlsym(handle_, "sandpile_run"));
    if (run == nullptr) {
      throw std::runtime_error(
          "submission library does not export sandpile_run");
    }
  }

  ~SubmissionLibrary() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  SubmissionLibrary(const SubmissionLibrary &) = delete;
  SubmissionLibrary &operator=(const SubmissionLibrary &) = delete;

  RunFn run = nullptr;

 private:
  void *handle_ = nullptr;
};

struct ReferenceResources {
  std::uint32_t *input = nullptr;
  std::uint8_t *stable = nullptr;
  std::uint64_t *odometer = nullptr;
  void *workspace = nullptr;
  cudaStream_t stream = nullptr;

  ~ReferenceResources() {
    cudaFree(workspace);
    cudaFree(odometer);
    cudaFree(stable);
    cudaFree(input);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

constexpr std::size_t kByteGuard = 4096;
constexpr std::size_t kWordGuard = 512;
constexpr std::uint8_t kBytePattern = 0xa5;
constexpr std::uint64_t kWordPattern = 0xa5a5a5a5a5a5a5a5ULL;

bool GuardsIntact(const std::vector<std::uint8_t> &stable,
                  const std::vector<std::uint64_t> &odometer,
                  std::size_t n) {
  for (std::size_t i = 0; i < kByteGuard; ++i) {
    if (stable[i] != kBytePattern || stable[kByteGuard + n + i] != kBytePattern) {
      return false;
    }
  }
  for (std::size_t i = 0; i < kWordGuard; ++i) {
    if (odometer[i] != kWordPattern ||
        odometer[kWordGuard + n + i] != kWordPattern) {
      return false;
    }
  }
  return true;
}

void CpuReference(const std::vector<std::uint32_t> &input, int rows, int cols,
                  std::vector<std::uint8_t> *stable,
                  std::vector<std::uint64_t> *odometer) {
  const std::size_t n = input.size();
  std::vector<std::uint64_t> height(input.begin(), input.end());
  odometer->assign(n, 0);
  std::vector<std::uint8_t> queued(n, 0);
  std::deque<std::uint32_t> work;
  for (std::size_t i = 0; i < n; ++i) {
    if (height[i] >= 4) {
      work.push_back(static_cast<std::uint32_t>(i));
      queued[i] = 1;
    }
  }
  auto maybe_enqueue = [&](std::uint32_t i) {
    if (height[i] >= 4 && queued[i] == 0) {
      queued[i] = 1;
      work.push_back(i);
    }
  };
  while (!work.empty()) {
    const std::uint32_t i = work.front();
    work.pop_front();
    queued[i] = 0;
    const std::uint64_t q = height[i] / 4;
    if (q == 0) {
      continue;
    }
    height[i] -= 4 * q;
    (*odometer)[i] += q;
    const int y = static_cast<int>(i / static_cast<std::uint32_t>(cols));
    const int x = static_cast<int>(i - static_cast<std::uint32_t>(y * cols));
    if (x > 0) {
      height[i - 1] += q;
      maybe_enqueue(i - 1);
    }
    if (x + 1 < cols) {
      height[i + 1] += q;
      maybe_enqueue(i + 1);
    }
    if (y > 0) {
      height[i - cols] += q;
      maybe_enqueue(i - cols);
    }
    if (y + 1 < rows) {
      height[i + cols] += q;
      maybe_enqueue(i + cols);
    }
    maybe_enqueue(i);
  }
  stable->resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    (*stable)[i] = static_cast<std::uint8_t>(height[i]);
  }
}

void RunReferenceTask(const std::vector<std::uint32_t> &input,
                      const CaseSpec &spec,
                      std::vector<std::uint8_t> *stable,
                      std::vector<std::uint64_t> *odometer,
                      std::uint64_t *iterations) {
  const std::size_t n = input.size();
  if (stable->size() != n || odometer->size() != n) {
    throw std::invalid_argument("reference host output has the wrong size");
  }
  ReferenceResources d;
  CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&d.input, n * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&d.stable, n * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d.odometer, n * sizeof(std::uint64_t)));
  CUDA_CHECK(cudaMalloc(&d.workspace,
                        ReferenceWorkspaceSize(spec.rows, spec.cols)));
  CUDA_CHECK(cudaMemcpyAsync(d.input, input.data(),
                             n * sizeof(std::uint32_t),
                             cudaMemcpyHostToDevice, d.stream));
  ReferenceSolve(d.input, d.stable, d.odometer, spec.rows, spec.cols,
                 d.workspace, d.stream, iterations);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(stable->data(), d.stable,
                             n * sizeof(std::uint8_t),
                             cudaMemcpyDeviceToHost, d.stream));
  CUDA_CHECK(cudaMemcpyAsync(odometer->data(), d.odometer,
                             n * sizeof(std::uint64_t),
                             cudaMemcpyDeviceToHost, d.stream));
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
}

double ElapsedMs(std::chrono::steady_clock::time_point start,
                 std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

CaseSpec SelectCase(int case_id) {
  const auto &cases = PublicCases();
  const int public_count = static_cast<int>(cases.size());
  if (case_id < 0 || case_id >= public_count + RandomCaseCount()) {
    throw std::invalid_argument("case id out of range");
  }
  if (case_id < public_count) {
    return cases[case_id];
  }
  const char *seed = std::getenv("SANDPILE_RANDOM_SEED");
  if (seed == nullptr) {
    throw std::invalid_argument("SANDPILE_RANDOM_SEED is required");
  }
  return RandomCase(std::stoull(seed), case_id - public_count);
}

void WriteOutput(const char *path, const std::uint8_t *stable,
                 const std::uint64_t *odometer, std::size_t n) {
  if (std::string(path) == "-") {
    return;
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create output file");
  }
  output.write(reinterpret_cast<const char *>(stable),
               static_cast<std::streamsize>(n));
  output.write(reinterpret_cast<const char *>(odometer),
               static_cast<std::streamsize>(n * sizeof(std::uint64_t)));
  if (!output) {
    throw std::runtime_error("cannot write output file");
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::fprintf(stderr,
                   "usage: %s CASE_ID SUBMISSION_LIBRARY candidate|reference OUTPUT\n",
                   argv[0]);
      return 2;
    }
    const int case_id = std::stoi(argv[1]);
    const std::string mode = argv[3];
    if (mode != "candidate" && mode != "reference") {
      throw std::invalid_argument("mode must be candidate or reference");
    }
    const CaseSpec spec = SelectCase(case_id);
    const std::vector<std::uint32_t> input = GenerateCase(spec);
    const std::size_t n = input.size();

    if (mode == "reference") {
      std::vector<std::uint8_t> stable;
      std::vector<std::uint64_t> odometer;
      stable.resize(n);
      odometer.resize(n);
      std::uint64_t iterations = 0;
      const auto start = std::chrono::steady_clock::now();
      RunReferenceTask(input, spec, &stable, &odometer, &iterations);
      const auto stop = std::chrono::steady_clock::now();
      if (case_id == 0 || case_id >= static_cast<int>(PublicCases().size())) {
        std::vector<std::uint8_t> cpu_stable;
        std::vector<std::uint64_t> cpu_odometer;
        CpuReference(input, spec.rows, spec.cols, &cpu_stable, &cpu_odometer);
        if (cpu_stable != stable || cpu_odometer != odometer) {
          throw std::runtime_error(
              "trusted GPU reference disagrees with CPU reference");
        }
      }
      WriteOutput(argv[4], stable.data(), odometer.data(), n);
      std::printf(
          "RESULT {\"mode\":\"reference\",\"case_id\":%d,"
          "\"name\":\"%s\",\"correct\":true,\"reference_ms\":%.6f,"
          "\"rows\":%d,\"cols\":%d,\"iterations\":%" PRIu64 ","
          "\"seed\":%" PRIu64 "}\n",
          case_id, spec.name, ElapsedMs(start, stop), spec.rows, spec.cols,
          iterations, spec.seed);
      return 0;
    }

    SubmissionLibrary submission(argv[2]);
    const std::vector<std::uint32_t> input_before = input;
    std::vector<std::uint8_t> guarded_stable(n + 2 * kByteGuard,
                                              kBytePattern);
    std::vector<std::uint64_t> guarded_odometer(n + 2 * kWordGuard,
                                                kWordPattern);
    std::uint8_t *stable = guarded_stable.data() + kByteGuard;
    std::uint64_t *odometer = guarded_odometer.data() + kWordGuard;

    const auto start = std::chrono::steady_clock::now();
    const int submission_status =
        submission.run(input.data(), stable, odometer, spec.rows, spec.cols);
    const auto stop = std::chrono::steady_clock::now();

    const std::vector<std::uint8_t> observed_stable(stable, stable + n);
    const std::vector<std::uint64_t> observed_odometer(odometer, odometer + n);
    const bool input_unchanged = input == input_before;
    const bool guards_intact = GuardsIntact(guarded_stable, guarded_odometer, n);

    const bool correct =
        submission_status == 0 && input_unchanged && guards_intact;
    WriteOutput(argv[4], observed_stable.data(), observed_odometer.data(), n);
    std::printf(
        "RESULT {\"mode\":\"candidate\",\"case_id\":%d,"
        "\"name\":\"%s\",\"correct\":%s,\"time_ms\":%.6f,"
        "\"submission_status\":%d,\"input_unchanged\":%s,"
        "\"guards_intact\":%s,\"rows\":%d,\"cols\":%d,"
        "\"seed\":%" PRIu64 "}\n",
        case_id, spec.name, correct ? "true" : "false",
        ElapsedMs(start, stop), submission_status,
        input_unchanged ? "true" : "false",
        guards_intact ? "true" : "false", spec.rows, spec.cols, spec.seed);
    return correct ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "judge error: %s\n", error.what());
    std::printf("RESULT {\"correct\":false,\"error\":\"judge failure\"}\n");
    return 2;
  }
}
