#include "cases.hpp"
#include "closure_api.h"
#include "reference.hpp"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

using RunFn = int (*)(const std::uint64_t *, std::uint64_t *, int, int);

class SubmissionLibrary {
 public:
  explicit SubmissionLibrary(const char *path) {
    handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      throw std::runtime_error(std::string("cannot load submission library: ") +
                               dlerror());
    }
    run = reinterpret_cast<RunFn>(dlsym(handle_, "closure_run"));
    if (run == nullptr) {
      throw std::runtime_error(
          "submission library does not export closure_run");
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
  std::uint64_t *adjacency = nullptr;
  std::uint64_t *reachability = nullptr;
  cudaStream_t stream = nullptr;

  ~ReferenceResources() {
    cudaFree(reachability);
    cudaFree(adjacency);
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

constexpr std::size_t kGuardWords = 512;
constexpr std::uint64_t kGuardPattern = 0xa5a5a5a5a5a5a5a5ULL;

std::vector<std::uint64_t> CpuReference(
    const std::vector<std::uint64_t> &adjacency, int vertices, int words) {
  std::vector<std::uint64_t> reach = adjacency;
  for (int row = 0; row < vertices; ++row) {
    reach[static_cast<std::size_t>(row) * words + row / 64] |=
        1ULL << (row & 63);
  }
  for (int pivot = 0; pivot < vertices; ++pivot) {
    const std::size_t pivot_base = static_cast<std::size_t>(pivot) * words;
    for (int row = 0; row < vertices; ++row) {
      const std::size_t base = static_cast<std::size_t>(row) * words;
      if ((reach[base + pivot / 64] & (1ULL << (pivot & 63))) == 0) {
        continue;
      }
      for (int word = 0; word < words; ++word) {
        reach[base + word] |= reach[pivot_base + word];
      }
    }
  }
  return reach;
}

void RunReferenceTask(const std::vector<std::uint64_t> &adjacency,
                      int vertices, int words,
                      std::vector<std::uint64_t> *reachability) {
  const std::size_t bytes = adjacency.size() * sizeof(std::uint64_t);
  if (reachability->size() != adjacency.size()) {
    throw std::invalid_argument("reference host output has the wrong size");
  }
  ReferenceResources d;
  CUDA_CHECK(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&d.adjacency, bytes));
  CUDA_CHECK(cudaMalloc(&d.reachability, bytes));
  CUDA_CHECK(cudaMemcpyAsync(d.adjacency, adjacency.data(), bytes,
                             cudaMemcpyHostToDevice, d.stream));
  ReferenceSolve(d.adjacency, d.reachability, vertices, words, d.stream);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(reachability->data(), d.reachability, bytes,
                             cudaMemcpyDeviceToHost, d.stream));
  CUDA_CHECK(cudaStreamSynchronize(d.stream));
}

bool GuardsIntact(const std::vector<std::uint64_t> &guarded,
                  std::size_t elements) {
  for (std::size_t i = 0; i < kGuardWords; ++i) {
    if (guarded[i] != kGuardPattern ||
        guarded[kGuardWords + elements + i] != kGuardPattern) {
      return false;
    }
  }
  return true;
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
  const char *seed = std::getenv("CLOSURE_RANDOM_SEED");
  if (seed == nullptr) {
    throw std::invalid_argument("CLOSURE_RANDOM_SEED is required");
  }
  return RandomCase(std::stoull(seed), case_id - public_count);
}

void WriteOutput(const char *path, const std::uint64_t *reachability,
                 std::size_t elements) {
  if (std::string(path) == "-") {
    return;
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create output file");
  }
  output.write(reinterpret_cast<const char *>(reachability),
               static_cast<std::streamsize>(elements * sizeof(std::uint64_t)));
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
    const int words = WordsPerRow(spec.vertices);
    const std::vector<std::uint64_t> adjacency = GenerateCase(spec);
    const std::size_t elements = adjacency.size();

    if (mode == "reference") {
      std::vector<std::uint64_t> reachability;
      reachability.resize(elements);
      const auto start = std::chrono::steady_clock::now();
      RunReferenceTask(adjacency, spec.vertices, words, &reachability);
      const auto stop = std::chrono::steady_clock::now();
      if (case_id == 0 || case_id >= static_cast<int>(PublicCases().size())) {
        if (CpuReference(adjacency, spec.vertices, words) != reachability) {
          throw std::runtime_error(
              "trusted GPU reference disagrees with CPU reference");
        }
      }
      WriteOutput(argv[4], reachability.data(), elements);
      std::printf(
          "RESULT {\"mode\":\"reference\",\"case_id\":%d,"
          "\"name\":\"%s\",\"correct\":true,\"reference_ms\":%.6f,"
          "\"vertices\":%d,\"words_per_row\":%d,\"seed\":%" PRIu64
          "}\n",
          case_id, spec.name, ElapsedMs(start, stop), spec.vertices, words,
          spec.seed);
      return 0;
    }

    SubmissionLibrary submission(argv[2]);
    const std::vector<std::uint64_t> adjacency_before = adjacency;
    std::vector<std::uint64_t> guarded(elements + 2 * kGuardWords,
                                       kGuardPattern);
    std::uint64_t *reachability = guarded.data() + kGuardWords;
    const auto start = std::chrono::steady_clock::now();
    const int submission_status = submission.run(
        adjacency.data(), reachability, spec.vertices, words);
    const auto stop = std::chrono::steady_clock::now();

    const std::vector<std::uint64_t> observed(reachability,
                                               reachability + elements);
    const bool input_unchanged = adjacency == adjacency_before;
    const bool guards_intact = GuardsIntact(guarded, elements);
    const bool correct =
        submission_status == 0 && input_unchanged && guards_intact;
    WriteOutput(argv[4], observed.data(), elements);
    std::printf(
        "RESULT {\"mode\":\"candidate\",\"case_id\":%d,"
        "\"name\":\"%s\",\"correct\":%s,\"time_ms\":%.6f,"
        "\"submission_status\":%d,\"input_unchanged\":%s,"
        "\"guards_intact\":%s,\"vertices\":%d,"
        "\"words_per_row\":%d,\"seed\":%" PRIu64 "}\n",
        case_id, spec.name, correct ? "true" : "false",
        ElapsedMs(start, stop), submission_status,
        input_unchanged ? "true" : "false",
        guards_intact ? "true" : "false", spec.vertices, words, spec.seed);
    return correct ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "judge error: %s\n", error.what());
    std::printf("RESULT {\"correct\":false,\"error\":\"judge failure\"}\n");
    return 2;
  }
}
