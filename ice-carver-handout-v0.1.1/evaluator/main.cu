#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "config_parser.h"
#include "generator/field_generator.h"
#include "icecarver/api.h"
#include "reference/cuda_reference.h"
#include "validator.h"

namespace {

using SolverFunction = int (*)(const icecarver::Input*, icecarver::Output*,
                               void*, std::size_t, cudaStream_t);

struct Options {
  std::string config_path;
  std::string solver;
  std::string output_path;
  bool worker = false;
};

struct Timing {
  double cuda_ms = 0.0;
  double wall_ms = 0.0;
  double official_ms = 0.0;
};

struct RunRecord {
  bool warmup = false;
  int index = 0;
  std::uint64_t seed = 0;
  int solver_status = icecarver::kCudaFailure;
  int reference_status = icecarver::kCudaFailure;
  bool timed_out = false;
  Timing timing;
  icecarver::ValidationReport validation;
};

struct RuntimeResources {
  float* volume = nullptr;
  float* isovalues = nullptr;
  void* workspace = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  icecarver::GuardedOutput candidate;
  icecarver::GuardedOutput reference;

  ~RuntimeResources() {
    icecarver::FreeGuardedOutput(&candidate);
    icecarver::FreeGuardedOutput(&reference);
    if (workspace != nullptr) cudaFree(workspace);
    if (isovalues != nullptr) cudaFree(isovalues);
    if (volume != nullptr) cudaFree(volume);
    if (start_event != nullptr) cudaEventDestroy(start_event);
    if (stop_event != nullptr) cudaEventDestroy(stop_event);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
};

std::string JsonEscape(const std::string& value) {
  std::ostringstream escaped;
  for (const unsigned char c : value) {
    switch (c) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (c < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<unsigned int>(c) << std::dec;
        } else {
          escaped << static_cast<char>(c);
        }
    }
  }
  return escaped.str();
}

const char* StatusName(int status) {
  switch (status) {
    case icecarver::kSuccess:
      return "success";
    case icecarver::kInvalidArgument:
      return "invalid_argument";
    case icecarver::kInsufficientWorkspace:
      return "insufficient_workspace";
    case icecarver::kInsufficientOutput:
      return "insufficient_output";
    case icecarver::kSizeOverflow:
      return "size_overflow";
    case icecarver::kCudaFailure:
      return "cuda_failure";
    default:
      return "unknown";
  }
}

bool ParseArguments(int argc, char** argv, Options* options,
                    std::string* error) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: icecarver_eval --config FILE "
                   "--solver public|target --output FILE\n";
      std::exit(0);
    }
    if (argument == "--worker") {
      options->worker = true;
      continue;
    }
    if (argument != "--config" && argument != "--solver" &&
        argument != "--output") {
      *error = "unknown argument: " + argument;
      return false;
    }
    if (i + 1 >= argc) {
      *error = "missing value after " + argument;
      return false;
    }
    const std::string value = argv[++i];
    if (argument == "--config") {
      options->config_path = value;
    } else if (argument == "--solver") {
      options->solver = value;
    } else {
      options->output_path = value;
    }
  }
  if (options->config_path.empty() || options->solver.empty() ||
      options->output_path.empty()) {
    *error = "--config, --solver, and --output are all required";
    return false;
  }
  if (options->solver != "public" && options->solver != "target") {
    *error = "--solver must be 'public' or 'target'";
    return false;
  }
  return true;
}

#if defined(__linux__)
int RunWithWatchdog(int argc, char** argv, double timeout_seconds) {
  const pid_t child = fork();
  if (child < 0) {
    std::cerr << "error: fork failed\n";
    return 70;
  }
  if (child == 0) {
    std::vector<char*> child_arguments;
    child_arguments.reserve(static_cast<std::size_t>(argc) + 2);
    for (int i = 0; i < argc; ++i) {
      child_arguments.push_back(argv[i]);
    }
    char worker_flag[] = "--worker";
    child_arguments.push_back(worker_flag);
    child_arguments.push_back(nullptr);
    execv("/proc/self/exe", child_arguments.data());
    _exit(127);
  }

  // timeout_seconds is the hard limit for the complete test point. A small
  // grace period covers process startup and failure-report flushing; a hung
  // CUDA context is then destroyed by killing the worker process.
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_seconds + 5.0);
  int status = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const pid_t observed = waitpid(child, &status, WNOHANG);
    if (observed == child) {
      if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
      }
      if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
      }
      return 70;
    }
    if (observed < 0) {
      std::cerr << "error: waitpid failed\n";
      return 70;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  kill(child, SIGKILL);
  (void)waitpid(child, &status, 0);
  std::cerr << "error: evaluator worker exceeded the " << timeout_seconds
            << " second test-point limit and was killed\n";
  return 124;
}
#endif

bool CudaOk(cudaError_t status, const char* operation, std::string* error) {
  if (status == cudaSuccess) {
    return true;
  }
  *error = std::string(operation) + " failed: " + cudaGetErrorString(status);
  return false;
}

bool InitializeResources(const icecarver::CaseConfig& config,
                         std::uint64_t sample_count,
                         std::size_t workspace_bytes,
                         RuntimeResources* resources, std::string* error) {
  if (!CudaOk(cudaStreamCreateWithFlags(&resources->stream,
                                        cudaStreamNonBlocking),
              "cudaStreamCreateWithFlags", error) ||
      !CudaOk(cudaEventCreate(&resources->start_event),
              "cudaEventCreate(start)", error) ||
      !CudaOk(cudaEventCreate(&resources->stop_event),
              "cudaEventCreate(stop)", error) ||
      !CudaOk(cudaMalloc(reinterpret_cast<void**>(&resources->volume),
                         static_cast<std::size_t>(sample_count) *
                             sizeof(float)),
              "cudaMalloc(volume)", error) ||
      !CudaOk(cudaMalloc(reinterpret_cast<void**>(&resources->isovalues),
                         config.isovalues.size() * sizeof(float)),
              "cudaMalloc(isovalues)", error) ||
      !CudaOk(cudaMalloc(&resources->workspace, workspace_bytes),
              "cudaMalloc(workspace)", error)) {
    return false;
  }

  if (!CudaOk(cudaMemcpyAsync(resources->isovalues, config.isovalues.data(),
                              config.isovalues.size() * sizeof(float),
                              cudaMemcpyHostToDevice, resources->stream),
              "copy isovalues", error) ||
      !CudaOk(cudaStreamSynchronize(resources->stream),
              "synchronize isovalues", error)) {
    return false;
  }

  if (!icecarver::AllocateGuardedOutput(
          static_cast<int>(config.isovalues.size()), config.per_iso_capacity,
          &resources->candidate, error)) {
    return false;
  }
  return true;
}

bool InputDescriptorMatches(const icecarver::Input& expected,
                            const icecarver::Input& observed) {
  return expected.nx == observed.nx && expected.ny == observed.ny &&
         expected.nz == observed.nz &&
         expected.num_isovalues == observed.num_isovalues &&
         expected.volume == observed.volume &&
         expected.isovalues == observed.isovalues &&
         expected.emit_triangles == observed.emit_triangles;
}

bool OutputDescriptorMatches(const icecarver::Output& expected,
                             const icecarver::Output& observed) {
  if (expected.triangle_counts != observed.triangle_counts) {
    return false;
  }
  for (int iso = 0; iso < icecarver::kMaxIsovalues; ++iso) {
    if (expected.capacities[iso] != observed.capacities[iso] ||
        expected.triangles[iso] != observed.triangles[iso]) {
      return false;
    }
  }
  return true;
}

bool TimeSolver(SolverFunction solver, const icecarver::Input& input,
                icecarver::Output* output, void* workspace,
                std::size_t workspace_bytes, RuntimeResources* resources,
                int* solver_status, Timing* timing, std::string* error) {
  if (!CudaOk(cudaMemsetAsync(workspace, 0, workspace_bytes,
                              resources->stream),
              "reset candidate workspace", error) ||
      !CudaOk(cudaStreamSynchronize(resources->stream),
              "synchronize candidate reset", error)) {
    return false;
  }

  const auto wall_start = std::chrono::steady_clock::now();
  cudaError_t cuda_status =
      cudaEventRecord(resources->start_event, resources->stream);
  if (cuda_status != cudaSuccess) {
    return CudaOk(cuda_status, "cudaEventRecord(start)", error);
  }
  *solver_status =
      solver(&input, output, workspace, workspace_bytes, resources->stream);
  cuda_status = cudaEventRecord(resources->stop_event, resources->stream);
  if (cuda_status != cudaSuccess) {
    return CudaOk(cuda_status, "cudaEventRecord(stop)", error);
  }

  // Device-wide synchronization deliberately catches work placed on streams
  // other than the supplied stream. The official time is the larger of this
  // wall clock and the supplied-stream CUDA event interval.
  cuda_status = cudaDeviceSynchronize();
  const auto wall_stop = std::chrono::steady_clock::now();
  timing->wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
  if (cuda_status != cudaSuccess) {
    return CudaOk(cuda_status, "cudaDeviceSynchronize(candidate)", error);
  }

  float cuda_ms = 0.0f;
  if (!CudaOk(cudaEventElapsedTime(&cuda_ms, resources->start_event,
                                   resources->stop_event),
              "cudaEventElapsedTime", error)) {
    return false;
  }
  timing->cuda_ms = static_cast<double>(cuda_ms);
  timing->official_ms = std::max(timing->cuda_ms, timing->wall_ms);
  return true;
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}

bool WriteResult(const std::string& path, const Options& options,
                 const icecarver::CaseConfig& config,
                 const cudaDeviceProp& properties, int driver_version,
                 int runtime_version, std::size_t workspace_bytes,
                 const std::vector<RunRecord>& runs, bool case_correct,
                 double median_ms, std::string* error) {
  const std::filesystem::path output_path(path);
  if (!output_path.parent_path().empty()) {
    std::error_code directory_error;
    std::filesystem::create_directories(output_path.parent_path(),
                                        directory_error);
    if (directory_error) {
      *error = "cannot create output directory: " + directory_error.message();
      return false;
    }
  }
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    *error = "cannot open output file: " + path;
    return false;
  }

  output << std::setprecision(10);
  output << "{\n"
         << "  \"schema_version\": \"1.0\",\n"
         << "  \"problem\": \"ice-carver\",\n"
         << "  \"solver\": \"" << JsonEscape(options.solver) << "\",\n"
         << "  \"environment\": {\n"
         << "    \"gpu_name\": \"" << JsonEscape(properties.name)
         << "\",\n"
         << "    \"compute_capability\": \"" << properties.major << "."
         << properties.minor << "\",\n"
         << "    \"global_memory_bytes\": " << properties.totalGlobalMem
         << ",\n"
         << "    \"driver_version\": " << driver_version << ",\n"
         << "    \"cuda_runtime_version\": " << runtime_version << "\n"
         << "  },\n"
         << "  \"case\": {\n"
         << "    \"id\": \"" << JsonEscape(config.id) << "\",\n"
         << "    \"field\": \"" << icecarver::FieldKindName(config.field)
         << "\",\n"
         << "    \"shape\": [" << config.nx << ", " << config.ny << ", "
         << config.nz << "],\n"
         << "    \"num_isovalues\": " << config.isovalues.size() << ",\n"
         << "    \"isovalues\": [";
  for (std::size_t i = 0; i < config.isovalues.size(); ++i) {
    if (i != 0) output << ", ";
    output << config.isovalues[i];
  }
  output << "],\n"
         << "    \"base_seed\": \"" << config.seed << "\",\n"
         << "    \"per_iso_capacity\": " << config.per_iso_capacity
         << ",\n"
         << "    \"workspace_bytes\": " << workspace_bytes << ",\n"
         << "    \"warmup_runs\": " << config.warmup_runs << ",\n"
         << "    \"measure_runs\": " << config.measure_runs << ",\n"
         << "    \"abs_tolerance\": " << config.abs_tolerance << ",\n"
         << "    \"rel_tolerance\": " << config.rel_tolerance << ",\n"
         << "    \"timeout_seconds\": " << config.timeout_seconds << ",\n"
         << "    \"correctness_weight\": " << config.correctness_weight
         << ",\n"
         << "    \"performance_weight\": " << config.performance_weight
         << ",\n"
         << "    \"correct\": " << (case_correct ? "true" : "false")
         << ",\n"
         << "    \"median_time_ms\": ";
  if (std::isfinite(median_ms)) {
    output << median_ms;
  } else {
    output << "null";
  }
  output << ",\n    \"runs\": [\n";

  for (std::size_t run_index = 0; run_index < runs.size(); ++run_index) {
    const RunRecord& run = runs[run_index];
    output << "      {\n"
           << "        \"phase\": \""
           << (run.warmup ? "warmup" : "measure") << "\",\n"
           << "        \"index\": " << run.index << ",\n"
           << "        \"seed\": \"" << run.seed << "\",\n"
           << "        \"solver_status\": " << run.solver_status << ",\n"
           << "        \"solver_status_name\": \""
           << StatusName(run.solver_status) << "\",\n"
           << "        \"reference_status\": " << run.reference_status
           << ",\n"
           << "        \"reference_status_name\": \""
           << StatusName(run.reference_status) << "\",\n"
           << "        \"cuda_time_ms\": " << run.timing.cuda_ms << ",\n"
           << "        \"wall_time_ms\": " << run.timing.wall_ms << ",\n"
           << "        \"official_time_ms\": " << run.timing.official_ms
           << ",\n"
           << "        \"timed_out\": "
           << (run.timed_out ? "true" : "false") << ",\n"
           << "        \"validation\": {\n"
           << "          \"correct\": "
           << (run.validation.correct ? "true" : "false") << ",\n"
           << "          \"counts_match\": "
           << (run.validation.counts_match ? "true" : "false") << ",\n"
           << "          \"capacity_ok\": "
           << (run.validation.capacity_ok ? "true" : "false") << ",\n"
           << "          \"finite\": "
           << (run.validation.finite ? "true" : "false") << ",\n"
           << "          \"guards_intact\": "
           << (run.validation.guards_intact ? "true" : "false") << ",\n"
           << "          \"values_within_tolerance\": "
           << (run.validation.values_within_tolerance ? "true" : "false")
           << ",\n"
           << "          \"mismatched_values\": "
           << run.validation.mismatched_values << ",\n"
           << "          \"nonfinite_values\": "
           << run.validation.nonfinite_values << ",\n"
           << "          \"guard_corruptions\": "
           << run.validation.guard_corruptions << ",\n"
           << "          \"max_abs_error\": "
           << run.validation.max_abs_error << ",\n"
           << "          \"max_rel_error\": "
           << run.validation.max_rel_error << ",\n"
           << "          \"candidate_counts\": [";
    for (std::size_t i = 0; i < run.validation.candidate_counts.size(); ++i) {
      if (i != 0) output << ", ";
      output << run.validation.candidate_counts[i];
    }
    output << "],\n          \"reference_counts\": [";
    for (std::size_t i = 0; i < run.validation.reference_counts.size(); ++i) {
      if (i != 0) output << ", ";
      output << run.validation.reference_counts[i];
    }
    output << "],\n"
           << "          \"message\": \""
           << JsonEscape(run.validation.message) << "\"\n"
           << "        }\n"
           << "      }" << (run_index + 1 == runs.size() ? "\n" : ",\n");
  }
  output << "    ]\n  }\n}\n";
  if (!output) {
    *error = "failed while writing output file: " + path;
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string error;
  if (!ParseArguments(argc, argv, &options, &error)) {
    std::cerr << "error: " << error << "\n"
              << "Usage: icecarver_eval --config FILE "
                 "--solver public|target --output FILE\n";
    return 64;
  }

#if defined(__linux__)
  if (!options.worker) {
    icecarver::CaseConfig watchdog_config;
    if (!icecarver::LoadCaseConfig(options.config_path, &watchdog_config,
                                   &error)) {
      std::cerr << "error: invalid case config: " << error << "\n";
      return 65;
    }
    return RunWithWatchdog(argc, argv, watchdog_config.timeout_seconds);
  }
#endif

  SolverFunction candidate_solver = nullptr;
  if (options.solver == "public") {
    candidate_solver = &icecarver_solve;
  } else {
#ifdef ICECARVER_ENABLE_TARGET
    candidate_solver = &icecarver_target_solve;
#else
    std::cerr << "error: --solver target requires a build configured with "
                 "-DICECARVER_ENABLE_TARGET=ON\n";
    return 64;
#endif
  }

  icecarver::CaseConfig config;
  if (!icecarver::LoadCaseConfig(options.config_path, &config, &error)) {
    std::cerr << "error: invalid case config: " << error << "\n";
    return 65;
  }

  const std::uint64_t sample_count =
      static_cast<std::uint64_t>(config.nx) *
      static_cast<std::uint64_t>(config.ny) *
      static_cast<std::uint64_t>(config.nz);
  constexpr std::uint64_t kWorkspaceReserve = 256ull * 1024ull * 1024ull;
  if (sample_count >
      (static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) -
       kWorkspaceReserve) /
          (2ull * sizeof(float))) {
    std::cerr << "error: workspace size overflows size_t\n";
    return 65;
  }
  const std::size_t workspace_bytes = static_cast<std::size_t>(
      2ull * sample_count * sizeof(float) + kWorkspaceReserve);

  RuntimeResources resources;
  if (!InitializeResources(config, sample_count, workspace_bytes, &resources,
                           &error)) {
    std::cerr << "error: resource initialization failed: " << error << "\n";
    return 70;
  }

  int device = 0;
  cudaDeviceProp properties{};
  int driver_version = 0;
  int runtime_version = 0;
  if (!CudaOk(cudaGetDevice(&device), "cudaGetDevice", &error) ||
      !CudaOk(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties", &error) ||
      !CudaOk(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion",
              &error) ||
      !CudaOk(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion",
              &error)) {
    std::cerr << "error: device inspection failed: " << error << "\n";
    return 70;
  }

  icecarver::Input solver_input{};
  solver_input.nx = config.nx;
  solver_input.ny = config.ny;
  solver_input.nz = config.nz;
  solver_input.num_isovalues = static_cast<int>(config.isovalues.size());
  solver_input.volume = resources.volume;
  solver_input.isovalues = resources.isovalues;
  solver_input.emit_triangles = 1;

  std::vector<RunRecord> runs;
  std::vector<double> measured_times;
  bool case_correct = true;
  const int total_runs = config.warmup_runs + config.measure_runs;
  for (int variant = 0; variant < total_runs; ++variant) {
    // A trusted output allocation must not exist while contestant code runs.
    // Besides reducing the exposed attack surface, this prevents a submitted
    // solver from aliasing the future oracle descriptor via host pointer
    // arithmetic. Allocate it only after the candidate has fully quiesced.
    icecarver::FreeGuardedOutput(&resources.reference);

    RunRecord record;
    record.warmup = variant < config.warmup_runs;
    record.index = record.warmup ? variant : variant - config.warmup_runs;
    record.seed = icecarver::DeriveVariantSeed(
        config.seed, static_cast<std::uint64_t>(variant));

    if (!CudaOk(icecarver::GenerateField(config, record.seed, resources.volume,
                                         resources.stream),
                "GenerateField", &error) ||
        !icecarver::ResetGuardedOutput(
            solver_input.num_isovalues, &resources.candidate, resources.stream,
            &error) ||
        !CudaOk(cudaStreamSynchronize(resources.stream),
                "synchronize generated input", &error)) {
      std::cerr << "error: preparing run " << variant << " failed: " << error
                << "\n";
      return 70;
    }

    // Pass disposable descriptor copies into contestant code. The allocator
    // metadata in GuardedOutput and the canonical input descriptor must never
    // be directly adjacent to, or aliased by, an untrusted host pointer.
    const icecarver::Input expected_input = solver_input;
    const icecarver::Output expected_output = resources.candidate.output;
    icecarver::Input candidate_input = expected_input;
    icecarver::Output candidate_output = expected_output;
    if (!TimeSolver(candidate_solver, candidate_input, &candidate_output,
                    resources.workspace,
                    workspace_bytes, &resources, &record.solver_status,
                    &record.timing, &error)) {
      record.validation.message = error;
      case_correct = false;
      runs.push_back(record);
      break;
    }

    if (!InputDescriptorMatches(expected_input, candidate_input) ||
        !OutputDescriptorMatches(expected_output, candidate_output)) {
      record.solver_status = icecarver::kInvalidArgument;
      record.validation.message =
          "solver modified a host input/output descriptor";
      case_correct = false;
      runs.push_back(record);
      break;
    }
    record.timed_out =
        record.timing.official_ms > config.timeout_seconds * 1000.0;

    // Restore all read-only inputs before invoking the oracle. A candidate is
    // allowed to read d_volume/d_isovalues but never to modify them; rebuilding
    // them here makes such a modification observable rather than self-validating.
    if (!CudaOk(icecarver::GenerateField(config, record.seed, resources.volume,
                                         resources.stream),
                "restore generated field", &error) ||
        !CudaOk(cudaMemcpyAsync(resources.isovalues, config.isovalues.data(),
                                config.isovalues.size() * sizeof(float),
                                cudaMemcpyHostToDevice, resources.stream),
                "restore isovalues", &error) ||
        !icecarver::AllocateGuardedOutput(
            solver_input.num_isovalues, config.per_iso_capacity,
            &resources.reference, &error) ||
        !icecarver::ResetGuardedOutput(
            solver_input.num_isovalues, &resources.reference,
            resources.stream, &error) ||
        !CudaOk(cudaMemsetAsync(resources.workspace, 0, workspace_bytes,
                                resources.stream),
                "reset reference workspace", &error) ||
        !CudaOk(cudaStreamSynchronize(resources.stream),
                "synchronize restored oracle input", &error)) {
      record.validation.message = error;
      case_correct = false;
      runs.push_back(record);
      break;
    }
    record.reference_status = icecarver_reference_solve(
        &solver_input, &resources.reference.output, resources.workspace,
        workspace_bytes, resources.stream);
    const cudaError_t reference_sync = cudaDeviceSynchronize();
    if (reference_sync != cudaSuccess) {
      record.validation.message =
          std::string("trusted reference CUDA failure: ") +
          cudaGetErrorString(reference_sync);
    } else if (record.solver_status == icecarver::kSuccess &&
               record.reference_status == icecarver::kSuccess) {
      record.validation = icecarver::ValidateOutput(
          resources.candidate, resources.reference,
          solver_input.num_isovalues, config.abs_tolerance,
          config.rel_tolerance, resources.stream);
    } else {
      std::ostringstream message;
      message << "solver status " << StatusName(record.solver_status)
              << ", reference status " << StatusName(record.reference_status);
      record.validation.message = message.str();
    }

    const bool run_correct =
        record.solver_status == icecarver::kSuccess &&
        record.reference_status == icecarver::kSuccess &&
        reference_sync == cudaSuccess && !record.timed_out &&
        record.validation.correct;
    case_correct = case_correct && run_correct;
    if (!record.warmup) {
      measured_times.push_back(record.timing.official_ms);
    }
    icecarver::FreeGuardedOutput(&resources.reference);
    runs.push_back(std::move(record));
  }

  if (runs.size() != static_cast<std::size_t>(total_runs) ||
      measured_times.size() != static_cast<std::size_t>(config.measure_runs)) {
    case_correct = false;
  }
  const double median_ms = Median(measured_times);
  if (!WriteResult(options.output_path, options, config, properties,
                   driver_version, runtime_version, workspace_bytes, runs,
                   case_correct, median_ms, &error)) {
    std::cerr << "error: " << error << "\n";
    return 74;
  }

  std::cout << config.id << ": " << (case_correct ? "PASS" : "FAIL")
            << ", median " << median_ms << " ms, result "
            << options.output_path << "\n";
  return case_correct ? 0 : 2;
}
